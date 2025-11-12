"""Kimi provider implementation with special handling for reasoning content."""

import json
import time
from typing import AsyncIterator, Dict, Any, Optional, Union
import httpx

from .base import BaseProvider, ProviderRequest, ProviderResponse
from ..models import ChatCompletionResponse, ChatCompletionChunk
from ..utils.tool_call import ToolCallManager
from ..utils.sse import SSEParser
from ..utils.json_utils import JSONProcessor
from ..utils.error import ProviderError, ParseError


class KimiProvider(BaseProvider):
    """Kimi-specific provider with reasoning content support."""
    
    def __init__(self, settings: Any) -> None:
        super().__init__("kimi", settings)
        self.tool_call_manager = ToolCallManager(settings)
        
    async def chat_completions(
        self, 
        request: ProviderRequest
    ) -> AsyncIterator[Union[ProviderResponse, ChatCompletionResponse, ChatCompletionChunk]]:
        """Handle Kimi chat completions with reasoning parsing."""
        
        url = self._get_upstream_url("chat/completions", request)
        is_stream = self.detect_stream_mode(request)
        
        # Transform request
        transformed = self._transform_chat_request(request, is_stream)
        
        headers = self.build_headers()
        
        if is_stream:
            async for chunk in self._stream_request(url, headers, transformed):
                yield chunk
        else:
            response = await self._non_stream_request(url, headers, transformed)
            yield response
    
    async def completions(
        self, 
        request: ProviderRequest
    ) -> AsyncIterator[Union[ProviderResponse, ChatCompletionResponse, ChatCompletionChunk]]:
        """Handle text completions."""
        
        url = self._get_upstream_url("completions", request)
        headers = self.build_headers()
        
        if request.get("stream", False):
            async for chunk in self._stream_request(url, headers, request.data):
                yield chunk
        else:
            response = await self._non_stream_request(url, headers, request.data)
            yield response
    
    def _transform_chat_request(self, request: ProviderRequest, stream: bool) -> Dict[str, Any]:
        """Transform chat request for Kimi API."""
        data = request.data.copy()
        
        # Ensure model is set
        model = self.get_model_name(request)
        data["model"] = model
        
        # Ensure streaming if configured
        if self.settings.ensure_stream and stream:
            data["stream"] = True
        
        # Apply overrides
        if self.settings.override_temperature is not None:
            data["temperature"] = self.settings.override_temperature
        if self.settings.override_max_tokens is not None:
            data["max_tokens"] = self.settings.override_max_tokens
        
        # Strip empty values
        data = {k: v for k, v in data.items() if v is not None}
        
        return data
    
    async def _stream_request(
        self, 
        url: str, 
        headers: Dict[str, str], 
        data: Dict[str, Any]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Make streaming request to Kimi API."""
        
        self.tool_call_manager.reset()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                async with client.stream("POST", url, headers=headers, json=data) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise ProviderError(
                            f"Kimi API error: {response.status_code} {body.decode()}",
                            "kimi",
                            response.status_code
                        )
                    
                    sse_parser = SSEParser()
                    
                    async for chunk_bytes in response.aiter_bytes():
                        chunk_text = chunk_bytes.decode("utf-8", errors="replace")
                        
                        for event in sse_parser.feed(chunk_text):
                            chunk = self._parse_stream_chunk(event)
                            if chunk:
                                yield chunk
                    # Flush any remaining buffered SSE fragment if upstream ended without blank line
                    try:
                        for event in sse_parser.flush():
                            chunk = self._parse_stream_chunk(event)
                            if chunk:
                                yield chunk
                    except Exception:
                        pass

                    # Check if we need to emit finish
                    if self.tool_call_manager.should_emit_finish():
                        yield self._create_finish_chunk()
            
            except httpx.RequestError as e:
                raise ProviderError(f"Request error: {str(e)}", "kimi", 502)
    
    async def _non_stream_request(
        self, 
        url: str, 
        headers: Dict[str, str], 
        data: Dict[str, Any]
    ) -> ChatCompletionResponse:
        """Make non-streaming request to Kimi API."""
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, headers=headers, json=data)
                
                if response.status_code != 200:
                    raise ProviderError(
                        f"Kimi API error: {response.status_code} {response.text}",
                        "kimi",
                        response.status_code
                    )
                
                response_data = response.json()
                
                # Reconstruct tool calls from reasoning if needed
                if "choices" in response_data:
                    for choice in response_data["choices"]:
                        if ("message" in choice and 
                            "reasoning_content" in choice["message"] and
                            "tool_calls" not in choice["message"]):
                            
                            reasoning = choice["message"]["reasoning_content"]
                            tool_calls = self._synthesize_tool_calls(reasoning)
                            if tool_calls:
                                choice["message"]["tool_calls"] = tool_calls
                                choice["finish_reason"] = "tool_calls"
                
                return ChatCompletionResponse(**response_data)
            
            except (httpx.RequestError, json.JSONDecodeError) as e:
                raise ProviderError(f"Request error: {str(e)}", "kimi", 502)
    
    def _parse_stream_chunk(self, event: Any) -> Optional[ChatCompletionChunk]:
        """Parse SSE event into normalized chunk."""
        
        if not event.data or event.data.strip() == "[DONE]":
            return None
        
        try:
            data = JSONProcessor.parse_safe(event.data)
            if not data or "choices" not in data:
                return None
            
            # Process each choice delta
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                
                # Handle reasoning content
                if "reasoning_content" in delta:
                    processed = self.tool_call_manager.process_upstream_delta(delta)
                    if "tool_calls" in processed:
                        choice["delta"]["tool_calls"] = processed["tool_calls"]
                    if "finish_reason" in processed:
                        choice["finish_reason"] = processed["finish_reason"]
                
                # Process tool_calls delta from provider
                elif "tool_calls" in delta:
                    deltas = self.tool_call_manager.process_upstream_delta(delta)
                    if "tool_calls" in deltas:
                        choice["delta"]["tool_calls"] = deltas["tool_calls"]
            
            # Clean Kimi-specific fields
            data = self._strip_kimi_fields(data)
            
            return ChatCompletionChunk(**data)
        
        except Exception as e:
            # Log error but continue streaming
            print(f"Error parsing chunk: {e}", file=sys.stderr)
            return None
    
    def _create_finish_chunk(self) -> ChatCompletionChunk:
        """Create a finish chunk for tool calls."""
        tool_calls, _ = self.tool_call_manager.finalize()
        
        return ChatCompletionChunk(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=self.settings.default_model or "unknown",
            choices=[{
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }]
        )
    
    def _synthesize_tool_calls(self, reasoning: str) -> Optional[list]:
        """Synthesize tool calls from reasoning content."""
        if not reasoning:
            return None
        
        # Simple extraction for common patterns
        tool_calls = []
        
        # Look for JSON objects that might be tool arguments
        try:
            # Try to find function calls in reasoning
            import re
            json_pattern = r'\{[^}]*(?:command|name|explanation)[^}]*\}'
            matches = re.findall(json_pattern, reasoning, re.IGNORECASE)
            
            for i, match in enumerate(matches):
                if JSONProcessor.is_valid_json_object(match):
                    tool_calls.append({
                        "type": "function",
                        "function": {
                            "name": "run_notebook_cell",
                            "arguments": match
                        }
                    })
        except Exception:
            pass
        
        return tool_calls if tool_calls else None
    
    def _strip_kimi_fields(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Remove Kimi-specific fields."""
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                if k not in {"reasoning_content", "reasoning_tokens", "thinking", "k2_metadata"}:
                    cleaned[k] = self._strip_kimi_fields(v)
            return cleaned
        elif isinstance(obj, list):
            return [self._strip_kimi_fields(x) for x in obj]
        return obj
    
    def _get_upstream_url(self, endpoint: str, request: ProviderRequest) -> str:
        """Build upstream URL."""
        model = self.get_model_name(request)
        return self.settings.get_upstream_url(f"{endpoint}", model)
