"""Chutes provider implementation."""

import json
import sys
import time
from typing import AsyncIterator, Dict, Any, Optional, Union
import httpx
from typing import Dict
import asyncio

from .base import BaseProvider, ProviderRequest, ProviderResponse
from ..models import ChatCompletionResponse, ChatCompletionChunk
from ..utils.sse import SSEParser
from ..utils.json_utils import JSONProcessor
from ..utils.error import ProviderError
from ..streaming import ToolCallBuilder, process_delta, fix_delta


class ChutesProvider(BaseProvider):
    """Chutes provider with standard OpenAI compatibility."""
    
    def __init__(self, settings: Any) -> None:
        super().__init__("chutes", settings)
        self._tool_call_accumulator: Dict[int, ToolCallBuilder] = {}
        
    async def chat_completions(
        self, 
        request: ProviderRequest
    ) -> AsyncIterator[Union[ProviderResponse, ChatCompletionResponse, ChatCompletionChunk]]:
        """Handle chat completions."""
        
        url = self._get_upstream_url("chat/completions", request)
        is_stream = self.detect_stream_mode(request)
        
        # Transform request
        transformed = self._transform_request(request, is_stream)
        
        # USE THE HEADERS FROM THE REQUEST (set in main.py), not build_headers()
        headers = request.headers or self.build_headers()
        
        # DEBUG LOGGING
        import sys
        print(f"\n[CHUTES DEBUG] URL: {url}", file=sys.stderr)
        print(f"[CHUTES DEBUG] Headers: {json.dumps({k: v[:50] if k.lower() == 'authorization' else v for k, v in headers.items()}, indent=2)}", file=sys.stderr)
        print(f"[CHUTES DEBUG] Request data: {json.dumps(transformed, default=str, indent=2)}", file=sys.stderr)
        
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
        headers = request.headers or self.build_headers()
        
        if request.get("stream", False):
            async for chunk in self._stream_request(url, headers, request.data):
                yield chunk
        else:
            response = await self._non_stream_request(url, headers, request.data)
            yield response
    
    def _transform_request(self, request: ProviderRequest, stream: bool) -> Dict[str, Any]:
        """Transform request for Chutes API."""
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
        """Make streaming request to Chutes API with retry logic."""
        
        import sys
        import asyncio
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"[CHUTES STREAM REQUEST] Starting", file=sys.stderr)
        print(f"[CHUTES STREAM REQUEST] URL: {url}", file=sys.stderr)
        print(f"[CHUTES STREAM REQUEST] Timeout: 60.0s", file=sys.stderr)
        print(f"[CHUTES STREAM REQUEST] Data keys: {list(data.keys())}", file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        
        max_attempts = 3
        initial_delay = 1.0
        backoff_factor = 2.0
        timeout_seconds = 60.0
        
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                print(f"[RETRY] Attempt {attempt + 1}/{max_attempts}", file=sys.stderr)
                
                # Reset accumulator for each retry
                self._tool_call_accumulator = {}
                sse_parser = SSEParser()
                chunk_count = 0
                stream_complete = False
                
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    async with client.stream("POST", url, headers=headers, json=data) as response:
                        if response.status_code != 200:
                            body = await response.aread()
                            # Don't retry on 4xx errors
                            if 400 <= response.status_code < 500:
                                raise ProviderError(
                                    f"Chutes API error: {response.status_code} {body.decode()}",
                                    "chutes",
                                    response.status_code
                                )
                            # 5xx errors are retryable
                            raise ProviderError(
                                f"Chutes API error: {response.status_code} {body.decode()}",
                                "chutes",
                                response.status_code
                            )
                        
                        print(f"DEBUG: Response status {response.status_code}", file=sys.stderr)
                        
                        try:
                            async for chunk_bytes in response.aiter_bytes():
                                chunk_text = chunk_bytes.decode("utf-8", errors="replace")
                                print(f"DEBUG: Raw chunk {chunk_count}: {chunk_text}", file=sys.stderr)
                                
                                for event in sse_parser.feed(chunk_text):
                                    print(f"DEBUG: SSE event: {event.event or 'message'} with data: {event.data[:100] if event.data else 'none'}", file=sys.stderr)
                                    parsed = self._parse_stream_chunk(event)
                                    if parsed:
                                        print(f"DEBUG: Yielding parsed chunk {chunk_count}", file=sys.stderr)
                                        yield parsed
                                chunk_count += 1
                            
                            # If we reach here, the stream completed successfully
                            # Flush any remaining buffered SSE fragment (no trailing blank line)
                            try:
                                for event in sse_parser.flush():
                                    print(f"DEBUG: SSE event (flush): {event.event or 'message'} with data: {event.data[:100] if event.data else 'none'}", file=sys.stderr)
                                    parsed = self._parse_stream_chunk(event)
                                    if parsed:
                                        print(f"DEBUG: Yielding parsed chunk {chunk_count} (flush)", file=sys.stderr)
                                        yield parsed
                            except Exception as _:
                                pass
                            stream_complete = True
                            print(f"[RETRY] Stream completed successfully", file=sys.stderr)
                            return
                        
                        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout) as timeout_error:
                            # Check if we got any chunks at all
                            if chunk_count > 0:
                                print(f"[TIMEOUT] Timeout after receiving {chunk_count} chunks", file=sys.stderr)
                                # We got something, don't retry - let upstream handle the partial result
                                raise timeout_error
                            else:
                                print(f"[TIMEOUT] Timeout without receiving any chunks", file=sys.stderr)
                                raise
                        
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout) as e:
                last_exception = e
                print(f"[RETRY] Timeout error on attempt {attempt + 1}: {str(e)}", file=sys.stderr)
                
                if attempt < max_attempts - 1:
                    # Calculate delay with exponential backoff
                    delay = min(initial_delay * (backoff_factor ** attempt), 30.0)
                    print(f"[RETRY] Waiting {delay:.2f}s before retry...", file=sys.stderr)
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(f"[RETRY] Max attempts reached, giving up", file=sys.stderr)
                    break
            
            except ProviderError as e:
                # Only retry on 5xx errors, not 4xx
                if e.status_code and 500 <= e.status_code < 600:
                    print(f"[RETRY] 5xx error, retryable: {str(e)}", file=sys.stderr)
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = min(initial_delay * (backoff_factor ** attempt), 30.0)
                        print(f"[RETRY] Waiting {delay:.2f}s before retry...", file=sys.stderr)
                        await asyncio.sleep(delay)
                        continue
                else:
                    # Non-retryable error
                    print(f"[RETRY] Non-retryable error [{e.status_code}]: {str(e)}", file=sys.stderr)
                    raise
            
            except Exception as e:
                last_exception = e
                print(f"[RETRY] Unexpected error on attempt {attempt + 1}: {str(e)}", file=sys.stderr)
                
                if attempt < max_attempts - 1:
                    delay = min(initial_delay * (backoff_factor ** attempt), 30.0)
                    print(f"[RETRY] Waiting {delay:.2f}s before retry...", file=sys.stderr)
                    await asyncio.sleep(delay)
        
        # If we get here, all retries failed
        if last_exception:
            if isinstance(last_exception, ProviderError):
                raise last_exception
            raise ProviderError(f"Request error after {max_attempts} attempts: {str(last_exception)}", "chutes", 502)
    
    async def _non_stream_request(
        self, 
        url: str, 
        headers: Dict[str, str], 
        data: Dict[str, Any]
    ) -> ChatCompletionResponse:
        """Make non-streaming request to Chutes API."""
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, headers=headers, json=data)
                
                if response.status_code != 200:
                    raise ProviderError(
                        f"Chutes API error: {response.status_code} {response.text}",
                        "chutes",
                        response.status_code
                    )
                
                response_data = response.json()
                return ChatCompletionResponse(**response_data)
            
            except (httpx.RequestError, json.JSONDecodeError) as e:
                raise ProviderError(f"Request error: {str(e)}", "chutes", 502)
    
    def _parse_stream_chunk(self, event: Any) -> Optional[ChatCompletionChunk]:
        """Parse SSE event into normalized chunk."""
        
        if not event.data or event.data.strip() == "[DONE]":
            return None
        
        try:
            data = JSONProcessor.parse_safe(event.data)
            if not data:
                return None
            
            # Process tool calls if present
            if "choices" in data:
                for choice in data["choices"]:
                    delta = choice.get("delta", {})
                    
                    # Fix null role first
                    delta = fix_delta(delta)
                    
                    # Process and merge tool calls
                    if "tool_calls" in delta:
                        result = process_delta(delta, self._tool_call_accumulator)
                        if result:
                            choice["delta"]["tool_calls"] = result["tool_calls"]
            
            return ChatCompletionChunk(**data)
        
        except Exception as e:
            # Log error but continue streaming
            print(f"Error parsing chunk: {e}", file=sys.stderr)
            return None
    
    # No complex finish chunk needed - simple streaming handles it
    
    def _get_upstream_url(self, endpoint: str, request: ProviderRequest) -> str:
        """Build upstream URL."""
        model = self.get_model_name(request)
        return self.settings.get_upstream_url(f"{endpoint}", model)
