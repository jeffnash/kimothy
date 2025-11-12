"""Main FastAPI application for Kimothy proxy."""

import asyncio
import sys
import time
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from .config import get_settings, Settings
from .models import ChatCompletionRequest, CompletionRequest, ChatCompletionResponse, ChatCompletionChunk
from .models import Choice, Message, ToolCall, FunctionDefinition
from .providers.factory import create_provider
from .providers.base import ProviderRequest
from .utils.sse import SSEGenerator
from .utils.error import ProxyError, ValidationError, ProviderError
from .utils.logging import get_logger
from .utils.tool_call import ToolCallManager
from .streaming import process_delta, fix_delta
from .utils.tool_schema import build_tool_schemas
from .utils.tool_sanitizer import sanitize_tool_calls


app = FastAPI(
    title="Kimothy Proxy",
    description="OpenAI-compatible proxy for Kimi-like providers",
    version="2.0.0"
)

# Global settings
_settings: Settings = None
_provider = None
_logger = None


def _redact_api_key(key: str) -> str:
    """Return first 3 and last 2 characters of a key, redacting the middle.

    Handles empty/short keys gracefully.
    """
    if not key:
        return ""
    try:
        if len(key) <= 5:
            # Keep at least first and last character for very short keys
            return f"{key[:1]}...{key[-1:]}"
        return f"{key[:3]}...{key[-2:]}"
    except Exception:
        return "***"


def _redact_auth_header(value: str) -> str:
    """Redact an Authorization header value.

    If it's a Bearer token, redact the token portion (first 3, last 2).
    Otherwise, generically redact the whole value.
    """
    if value is None:
        return ""
    try:
        if value.lower().startswith("bearer "):
            token = value.split(" ", 1)[1]
            return f"Bearer {_redact_api_key(token)}"
        return _redact_api_key(value)
    except Exception:
        return "***"


def _redact_headers(headers: dict) -> dict:
    """Return a shallow copy of headers with Authorization redacted."""
    try:
        redacted = dict(headers or {})
        for k in list(redacted.keys()):
            if k.lower() == "authorization":
                redacted[k] = _redact_auth_header(redacted.get(k))
        return redacted
    except Exception:
        return {}


@app.on_event("startup")
async def startup_event():
    """Initialize application."""
    global _settings, _provider, _logger
    
    _settings = get_settings()
    _provider = create_provider(_settings)
    _logger = get_logger("kimothy.main")
    
    # Log to file for debugging
    import sys
    print(f"\n[!] PROXY STARTED ON PORT {_settings.port}", file=sys.stderr)
    print(f"[!] UPSTREAM: {_settings.upstream_base_url}", file=sys.stderr)
    print(f"[!] MODEL: {_settings.default_model}", file=sys.stderr)
    print(f"[!] ENSURE_STREAM: {_settings.ensure_stream}", file=sys.stderr)
    print(f"[!] API KEY: {_redact_api_key(_settings.upstream_api_key)}", file=sys.stderr)
    print(f"[!] Test with: curl -X POST http://localhost:{_settings.port}/v1/chat/completions ...\n", file=sys.stderr)

    # Log startup without exposing full API key
    try:
        settings_dict = _settings.model_dump()
    except Exception:
        # Fallback for older Pydantic versions
        settings_dict = _settings.dict()
    if isinstance(settings_dict, dict) and "upstream_api_key" in settings_dict:
        settings_dict["upstream_api_key"] = _redact_api_key(settings_dict.get("upstream_api_key"))

    _logger.info("Kimothy proxy started", settings=settings_dict)


# Add middleware to log ALL requests (matches original behavior)
@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    """Log all incoming requests at middleware level."""
    import sys
    start_time = time.time()
    
    # Log incoming request
    print(f"\n[!] INCOMING REQUEST (middleware): {request.method} {request.url}", file=sys.stderr)
    print(f"[!] Client Authorization: {_redact_auth_header(request.headers.get('authorization', 'NONE'))}", file=sys.stderr)
    
    response = await call_next(request)
    
    duration_ms = int((time.time() - start_time) * 1000)
    print(f"[!] RESPONSE: {response.status_code} ({duration_ms}ms)", file=sys.stderr)
    
    return response


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ProxyError)
async def proxy_error_handler(request: Request, exc: ProxyError):
    """Handle proxy errors."""
    # Ensure status_code is an integer
    status_code = int(exc.status_code) if isinstance(exc.status_code, str) else exc.status_code
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": exc.message,
                "code": exc.code,
                "type": exc.__class__.__name__
            }
        }
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    content = {
        "error": {
            "message": exc.message,
            "code": exc.code,
            "type": "invalid_request_error"
        }
    }
    if exc.field:
        content["error"]["field"] = exc.field
    
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    _logger.exception("Unexpected error", exc=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "code": "internal_error",
                "type": "internal_error"
            }
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


def _get_request_model(request: Request) -> str:
    """Extract model from request."""
    model = request.path_params.get("model")
    if not model:
        model = _settings.default_model
    return model


def _validate_api_key(request: Request) -> str:
    """Validate and return API key from request."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]  # Remove "Bearer " prefix
    return ""


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completions requests."""
    
    # LOG INCOMING REQUEST
    import sys
    print(f"\n[!] INCOMING REQUEST: {request.method} {request.url}", file=sys.stderr)
    print(f"[!] Client Authorization: {_redact_auth_header(request.headers.get('authorization', 'NONE'))}", file=sys.stderr)
    print(f"[!] Content-Type: {request.headers.get('content-type', 'NONE')}", file=sys.stderr)
    print(f"[!] User-Agent: {request.headers.get('user-agent', 'NONE')}", file=sys.stderr)
    
    _logger.info("INCOMING REQUEST", 
                 method=request.method, 
                 url=str(request.url),
                 headers=_redact_headers(dict(request.headers)),
                 client_auth=_redact_auth_header(request.headers.get('authorization', 'NONE')))
    
    # Validate request
    try:
        body = await request.json()
        chat_request = ChatCompletionRequest(**body)
        print(f"[!] Parsed body - model: {chat_request.model}, stream: {chat_request.stream}", file=sys.stderr)
        _logger.info("PARSED REQUEST", 
                     client_model=chat_request.model,
                     client_stream=chat_request.stream,
                     default_model=_settings.default_model)
    except Exception as e:
        print(f"[!] ERROR parsing request: {e}", file=sys.stderr)
        _logger.error("INVALID REQUEST BODY", error=str(e))
        raise ValidationError(f"Invalid request body: {str(e)}")
    
    model = chat_request.model or _settings.default_model
    if not model:
        raise ValidationError("Model not specified")
    
    # MATCH ORIGINAL: respect client stream setting with default True
    caller_wants_stream = bool(body.get('stream', True))
    # If ensure_stream is configured, override
    if _settings.ensure_stream and not caller_wants_stream:
        body['stream'] = True
        _logger.info("OVERRIDING STREAM TO TRUE", original=chat_request.stream, forced=True)
    else:
        body['stream'] = caller_wants_stream 
    
    # MATCH ORIGINAL: filter incoming headers, drop client auth completely, add upstream key
    drop_headers = {
        "host", "authorization", "content-length", "connection",
        "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade"
    }
    cleaned_headers = {k: v for k, v in dict(request.headers).items() if k.lower() not in drop_headers}
    
    provider_headers = cleaned_headers.copy()
    if _settings.upstream_api_key:
        provider_headers["Authorization"] = f"Bearer {_settings.upstream_api_key}"
    provider_headers["Accept"] = "text/event-stream" if body['stream'] else "application/json"
    provider_headers["Content-Type"] = "application/json"
    
    _logger.info("OUTGOING TO UPSTREAM", 
                 upstream_url=_settings.upstream_base_url,
                 upstream_auth=_redact_auth_header(provider_headers.get('Authorization', '')),
                 model=model)
    
    provider_request = ProviderRequest(
        data=body,
        headers=provider_headers
    )
    
    # Always use streaming response since we forced stream=True
    return StreamingResponse(
        _stream_chat_response(provider_request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable proxy buffering for nginx
        }
    )


@app.post("/v1/completions")
@app.post("/completions")
async def completions(request: Request):
    """Handle text completions requests."""
    
    try:
        body = await request.json()
        completion_request = CompletionRequest(**body)
    except Exception as e:
        raise ValidationError(f"Invalid request body: {str(e)}")
    
    model = completion_request.model or _settings.default_model
    if not model:
        raise ValidationError("Model not specified")
    
    # FORCE upstream API key, ignore client auth completely
    provider_headers = {
        "Authorization": f"Bearer {_settings.upstream_api_key}",
        "Content-Type": "application/json",
    }
    
    provider_request = ProviderRequest(
        data=body,
        headers=provider_headers
    )
    
    if completion_request.stream:
        return StreamingResponse(
            _stream_completion_response(provider_request),
            media_type="text/event-stream"
        )
    else:
        return await _handle_non_stream_completion(provider_request)


async def _stream_chat_response(request: ProviderRequest) -> AsyncIterator[bytes]:
    """Stream chat completions response."""
    
    first_chunk = True
    role_emitted = False
    
    # Initialize tool call processor for this stream
    tool_manager = ToolCallManager(_settings)
    accumulated_reasoning = ""
    
    # Create persistent reasoning parser for Kimi-style markers
    from kimothy.providers.reasoning_handler import SimpleReasoningParser
    _reasoning_parser = SimpleReasoningParser()
    
    # Build tool schema map (client-provided tool definitions)
    tool_schemas = build_tool_schemas(request.data)

    try:
        async for chunk in _provider.chat_completions(request):
            # Log full chunk without truncation
            _logger.debug("INCOMING CHUNK", chunk=chunk)
            
            if isinstance(chunk, ChatCompletionChunk):
                emitted_something = False
                force_emit_chunk = False  # Ensure we emit when synthetic tool calls appear
                
                # Process each choice
                for choice in chunk.choices:
                    if hasattr(choice, 'delta') and choice.delta is not None:
                        # Fix null role on first chunk
                        if first_chunk and choice.delta.role is None:
                            choice.delta.role = "assistant"
                            role_emitted = True
                        
                        # Check for reasoning_content before we strip it
                        has_reasoning_content = bool(
                            hasattr(choice.delta, 'reasoning_content') and 
                            choice.delta.reasoning_content
                        )
                        
                        # Store reasoning_content value before we clear it
                        reasoning_content_value = None
                        if has_reasoning_content:
                            reasoning_content_value = choice.delta.reasoning_content
                            accumulated_reasoning += reasoning_content_value
                        
                        # Process tool_calls - convert to dict format for processing
                        if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                            # Convert ToolCall objects to dicts for processing
                            tool_calls_dict = []
                            for tc in choice.delta.tool_calls:
                                if hasattr(tc, 'dict'):
                                    tool_calls_dict.append(tc.dict())
                                else:
                                    # Already a dict
                                    tool_calls_dict.append(tc)
                            
                            # Process through ToolCallManager to handle null ids/names and fragmentation
                            delta_data = {"tool_calls": tool_calls_dict}
                            processed = tool_manager.process_upstream_delta(delta_data)
                            
                            # Replace with processed tool calls (keep as dicts to match Message model)
                            if 'tool_calls' in processed:
                                choice.delta.tool_calls = processed['tool_calls']
                                # If schemas are available, validate upstream tool calls too (schema-only); keep invalid to avoid drops
                                if tool_schemas:
                                    choice.delta.tool_calls = sanitize_tool_calls(
                                        choice.delta.tool_calls,
                                        tool_schemas=tool_schemas,
                                        enable_heuristics=False,
                                        keep_invalid=True,
                                    )
                            else:
                                choice.delta.tool_calls = None
                        
                        # Process reasoning_content for Kimi-style markers AFTER handling normal tool_calls
                        # Use the stored value since we haven't cleared it from the delta yet
                        if reasoning_content_value:
                            # Log the reasoning content we're about to parse
                            _logger.debug("Parsing reasoning content", length=len(reasoning_content_value), content=reasoning_content_value[:100])
                            
                            # Extract tool calls from reasoning content using persistent parser
                            synthetic_tool_calls = _reasoning_parser.extract_tool_calls(reasoning_content_value)
                            
                            if synthetic_tool_calls:
                                _logger.debug("Found synthetic tool calls", count=len(synthetic_tool_calls), calls=synthetic_tool_calls)
                                # Add these synthetic tool calls to the delta
                                if choice.delta.tool_calls:
                                    _logger.debug("Extending existing tool calls", existing=len(choice.delta.tool_calls))
                                    choice.delta.tool_calls.extend(synthetic_tool_calls)
                                else:
                                    _logger.debug("Setting new synthetic tool calls")
                                    choice.delta.tool_calls = synthetic_tool_calls
                                # Sanitize tool calls to avoid invalid path reads and duplicates (schema-driven when available)
                                choice.delta.tool_calls = sanitize_tool_calls(
                                    choice.delta.tool_calls,
                                    tool_schemas=tool_schemas,
                                    enable_heuristics=not bool(tool_schemas),
                                    keep_invalid=True,
                                )
                                # Force emission even if role/content aren't present
                                force_emit_chunk = True

                            # Optionally expose reasoning text as normal content for debugging
                            if _settings.expose_reasoning_as_content:
                                try:
                                    # Stream as incremental deltas: send only current slice
                                    if hasattr(choice.delta, 'content'):
                                        # If already has content for this delta, append
                                        if choice.delta.content:
                                            choice.delta.content = f"{choice.delta.content}{reasoning_content_value}"
                                        else:
                                            choice.delta.content = reasoning_content_value
                                    else:
                                        # Ensure attribute exists
                                        choice.delta.content = reasoning_content_value
                                except Exception:
                                    pass
                        
                        # Finally, don't emit reasoning_content in delta, keep it internal
                        if has_reasoning_content:
                            choice.delta.reasoning_content = None
                
                # Avoid premature finish if upstream signaled finish_reason=tool_calls but we have no usable tool_calls
                try:
                    any_tools = any(
                        hasattr(c, 'delta') and c.delta and c.delta.tool_calls not in (None, [])
                        for c in chunk.choices
                    )
                    if not any_tools:
                        for c in chunk.choices:
                            if hasattr(c, 'finish_reason') and c.finish_reason == 'tool_calls':
                                c.finish_reason = None
                except Exception:
                    pass

                # Only emit chunk if it has content (always emit first chunk with role)
                # Ensure we're emitting Assistant chunks even with reasoning_content
                has_content = any(
                    hasattr(choice, 'delta') and choice.delta and (
                        choice.delta.role is not None or
                        choice.delta.content is not None or
                        (choice.delta.tool_calls not in (None, []))
                    ) for choice in chunk.choices
                )
                
                # Always emit the first chunk to establish role
                if has_content or first_chunk or force_emit_chunk:
                    sse_data = SSEGenerator.generate_json(chunk.dict())
                    print(f"DEBUG: Yielding SSE chunk: {sse_data[:100]}...", file=sys.stderr)
                    yield sse_data.encode('utf-8')
                    # Force flush by yielding control back to event loop
                    await asyncio.sleep(0)
                    emitted_something = True
                
                first_chunk = False
        
        # Generate final tool calls if any
        if tool_manager.processor.tool_calls:
            final_calls, _ = tool_manager.finalize()
            if final_calls and any(final_calls):
                # Create synthetic finish chunk
                delta = Message()
                delta.tool_calls = final_calls
                
                choice = Choice(
                    index=0,
                    delta=delta,
                    finish_reason="tool_calls"
                )
                
                complete_chunk = ChatCompletionChunk(
                    id="kimothy_final",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=request.data.get("model", "unknown"),
                    choices=[choice]
                )
                
                yield SSEGenerator.generate_json(complete_chunk.dict()).encode('utf-8')
        
        # If the reasoning parser has a pending tool with parseable JSON, flush it
        try:
            pending_calls = _reasoning_parser.flush()
            if pending_calls:
                delta = Message()
                delta.tool_calls = pending_calls
                choice = Choice(index=0, delta=delta, finish_reason="tool_calls")
                complete_chunk = ChatCompletionChunk(
                    id="kimothy_final_pending",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=request.data.get("model", "unknown"),
                    choices=[choice]
                )
                yield SSEGenerator.generate_json(complete_chunk.dict()).encode('utf-8')
        except Exception:
            pass

        # Send final [DONE] event
        done_data = SSEGenerator.generate_done()
        print(f"DEBUG: Yielding final [DONE] event", file=sys.stderr)
        yield done_data
        
    except Exception as e:
        _logger.exception("Error in streaming response", exc=e)
        # Send error in SSE format
        error_data = {
            "error": {
                "message": str(e),
                "code": "streaming_error",
                "type": type(e).__name__
            }
        }
        yield SSEGenerator.generate_json(error_data).encode('utf-8')
        yield SSEGenerator.generate_done().encode('utf-8')


async def _handle_non_stream_chat(request: ProviderRequest) -> JSONResponse:
    """Handle non-streaming chat completions."""
    
    try:
        response = None
        async for result in _provider.chat_completions(request):
            if isinstance(result, ChatCompletionResponse):
                response = result
                break
        
        if not response:
            raise ProviderError("Empty response from provider")
        
        return JSONResponse(content=response.dict())
    
    except Exception as e:
        _logger.exception("Error in non-stream response", exc=e)
        if isinstance(e, ProxyError):
            raise
        raise ProviderError(f"Error processing request: {str(e)}")


async def _stream_completion_response(request: ProviderRequest) -> AsyncIterator[bytes]:
    """Stream text completions response."""
    
    try:
        async for chunk in _provider.completions(request):
            if isinstance(chunk, ChatCompletionChunk):
                yield SSEGenerator.generate_json(chunk.dict()).encode('utf-8')
        
        # Send final [DONE] event
        done_data = SSEGenerator.generate_done()
        print(f"DEBUG: Yielding final [DONE] event", file=sys.stderr)
        yield done_data.encode('utf-8')
        
    except Exception as e:
        _logger.exception("Error in streaming completion", exc=e)
        error_data = {
            "error": {
                "message": str(e),
                "code": "streaming_error",
                "type": type(e).__name__
            }
        }
        yield SSEGenerator.generate_json(error_data).encode('utf-8')
        yield SSEGenerator.generate_done().encode('utf-8')


async def _handle_non_stream_completion(request: ProviderRequest) -> JSONResponse:
    """Handle non-streaming text completions."""
    
    try:
        response = None
        async for result in _provider.completions(request):
            if isinstance(result, ChatCompletionResponse):
                response = result
                break
        
        if not response:
            raise ProviderError("Empty response from provider") 
        
        return JSONResponse(content=response.dict())
    
    except Exception as e:
        _logger.exception("Error in non-stream completion", exc=e)
        if isinstance(e, ProxyError):
            raise
        raise ProviderError(f"Error processing request: {str(e)}")
# (refactored) sanitizer lives in utils.tool_sanitizer
