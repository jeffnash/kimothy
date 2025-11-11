#!/usr/bin/env python3
"""
Kimothy: OpenAI-Compatible Proxy for Kimi-like Streaming

Goal
- Accept OpenAI-style requests (e.g., POST /v1/chat/completions)
- Call the upstream Kimi-like API
- Buffer/normalize the upstream's "weird" streaming (e.g., reasoning_content)
- Return a clean, OpenAI-compatible response to the caller

Notes
- This proxy intentionally does not use or depend on kimi_proxy.py.
- It best-effort normalizes streams by discarding reasoning/thinking deltas and
  only forwarding human-visible assistant content in OpenAI Chat Completions format.

Config (env vars)
- UPSTREAM_BASE_URL: Base URL of the upstream API (e.g., https://api.moonshot.cn/v1)
- UPSTREAM_API_KEY:  Bearer token for upstream
- DEFAULT_MODEL:      Optional default model when request omits it
- LOG_REQUESTS:       "1" to log incoming/outgoing details (stderr)

Run
  uvicorn kimothy:app --host 0.0.0.0 --port 8800
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
from typing import Any, AsyncIterator, Dict, Optional
import random

import httpx
try:
    from rich.console import Console as _RichConsole
    from rich.live import Live as _RichLive
    from rich.spinner import Spinner as _RichSpinner
    _RICH_OK = True
except Exception:
    _RICH_OK = False
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse


# -----------------------------
# Configuration helpers
# -----------------------------

UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")
LOG_REQUESTS = os.getenv("LOG_REQUESTS", "0") == "1"
LOG_UPSTREAM_LINES = os.getenv("LOG_UPSTREAM_LINES", "0") == "1"
LOG_DOWNSTREAM_LINES = os.getenv("LOG_DOWNSTREAM_LINES", "0") == "1"
EMIT_FUNCTION_CALL_LEGACY = os.getenv("EMIT_FUNCTION_CALL_LEGACY", "1") == "1"
FINISH_ON_TOOL_END = os.getenv("FINISH_ON_TOOL_END", "1") == "1"
HEARTBEAT_MS = int(os.getenv("HEARTBEAT_MS", "0") or 0)
STATUS_COMMENTS = os.getenv("STATUS_COMMENTS", "1") == "1"
TTY_SPINNER = os.getenv("TTY_SPINNER", "1") == "1"
SPINNER_INTERVAL_MS = int(os.getenv("SPINNER_INTERVAL_MS", "150") or 150)
RETRY_429_MAX_ATTEMPTS = int(os.getenv("RETRY_429_MAX_ATTEMPTS", "3") or 3)
RETRY_429_BASE_MS = int(os.getenv("RETRY_429_BASE_MS", "500") or 500)
RETRY_429_MAX_MS = int(os.getenv("RETRY_429_MAX_MS", "4000") or 4000)
AUTO_FINISH_TOOLCALLS = os.getenv("AUTO_FINISH_TOOLCALLS", "1") == "1"
AUTO_FINISH_DELAY_MS = int(os.getenv("AUTO_FINISH_DELAY_MS", "800") or 800)

# Streaming and routing options inspired by needs seen in kimi-cli usage patterns
# - ENSURE_STREAM: Force `stream: true` in the upstream body when streaming
# - URL_MODEL_PLACEMENT: "body" (default) or "path"; if "path", place model in URL
# - UPSTREAM_ENDPOINT: Optional override for upstream endpoint path; one of
#                      "chat/completions" or "completions". If unset, mirrors caller path.
# - OVERRIDE_TEMPERATURE / OVERRIDE_MAX_TOKENS: Optional numeric overrides
ENSURE_STREAM = os.getenv("ENSURE_STREAM", "1") == "1"
URL_MODEL_PLACEMENT = os.getenv("URL_MODEL_PLACEMENT", "body")  # body | path
UPSTREAM_ENDPOINT = os.getenv("UPSTREAM_ENDPOINT")  # e.g., "chat/completions" or "completions"

OVERRIDE_TEMPERATURE = os.getenv("OVERRIDE_TEMPERATURE")
OVERRIDE_MAX_TOKENS = os.getenv("OVERRIDE_MAX_TOKENS")


def _log(msg: str) -> None:
    if LOG_REQUESTS:
        print(msg, file=sys.stderr)


def _strip_kimi_fields(obj: Any) -> Any:
    """Recursively remove Kimi-specific fields (e.g., reasoning_content, reasoning tokens).

    Best-effort cleanup so that the returned object looks like standard OpenAI
    chat completions output.
    """
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in obj.items():
            # Drop known Kimi-only keys
            if k in {"reasoning_content", "reasoning_tokens", "thinking", "k2_metadata"}:
                continue
            cleaned[k] = _strip_kimi_fields(v)
        return cleaned
    if isinstance(obj, list):
        return [_strip_kimi_fields(x) for x in obj]
    return obj


def _synthesize_tool_calls_from_reasoning(reasoning: str) -> list[Dict[str, Any]]:
    """Build a full tool_calls list from a complete reasoning_content string.

    Returns a list of objects matching OpenAI's non-stream tool_calls schema:
    [{"type": "function", "function": {"name": str, "arguments": str}}]
    """
    parser = ReasoningToolParser()
    events = parser.feed(reasoning or "")
    calls: list[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for ev in events:
        et = ev.get("type")
        if et == "begin":
            current = {"type": "function", "function": {"name": ev.get("name") or "", "arguments": ""}}
            calls.append(current)
        elif et == "args" and current is not None:
            current["function"]["arguments"] += ev.get("text", "")
        elif et == "end":
            current = None
    return calls


# --- Reasoning tool-call parser for Kimi-style markers ---
TOOL_MARKERS = {
    "SECTION_BEGIN": "<|tool_calls_section_begin|>",
    "SECTION_END": "<|tool_calls_section_end|>",
    "CALL_BEGIN": "<|tool_call_begin|>",
    "CALL_END": "<|tool_call_end|>",
    "ARG_BEGIN": "<|tool_call_argument_begin|>",
    "ARG_END": "<|tool_call_argument_end|>",
}


class ReasoningToolParser:
    """Parses reasoning_content text to synthesize tool call events.

    Produces events:
      - {"type":"begin","name": str | None}
      - {"type":"args","text": str}
      - {"type":"end"}
    Maintains state across feeds to handle chunked markers.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_section = False
        self._in_call = False
        self._in_args = False
        self._pending_name: Optional[str] = None

    def feed(self, text: str) -> list[dict[str, Any]]:
        self._buf += text or ""
        out: list[dict[str, Any]] = []

        def find_next_marker(s: str) -> tuple[int, Optional[str]]:
            best_i = -1
            best_key = None
            for key, mark in TOOL_MARKERS.items():
                i = s.find(mark)
                if i != -1 and (best_i == -1 or i < best_i):
                    best_i, best_key = i, key
            return best_i, best_key

        while True:
            i, key = find_next_marker(self._buf)
            if i == -1 or key is None:
                # No more complete markers in buffer. If currently in args, emit the tail as args text.
                if self._in_section and self._in_call and self._in_args and self._buf:
                    out.append({"type": "args", "text": self._buf})
                    self._buf = ""
                break

            # Prefix between last marker and next marker
            prefix = self._buf[:i]
            self._buf = self._buf[i + len(TOOL_MARKERS[key]) :]

            # If we are inside argument streaming, prefix is argument text
            if self._in_section and self._in_call and self._in_args and prefix:
                out.append({"type": "args", "text": prefix})

            # Handle marker
            if key == "SECTION_BEGIN":
                self._in_section = True
            elif key == "SECTION_END":
                # Close any open states
                if self._in_args:
                    self._in_args = False
                if self._in_call:
                    out.append({"type": "end"})
                    self._in_call = False
                    self._pending_name = None
                self._in_section = False
            elif key == "CALL_BEGIN":
                # Try to capture a function name in the immediate text before ARG_BEGIN
                self._in_call = True
                self._pending_name = None
            elif key == "CALL_END":
                if self._in_args:
                    self._in_args = False
                if self._in_call:
                    out.append({"type": "end"})
                self._in_call = False
                self._pending_name = None
            elif key == "ARG_BEGIN":
                # Attempt to parse tool name from any prefix carried over (e.g., "functions.name")
                if prefix and self._pending_name is None:
                    # crude extract: last token after 'functions.' up to whitespace or '<'
                    name = None
                    pf = prefix.strip()
                    idx = pf.rfind("functions.")
                    if idx != -1:
                        tail = pf[idx + len("functions.") :]
                        # stop at first non-identifier char
                        j = 0
                        while j < len(tail) and (tail[j].isalnum() or tail[j] in ("_", "-")):
                            j += 1
                        candidate = tail[:j]
                        if candidate:
                            name = candidate
                    self._pending_name = name
                out.append({"type": "begin", "name": self._pending_name})
                self._in_args = True
            elif key == "ARG_END":
                self._in_args = False
            else:
                # Unknown marker; ignore
                pass

        return out


def _to_openai_chunk(base_id: str, model: str, index: int, content: str, include_role: bool) -> dict[str, Any]:
    delta: Dict[str, Any] = {"content": content}
    if include_role:
        delta["role"] = "assistant"
    return {
        "id": base_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": delta,
                "finish_reason": None,
            }
        ],
    }


def _to_openai_finish_chunk(base_id: str, model: str, index: int, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": base_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
    }


async def _stream_upstream(
    client: httpx.AsyncClient,
    url: str,
    json_body: dict[str, Any],
    headers: dict[str, str],
) -> httpx.Response:
    # Retry on 429 with exponential backoff + jitter
    attempt = 0
    last_resp: httpx.Response | None = None
    while True:
        resp = await client.post(url, json=json_body, headers=headers, timeout=None)
        if resp.status_code != 429:
            return resp
        last_resp = resp
        if attempt >= RETRY_429_MAX_ATTEMPTS:
            return resp
        # Backoff with jitter
        delay_ms = min(RETRY_429_BASE_MS * (2 ** attempt), RETRY_429_MAX_MS)
        jitter = 0.5 + random.random()  # 0.5x to 1.5x
        sleep_s = (delay_ms * jitter) / 1000.0
        if LOG_REQUESTS:
            _log(f"429 received; retrying in {sleep_s:.2f}s (attempt {attempt+1}/{RETRY_429_MAX_ATTEMPTS})")
        await asyncio.sleep(sleep_s)
        attempt += 1
    # Fallback
    assert last_resp is not None
    return last_resp


async def _normalize_stream(
    upstream: httpx.Response,
    model: str,
    *,
    mode: str = "chat",  # "chat" or "completion"
) -> AsyncIterator[bytes]:
    """Consume an upstream SSE stream, drop reasoning deltas, forward content-only as OpenAI SSE.

    - Accepts typical SSE lines like:
        data: {"choices":[{"delta":{"content":"..","reasoning_content":".."}}]}
      or provider-specific variants that still wrap JSON in "data: ..." lines.
    - Forwards only content deltas as OpenAI chunks
    - Emits a single role chunk at first content
    - Emits [DONE] when upstream indicates done or when a finish chunk is sent
    """
    sent_role = False
    base_id = f"chatcmpl-{int(time.time()*1000)}" if mode == "chat" else f"cmpl-{int(time.time()*1000)}"
    # Track tool_calls incremental arguments per index
    tool_state: Dict[int, Dict[str, Any]] = {}
    # Parser to synthesize tool calls from reasoning_content markers
    rparser = ReasoningToolParser()
    next_tool_index = 0
    # Track tool activity for auto-finish hardening
    tool_active = False
    last_tool_index = -1
    last_tool_update_ts = 0.0
    tool_finish_emitted = False

    # Emit stream-open status comment if enabled
    if STATUS_COMMENTS:
        out = f": stream-open {int(time.time()*1000)}\n\n"
        if LOG_DOWNSTREAM_LINES:
            _log(f"DOWNSTREAM -> {out.strip()}")
        sys.stderr.write("STREAM START\n")
        sys.stderr.flush()
        yield out.encode("utf-8")

    # Optional terminal spinner on server side for visual indication
    spinner_task: Optional[asyncio.Task] = None
    spinner_running = True

    async def _spinner() -> None:
        if _RICH_OK:
            console = _RichConsole(file=sys.stderr, force_terminal=True, color_system=None, soft_wrap=False)
            spinner = _RichSpinner("dots", text="Streaming…")
            refresh_hz = max(4, int(1000 / max(SPINNER_INTERVAL_MS, 50)))
            with _RichLive(spinner, console=console, refresh_per_second=refresh_hz, transient=True):
                while spinner_running:
                    await asyncio.sleep(max(SPINNER_INTERVAL_MS, 50) / 1000)
            console.print("✓ stream closed")
        else:
            frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
            i = 0
            try:
                while spinner_running:
                    frame = frames[i % len(frames)]
                    i += 1
                    sys.stderr.write(f"\r{frame} streaming… ")
                    sys.stderr.flush()
                    await asyncio.sleep(max(SPINNER_INTERVAL_MS, 50) / 1000)
            finally:
                sys.stderr.write("\r✓ stream closed      \n")
                sys.stderr.flush()

    if TTY_SPINNER:
        spinner_task = asyncio.create_task(_spinner())

    # Reader task pumps upstream lines into a queue
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    async def _reader() -> None:
        try:
            async for raw_line in upstream.aiter_lines():
                await queue.put(raw_line)
        finally:
            await queue.put(None)

    reader_task = asyncio.create_task(_reader())

    # Choose polling interval to allow auto-finish checks even if heartbeat disabled
    def _poll_interval_sec() -> Optional[float]:
        candidates = [ms for ms in [HEARTBEAT_MS, AUTO_FINISH_DELAY_MS // 2] if ms and ms > 0]
        if not candidates:
            return None
        return max(0.05, min(candidates) / 1000)

    try:
        while True:
            try:
                raw_line = await asyncio.wait_for(queue.get(), timeout=_poll_interval_sec())
            except asyncio.TimeoutError:
                # Emit heartbeat comment
                if HEARTBEAT_MS > 0:
                    out = f": ping {int(time.time()*1000)}\n\n"
                    if LOG_DOWNSTREAM_LINES:
                        _log(f"DOWNSTREAM -> {out.strip()}")
                    yield out.encode("utf-8")
                # Auto finish tool-calls if configured and no explicit end arrives
                if AUTO_FINISH_TOOLCALLS and tool_active and not tool_finish_emitted:
                    if (time.time() - last_tool_update_ts) * 1000 >= AUTO_FINISH_DELAY_MS and last_tool_index >= 0:
                        finish_chunk = _to_openai_finish_chunk(base_id, model, last_tool_index, "tool_calls")
                        out2 = f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
                        if LOG_DOWNSTREAM_LINES:
                            _log(f"DOWNSTREAM -> {out2.strip()}")
                        yield out2.encode("utf-8")
                        tool_finish_emitted = True
                continue

            if raw_line is None:
                return

            line = (raw_line or "").strip()
            if not line:
                continue

            # SSE comments or heartbeat from upstream
            if line.startswith(":"):
                if LOG_UPSTREAM_LINES:
                    _log(f"UPSTREAM <- {line}")
                continue

            # Standard OpenAI terminator
            if line == "data: [DONE]":
                if LOG_UPSTREAM_LINES:
                    _log("UPSTREAM <- data: [DONE]")
                # Stop spinner when upstream finishes
                spinner_running = False
                if spinner_task:
                    with contextlib.suppress(Exception):
                        await spinner_task
                yield b"data: [DONE]\n\n"
                return

            # Expect JSON payload wrapped in data:
            if not line.startswith("data:"):
                # Some providers send plain JSON lines; try to parse anyway
                payload_text = line
            else:
                payload_text = line[len("data:"):].strip()

            if LOG_UPSTREAM_LINES:
                _log(f"UPSTREAM <- {payload_text}")

            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                # Ignore non-JSON lines
                continue

            choices = payload.get("choices")
            if not isinstance(choices, list):
                continue

            for choice in choices:
                idx = int(choice.get("index", 0))
                finish_reason = choice.get("finish_reason")

                # If upstream already indicates finish
                if finish_reason:
                    if mode == "chat":
                        chunk = _to_openai_finish_chunk(base_id, model, idx, str(finish_reason))
                    else:
                        # Completions: send empty text with finish reason
                        chunk = {
                            "id": base_id,
                            "object": "text_completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": idx,
                                    "text": "",
                                    "finish_reason": str(finish_reason),
                                    "logprobs": None,
                                }
                            ],
                        }
                    out = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    if LOG_DOWNSTREAM_LINES:
                        _log(f"DOWNSTREAM -> {out.strip()}")
                    yield out.encode("utf-8")
                    continue

                delta = choice.get("delta") or {}
                if mode == "chat":
                    # Discard provider-only fields like reasoning_content, but allow tool_calls passthrough if present
                    content = delta.get("content")
                    if content is None and isinstance(choice.get("message"), dict):
                        content = choice["message"].get("content")

                    # Normalize tool_calls: may appear under delta, message, or top-level choice
                    tool_calls = delta.get("tool_calls")
                    if tool_calls is None and isinstance(choice.get("message"), dict):
                        tool_calls = choice["message"].get("tool_calls")
                    if tool_calls is None:
                        tool_calls = choice.get("tool_calls")

                    out_tc: list[Dict[str, Any]] | None = None
                    if isinstance(tool_calls, list) and tool_calls:
                        out_tc = []
                        for i, tc in enumerate(tool_calls):
                            # Expected OpenAI-ish shape:
                            # { id?, type: 'function', function: { name, arguments } }
                            ttype = tc.get("type", "function")
                            func = tc.get("function") or {}
                            name = func.get("name")
                            args_full = func.get("arguments", "") or ""

                            st = tool_state.get(i)
                            if st is None:
                                st = {"id": f"call_{i}_{int(time.time()*1000)}", "name": name, "args": ""}
                                tool_state[i] = st
                            # Update name if first time provided
                            if name and not st.get("name"):
                                st["name"] = name

                            # Compute incremental arguments
                            prev = st.get("args", "")
                            append = ""
                            if isinstance(args_full, str):
                                if args_full.startswith(prev):
                                    append = args_full[len(prev):]
                                else:
                                    # Upstream might have reset or compacted; emit full as append and reset state
                                    append = args_full
                                st["args"] = args_full

                            out_tc.append(
                                {
                                    "index": i,
                                    "id": st["id"],
                                    "type": ttype,
                                    "function": {
                                        "name": st.get("name") or name or "",
                                        # Stream only incremental append per OpenAI delta semantics
                                        "arguments": append,
                                    },
                                }
                            )
                            tool_active = True
                            last_tool_index = i
                            last_tool_update_ts = time.time()

                    # Also synthesize tool calls from reasoning_content markers
                    rcontent = delta.get("reasoning_content")
                    if rcontent is None and isinstance(choice.get("message"), dict):
                        rcontent = choice["message"].get("reasoning_content")
                    if isinstance(rcontent, str) and rcontent:
                        events = rparser.feed(rcontent)
                        if events and out_tc is None:
                            out_tc = []
                        for ev in events:
                            if ev["type"] == "begin":
                                # start a new tool index
                                idx2 = next_tool_index
                                next_tool_index += 1
                                st = {"id": f"call_{idx2}_{int(time.time()*1000)}", "name": ev.get("name") or "", "args": ""}
                                tool_state[idx2] = st
                                out_tc.append(
                                    {
                                        "index": idx2,
                                        "id": st["id"],
                                        "type": "function",
                                        "function": {"name": st["name"], "arguments": ""},
                                    }
                                )
                                tool_active = True
                                last_tool_index = idx2
                                last_tool_update_ts = time.time()
                            elif ev["type"] == "args":
                                # append arguments to last started tool (next_tool_index-1)
                                if next_tool_index > 0:
                                    idx2 = next_tool_index - 1
                                    st = tool_state.get(idx2)
                                    if st is not None:
                                        prev = st.get("args", "")
                                        append = ev["text"]
                                        st["args"] = prev + append
                                        out_tc.append(
                                            {
                                                "index": idx2,
                                                "id": st["id"],
                                                "type": "function",
                                                "function": {"name": st.get("name", ""), "arguments": append},
                                            }
                                        )
                            elif ev["type"] == "end":
                                # Optionally emit a finish chunk to signal tool-calls phase end
                                if FINISH_ON_TOOL_END and next_tool_index > 0:
                                    idx2 = next_tool_index - 1
                                    finish_chunk = _to_openai_finish_chunk(base_id, model, idx2, "tool_calls")
                                    out = f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
                                    if LOG_DOWNSTREAM_LINES:
                                        _log(f"DOWNSTREAM -> {out.strip()}")
                                    yield out.encode("utf-8")

                    if content is not None or out_tc is not None:
                        out_delta: Dict[str, Any] = {}
                        if not sent_role:
                            out_delta["role"] = "assistant"
                            sent_role = True
                        if content is not None:
                            out_delta["content"] = str(content)
                        if out_tc is not None:
                            out_delta["tool_calls"] = out_tc
                            # Legacy function_call compatibility for single-call tools
                            if EMIT_FUNCTION_CALL_LEGACY and out_tc:
                                fc = out_tc[-1]  # last emitted
                                out_delta["function_call"] = {
                                    "name": (fc.get("function") or {}).get("name", ""),
                                    "arguments": (fc.get("function") or {}).get("arguments", ""),
                                }
                        chunk = {
                            "id": base_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": idx,
                                    "delta": out_delta,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        out = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        if LOG_DOWNSTREAM_LINES:
                            _log(f"DOWNSTREAM -> {out.strip()}")
                        yield out.encode("utf-8")
                else:
                    # Completions stream typically uses 'text'
                    text = delta.get("content") or delta.get("text") or choice.get("text")
                    if text:
                        chunk = {
                            "id": base_id,
                            "object": "text_completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": idx,
                                    "text": str(text),
                                    "finish_reason": None,
                                    "logprobs": None,
                                }
                            ],
                        }
                        out = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        if LOG_DOWNSTREAM_LINES:
                            _log(f"DOWNSTREAM -> {out.strip()}")
                        yield out.encode("utf-8")

    except (asyncio.CancelledError, GeneratorExit):
        # Client disconnected or server shutdown; ensure spinner cleanup
        pass
    finally:
        # Upstream ended without explicit [DONE]; send terminator for client parity
        # Stop spinner at normal end
        spinner_running = False
        if spinner_task:
            with contextlib.suppress(Exception):
                await spinner_task

        if STATUS_COMMENTS:
            out = f": stream-close {int(time.time()*1000)}\n\n"
            if LOG_DOWNSTREAM_LINES:
                _log(f"DOWNSTREAM -> {out.strip()}")
            yield out.encode("utf-8")
        if LOG_DOWNSTREAM_LINES:
            _log("DOWNSTREAM -> data: [DONE]")
        yield b"data: [DONE]\n\n"


def _ensure_stream_true(body: dict[str, Any]) -> dict[str, Any]:
    # Force stream true when ENSURE_STREAM is set; otherwise respect caller intent
    if ENSURE_STREAM:
        if body.get("stream") is not True:
            body = {**body, "stream": True}
    else:
        if "stream" not in body:
            body = {**body, "stream": True}
    return body


def _build_upstream_headers(original: dict[str, str], *, accept: str) -> dict[str, str]:
    # Drop hop-by-hop and client-specific headers that can break httpx/body sizing
    drop = {
        "host",
        "authorization",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
    headers = {k: v for k, v in original.items() if k.lower() not in drop}
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    headers["Accept"] = accept
    headers["Content-Type"] = "application/json"
    return headers


app = FastAPI(title="OpenAI-Compatible Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.middleware("http")
async def _access_log(request: Request, call_next):
    start = time.time()
    response = None
    try:
        response = await call_next(request)
        return response
    except asyncio.CancelledError:
        # Client disconnected or server shutdown; do not escalate
        raise
    finally:
        try:
            dur_ms = int((time.time() - start) * 1000)
            path = request.url.path
            method = request.method
            status = getattr(response, "status_code", "-") if response is not None else "-"
            sys.stderr.write(f"\nACCESS {method} {path} -> {status} ({dur_ms}ms)\n")
            sys.stderr.flush()
        except Exception:
            pass


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _inject_overrides(body: dict[str, Any]) -> dict[str, Any]:
    # Model handling
    model = body.get("model") or DEFAULT_MODEL
    if URL_MODEL_PLACEMENT == "body":
        if model:
            body = {**body, "model": model}
    # Optional overrides
    if OVERRIDE_TEMPERATURE is not None:
        try:
            body = {**body, "temperature": float(OVERRIDE_TEMPERATURE)}
        except ValueError:
            pass
    if OVERRIDE_MAX_TOKENS is not None:
        try:
            body = {**body, "max_tokens": int(OVERRIDE_MAX_TOKENS)}
        except ValueError:
            pass
    return body


def _build_target_url(incoming_path: str, model: str) -> str:
    """Decide upstream endpoint and model placement."""
    # Choose endpoint: env override or mirror incoming
    # Derive endpoint from caller path safely (avoid str.lstrip char semantics)
    if UPSTREAM_ENDPOINT:
        endpoint = UPSTREAM_ENDPOINT.strip("/")
    else:
        if incoming_path.startswith("/v1/"):
            endpoint = incoming_path[len("/v1/"):].strip("/")
        else:
            endpoint = incoming_path.lstrip("/")
    # Fallback sanity
    if endpoint not in {"chat/completions", "completions"}:
        endpoint = "chat/completions"

    base = UPSTREAM_BASE_URL.rstrip("/")
    # Respect whether the upstream base already includes /v1
    has_version = base.endswith("/v1")
    version_prefix = "" if has_version else "/v1"
    if URL_MODEL_PLACEMENT == "path" and model:
        return f"{base}{version_prefix}/models/{model}/{endpoint}"
    return f"{base}{version_prefix}/{endpoint}"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    if not UPSTREAM_BASE_URL:
        return JSONResponse({"error": "UPSTREAM_BASE_URL is not configured"}, status_code=500)

    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    model = str(body.get("model") or DEFAULT_MODEL or "")
    if not model:
        # OpenAI requires model
        return JSONResponse({"error": "model is required"}, status_code=400)

    # Decide streaming path
    caller_wants_stream = bool(body.get("stream", True))

    # Compose final upstream body with overrides
    body = _inject_overrides(body)

    url = _build_target_url("/v1/chat/completions", model)
    headers = _build_upstream_headers(dict(request.headers), accept="text/event-stream" if caller_wants_stream else "application/json")

    _log(f"REQ chat/completions -> {url}")
    if LOG_REQUESTS:
        # Avoid printing secrets
        safe_headers = {k: ("<redacted>" if k.lower()=="authorization" else v) for k, v in headers.items()}
        _log(f"HEADERS: {safe_headers}")
        try:
            _log(f"BODY: {json.dumps(body)[:2000]}")
        except Exception:
            pass

    async with httpx.AsyncClient(timeout=None) as client:
        if caller_wants_stream:
            upstream_resp = await _stream_upstream(client, url, _ensure_stream_true(body), headers)
            if upstream_resp.status_code >= 400:
                # Pass through upstream errors
                text = await upstream_resp.aread()
                return Response(content=text, status_code=upstream_resp.status_code, media_type=upstream_resp.headers.get("content-type", "application/json"))

            return StreamingResponse(_normalize_stream(upstream_resp, model=model, mode="chat"), media_type="text/event-stream")

        # Non-stream path: forward, then clean fields
        # Non-stream: also retry on 429
        attempt = 0
        while True:
            upstream_resp = await client.post(url, json={**body, "stream": False}, headers=headers)
            if upstream_resp.status_code != 429 or attempt >= RETRY_429_MAX_ATTEMPTS:
                break
            delay_ms = min(RETRY_429_BASE_MS * (2 ** attempt), RETRY_429_MAX_MS)
            jitter = 0.5 + random.random()
            sleep_s = (delay_ms * jitter) / 1000.0
            if LOG_REQUESTS:
                _log(f"429 received (non-stream); retrying in {sleep_s:.2f}s (attempt {attempt+1}/{RETRY_429_MAX_ATTEMPTS})")
            await asyncio.sleep(sleep_s)
            attempt += 1
        if upstream_resp.status_code >= 400:
            text = await upstream_resp.aread()
            return Response(content=text, status_code=upstream_resp.status_code, media_type=upstream_resp.headers.get("content-type", "application/json"))

        try:
            data = upstream_resp.json()
        except Exception:
            # If upstream sent non-JSON, pass-through
            text = await upstream_resp.aread()
            return Response(content=text, status_code=upstream_resp.status_code, media_type=upstream_resp.headers.get("content-type", "application/json"))

        if LOG_REQUESTS:
            _log(f"UPSTREAM STATUS: {upstream_resp.status_code}")
            try:
                _log(f"UPSTREAM JSON: {json.dumps(data)[:2000]}")
            except Exception:
                pass

        # Non-stream hardening: reconstruct message.tool_calls from reasoning_content when absent
        try:
            choices = data.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    msg = choice.get("message")
                    if isinstance(msg, dict):
                        if not msg.get("tool_calls"):
                            rc = msg.get("reasoning_content") or ""
                            tool_calls = _synthesize_tool_calls_from_reasoning(rc)
                            if tool_calls:
                                msg["tool_calls"] = tool_calls
                                # Set finish_reason if not present
                                if not choice.get("finish_reason"):
                                    choice["finish_reason"] = "tool_calls"
        except Exception:
            pass

        cleaned = _strip_kimi_fields(data)
        return JSONResponse(cleaned)


# Compatibility alias without /v1 prefix
@app.post("/chat/completions", include_in_schema=False)
async def chat_completions_compat(request: Request) -> Response:
    return await chat_completions(request)


@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    if not UPSTREAM_BASE_URL:
        return JSONResponse({"error": "UPSTREAM_BASE_URL is not configured"}, status_code=500)

    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    model = str(body.get("model") or DEFAULT_MODEL or "")
    if not model:
        return JSONResponse({"error": "model is required"}, status_code=400)

    caller_wants_stream = bool(body.get("stream", True))

    # Compose final upstream body with overrides
    body = _inject_overrides(body)

    url = _build_target_url("/v1/completions", model)
    headers = _build_upstream_headers(dict(request.headers), accept="text/event-stream" if caller_wants_stream else "application/json")

    _log(f"REQ completions -> {url}")
    if LOG_REQUESTS:
        safe_headers = {k: ("<redacted>" if k.lower()=="authorization" else v) for k, v in headers.items()}
        _log(f"HEADERS: {safe_headers}")
        try:
            _log(f"BODY: {json.dumps(body)[:2000]}")
        except Exception:
            pass

    async with httpx.AsyncClient(timeout=None) as client:
        if caller_wants_stream:
            upstream_resp = await _stream_upstream(client, url, _ensure_stream_true(body), headers)
            if upstream_resp.status_code >= 400:
                text = await upstream_resp.aread()
                return Response(content=text, status_code=upstream_resp.status_code, media_type=upstream_resp.headers.get("content-type", "application/json"))

            return StreamingResponse(_normalize_stream(upstream_resp, model=model, mode="completion"), media_type="text/event-stream")

        attempt = 0
        while True:
            upstream_resp = await client.post(url, json={**body, "stream": False}, headers=headers)
            if upstream_resp.status_code != 429 or attempt >= RETRY_429_MAX_ATTEMPTS:
                break
            delay_ms = min(RETRY_429_BASE_MS * (2 ** attempt), RETRY_429_MAX_MS)
            jitter = 0.5 + random.random()
            sleep_s = (delay_ms * jitter) / 1000.0
            if LOG_REQUESTS:
                _log(f"429 received (non-stream); retrying in {sleep_s:.2f}s (attempt {attempt+1}/{RETRY_429_MAX_ATTEMPTS})")
            await asyncio.sleep(sleep_s)
            attempt += 1
        if upstream_resp.status_code >= 400:
            text = await upstream_resp.aread()
            return Response(content=text, status_code=upstream_resp.status_code, media_type=upstream_resp.headers.get("content-type", "application/json"))

        try:
            data = upstream_resp.json()
        except Exception:
            text = await upstream_resp.aread()
            return Response(content=text, status_code=upstream_resp.status_code, media_type=upstream_resp.headers.get("content-type", "application/json"))

        if LOG_REQUESTS:
            _log(f"UPSTREAM STATUS: {upstream_resp.status_code}")
            try:
                _log(f"UPSTREAM JSON: {json.dumps(data)[:2000]}")
            except Exception:
                pass

        cleaned = _strip_kimi_fields(data)
        return JSONResponse(cleaned)


# Compatibility alias without /v1 prefix
@app.post("/completions", include_in_schema=False)
async def completions_compat(request: Request) -> Response:
    return await completions(request)


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(prog="kimothy", description="Kimothy: OpenAI-compatible proxy for Kimi-like streaming providers")
    parser.add_argument("--preset", choices=["chutes"], help="Apply preset defaults (e.g., chutes)")
    # Short flags for common args
    parser.add_argument("-H", "--host", default=os.getenv("HOST", "0.0.0.0"), help="Bind host (default: 0.0.0.0)")
    parser.add_argument("-p", "--port", type=int, default=int(os.getenv("PORT", "8800")), help="Port (default: 8800)")
    parser.add_argument("-b", "--upstream-base-url", default=UPSTREAM_BASE_URL, help="Upstream base URL, e.g. https://llm.chutes.ai/v1")
    parser.add_argument("-k", "--upstream-api-key", default=UPSTREAM_API_KEY, help="Upstream Bearer token")
    parser.add_argument("-m", "--default-model", default=DEFAULT_MODEL, help="Default model if client omits one")
    # Routing + shaping
    parser.add_argument("-e", "--upstream-endpoint", choices=["chat/completions", "completions"], default=UPSTREAM_ENDPOINT, help="Force upstream endpoint (optional)")
    parser.add_argument("-u", "--url-model-placement", choices=["body", "path"], default=URL_MODEL_PLACEMENT, help="Model placement for upstream: body or path (default: body)")
    parser.add_argument("--ensure-stream", choices=["0", "1"], default="1" if ENSURE_STREAM else "0", help="Ensure stream: true in upstream body (default: 1)")
    parser.add_argument("--override-temperature", type=float, default=float(OVERRIDE_TEMPERATURE) if OVERRIDE_TEMPERATURE is not None else None, help="Override temperature (optional)")
    parser.add_argument("--override-max-tokens", type=int, default=int(OVERRIDE_MAX_TOKENS) if OVERRIDE_MAX_TOKENS is not None else None, help="Override max_tokens (optional)")
    # Logging + UX
    parser.add_argument("--log-requests", action="store_true", default=LOG_REQUESTS, help="Verbose request/response logging to stderr")
    parser.add_argument("--status-comments", choices=["0","1"], default="1" if STATUS_COMMENTS else "0", help="Emit ': stream-open' and ': stream-close' comments (default: 1)")
    parser.add_argument("--heartbeat-ms", type=int, default=HEARTBEAT_MS, help="Heartbeat comment every N ms (0 to disable)")
    parser.add_argument("--tty-spinner", choices=["0","1"], default="1" if TTY_SPINNER else "0", help="Show terminal spinner during streams (default: 1)")
    # Tool-call semantics
    parser.add_argument("--finish-on-tool-end", choices=["0", "1"], default="1" if FINISH_ON_TOOL_END else "0", help='Emit finish chunk with finish_reason="tool_calls" on tool end (default: 1)')
    parser.add_argument("--auto-finish-toolcalls", choices=["0","1"], default="1" if AUTO_FINISH_TOOLCALLS else "0", help="Auto-emit tool_calls finish after idle (default: 1)")
    parser.add_argument("--auto-finish-delay-ms", type=int, default=AUTO_FINISH_DELAY_MS, help="Idle delay before auto-finish (default: 800)")
    # Retry 429
    parser.add_argument("--retry-429-attempts", type=int, default=RETRY_429_MAX_ATTEMPTS, help="Max retries for 429 (default: 3)")
    parser.add_argument("--retry-429-base-ms", type=int, default=RETRY_429_BASE_MS, help="Base backoff ms for 429 (default: 500)")
    parser.add_argument("--retry-429-max-ms", type=int, default=RETRY_429_MAX_MS, help="Max backoff ms for 429 (default: 4000)")
    args = parser.parse_args()

    # Update globals from CLI args
    # Apply preset defaults if provided
    if getattr(args, "preset", None) == "chutes":
        if not args.upstream_base_url:
            args.upstream_base_url = "https://llm.chutes.ai/v1"
        if not args.default_model:
            args.default_model = "moonshotai/Kimi-K2-Thinking"

    # OpenAI env fallback if UPSTREAM_* not set
    if not args.upstream_base_url and os.getenv("OPENAI_BASE_URL"):
        args.upstream_base_url = os.getenv("OPENAI_BASE_URL")
    if not args.upstream_api_key and os.getenv("OPENAI_API_KEY"):
        args.upstream_api_key = os.getenv("OPENAI_API_KEY")

    UPSTREAM_BASE_URL = args.upstream_base_url
    UPSTREAM_API_KEY = args.upstream_api_key
    DEFAULT_MODEL = args.default_model
    ENSURE_STREAM = args.ensure_stream == "1"
    URL_MODEL_PLACEMENT = args.url_model_placement
    UPSTREAM_ENDPOINT = args.upstream_endpoint
    OVERRIDE_TEMPERATURE = str(args.override_temperature) if args.override_temperature is not None else None
    OVERRIDE_MAX_TOKENS = str(args.override_max_tokens) if args.override_max_tokens is not None else None
    LOG_REQUESTS = args.log_requests
    FINISH_ON_TOOL_END = args.finish_on_tool_end == "1"
    HEARTBEAT_MS = int(args.heartbeat_ms or 0)
    STATUS_COMMENTS = args.status_comments == "1"
    TTY_SPINNER = args.tty_spinner == "1"
    AUTO_FINISH_TOOLCALLS = args.auto_finish_toolcalls == "1"
    AUTO_FINISH_DELAY_MS = int(args.auto_finish_delay_ms or 800)
    RETRY_429_MAX_ATTEMPTS = int(args.retry_429_attempts or RETRY_429_MAX_ATTEMPTS)
    RETRY_429_BASE_MS = int(args.retry_429_base_ms or RETRY_429_BASE_MS)
    RETRY_429_MAX_MS = int(args.retry_429_max_ms or RETRY_429_MAX_MS)

    # Important: pass the app object directly to avoid a second import
    # which would ignore CLI-updated globals.
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
