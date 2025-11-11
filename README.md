# Kimothy: OpenAI-Compatible Streaming Proxy for Kimi-like Providers

This FastAPI proxy normalizes Kimi-like streaming responses into OpenAI-compatible outputs so existing OpenAI clients and CLI agents “just work.” It transparently adapts request/response shapes, handles odd streaming/reasoning formats, and preserves tool-call semantics expected by typical agent loops.

Key features
- OpenAI API surface: `/v1/chat/completions`, `/v1/completions` (+ aliases without `/v1`).
- Streaming normalization:
  - Converts provider SSE to OpenAI `chat.completion.chunk`/`text_completion.chunk` deltas.
  - Emits `role=assistant` once on the first delta.
  - Synthesizes incremental `tool_calls` deltas from Kimi reasoning markers.
  - Optional legacy `function_call` deltas for older clients.
  - Guarantees `data: [DONE]` terminator.
- Non‑stream cleaning:
  - Recursively strips provider‑specific fields (e.g., `reasoning_content`, `reasoning_tokens`, `thinking`, `k2_metadata`).
- Request shaping:
  - Ensures `stream: true` (configurable) to enable SSE normalization.
  - Model placement in URL path or request body.
  - Upstream base URL with or without `/v1` supported.
- Tool-call finish semantics (on by default):
  - Emits a finish chunk with `finish_reason="tool_calls"` when a tool call ends (can be disabled).
- Robust logging for debugging:
  - Log target URLs, headers (sanitized), request/response JSON, upstream/downstream SSE lines.
 - Terminal spinner by default:
   - Shows a live spinner in the proxy terminal during active streams (prints to stderr). Disable with `--tty-spinner 0`.


## Quickstart

1) Install deps (Python 3.11+):
- `pip install -r requirements.txt`

- 2) Run the proxy (CLI flags or env vars):
- Quick preset (Chutes):
```
python kimothy.py --preset chutes -k <YOUR_KEY> -p 8928
```
- Short flags (custom base/model):
```
python kimothy.py -b https://llm.chutes.ai/v1 -k <YOUR_KEY> -m moonshotai/Kimi-K2-Thinking -p 8928 -u body --ensure-stream 1
```
- Same via env:
```
export UPSTREAM_BASE_URL=https://llm.chutes.ai/v1
export UPSTREAM_API_KEY=<YOUR_KEY>
export DEFAULT_MODEL=moonshotai/Kimi-K2-Thinking
export ENSURE_STREAM=1
uvicorn kimothy:app --host 0.0.0.0 --port 8928
```

3) Call it like OpenAI:
- Non‑stream JSON
```
curl -sS -X POST http://127.0.0.1:8928/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"moonshotai/Kimi-K2-Thinking",
    "messages":[{"role":"user","content":"Say hello"}],
    "stream":false
  }'
```
- Stream SSE
```
curl -sN -X POST http://127.0.0.1:8928/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'Accept: text/event-stream' \
  -d '{
    "model":"moonshotai/Kimi-K2-Thinking",
    "messages":[{"role":"user","content":"Say hello"}],
    "stream":true
  }'
```

Aliases `/chat/completions` and `/completions` are also accepted.


## Endpoints

- `POST /v1/chat/completions`
  - Streaming: emits `chat.completion.chunk` deltas; forwards `content` and emits incremental `tool_calls`.
  - Non‑stream: returns OpenAI‑like JSON, stripping provider‑only fields.
- `POST /v1/completions`
  - Streaming: emits `text_completion.chunk` deltas with `choices[].text`.
  - Non‑stream: returns standard OpenAI text completion JSON.
- `GET /health`: simple health check.


## Configuration

You may set these via environment variables or CLI flags.

Core
- `UPSTREAM_BASE_URL` / `--upstream-base-url`
  - Provider base URL, with or without `/v1` (proxy avoids double‑versioning). Example: `https://llm.chutes.ai/v1`.
- `UPSTREAM_API_KEY` / `--upstream-api-key`
  - Provider Bearer token. Forwarded in `Authorization` header.
- `DEFAULT_MODEL` / `--default-model`
  - Used if client omits `model`.

Shaping and routing
- `ENSURE_STREAM` / `--ensure-stream {0|1}` (default: 1)
  - Forces `stream: true` upstream for streaming calls so the proxy can normalize.
- `URL_MODEL_PLACEMENT` / `--url-model-placement {body|path}` (default: body)
  - `path` yields upstream URLs like `/v1/models/{model}/chat/completions`.
- `UPSTREAM_ENDPOINT` / `--upstream-endpoint {chat/completions|completions}` (optional)
  - Force a specific upstream endpoint regardless of caller path.
- `OVERRIDE_TEMPERATURE` / `--override-temperature <float>` (optional)
- `OVERRIDE_MAX_TOKENS` / `--override-max-tokens <int>` (optional)

Logging and debugging
- `LOG_REQUESTS=1`
  - Logs target upstream URL, sanitized headers, outgoing body, upstream status, and upstream JSON (non‑stream). 
- `LOG_UPSTREAM_LINES=1`
  - Logs every upstream SSE line (`UPSTREAM <- data: ...`).
- `LOG_DOWNSTREAM_LINES=1`
  - Logs every downstream SSE line (`DOWNSTREAM -> data: ...`).

Tool-call compatibility
- `EMIT_FUNCTION_CALL_LEGACY=1` (default)
  - Also emits `delta.function_call` alongside `delta.tool_calls` for older clients.
- `FINISH_ON_TOOL_END=1` / `--finish-on-tool-end {0|1}` (default: 1)
  - Emits a finish chunk with `finish_reason="tool_calls"` when a tool call ends (does not end the stream).
- `AUTO_FINISH_TOOLCALLS=1` / `--auto-finish-toolcalls {0|1}` (default: 1)
  - If upstream never signals tool end, auto-emit a `finish_reason="tool_calls"` after a short idle delay.
- `AUTO_FINISH_DELAY_MS` / `--auto-finish-delay-ms` (default: 800)
  - Idle delay before auto finish.
- `TTY_SPINNER=1` / `--tty-spinner {0|1}` (default: 1)
  - Shows a live spinner in the proxy terminal during active streams.


## Streaming behavior and tool calls

- Content deltas
  - The proxy forwards `delta.content` as standard OpenAI chunks and emits `role=assistant` only on the first delta.
- Tool calls from upstream `tool_calls`
  - If provider includes `tool_calls` directly, the proxy converts arguments into incremental deltas (by diffing vs accumulated state).
- Tool calls from Kimi reasoning markers
  - Many Kimi providers encode tool intent and arguments inside `reasoning_content` using markers like:
    - `<|tool_calls_section_begin|>`, `<|tool_call_begin|>`, `<|tool_call_argument_begin|>`, `<|tool_call_argument_end|>`, `<|tool_call_end|>`, `<|tool_calls_section_end|>`
  - The proxy parses these markers and synthesizes OpenAI `tool_calls` deltas:
    - Assigns stable synthetic IDs per tool index (e.g., `call_0_...`).
    - Streams only the newly appended portion of `function.arguments` per chunk (OpenAI delta semantics).
  - Optionally emits a finish chunk with `finish_reason="tool_calls"` at tool end when enabled.
  - If upstream does not emit an explicit end, the proxy auto-emits `finish_reason="tool_calls"` after brief inactivity (default on).

- Non-stream reconstruction
  - When the upstream non-stream response does not include `message.tool_calls` but includes `reasoning_content`, the proxy reconstructs a `tool_calls` array from markers and sets `finish_reason="tool_calls"` for that choice.


## Troubleshooting

- 404 on `/chat/completions`:
  - Use `/v1/chat/completions` or keep using `/chat/completions` (aliases added). If you see 404s, verify you’re calling the proxy, not the upstream.
- `UPSTREAM_BASE_URL is not configured` while passing CLI flags:
  - Ensure you run `python kimothy.py ...` (the CLI updates globals). We run `uvicorn.run(app, ...)` to avoid double‑import; don’t launch uvicorn with the module string target unless you set env vars.
- h11 `Too little data for declared Content-Length`:
  - The proxy strips `Content-Length` and other hop‑by‑hop headers before proxying. If you modified headers, ensure you aren’t forwarding `Content-Length`.
- Not seeing tool calls in clients:
  - Enable verbose logging: `LOG_UPSTREAM_LINES=1 LOG_DOWNSTREAM_LINES=1`.
  - If your provider encodes tools differently (non‑Kimi markers), share a sample stream so we can add a mapper.
- Double `/v1` or missing `/v1` upstream URLs:
  - The proxy detects whether `UPSTREAM_BASE_URL` ends with `/v1` and adjusts accordingly.


## Security notes

- Logs redact the `Authorization` header automatically when `LOG_REQUESTS=1`.
- Consider disabling detailed logs in production; SSE line logs may contain model output.


## Development tips

- File: `kimothy.py`
  - Entrypoint CLI and FastAPI app in a single file.
  - Stream normalization within `_normalize_stream`.
- Run with reload (dev only):
  - `uvicorn kimothy:app --reload --port 8928` with env vars set; avoid if you rely on CLI flags.
- Tests: Add cURL scripts or pytest-asyncio as needed.


## Limitations and roadmap

- If upstream never exposes tool names and only streams generic markers, the parser guesses names (e.g., from `functions.<name>`). Provide a sample if your provider uses a different convention.
- Non‑stream responses do not currently reconstruct `message.tool_calls` from captured reasoning markers. This can be added if your client requires complete tool calls in non‑stream mode.
- Optional chunk coalescing (buffering N ms/characters) can be added to reduce very fine‑grained streams.
