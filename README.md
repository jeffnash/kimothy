# Kimothy (v2): OpenAI‑Compatible Streaming Proxy

Kimothy is an OpenAI‑compatible proxy that normalizes streaming and non‑streaming responses from Kimi‑like providers (and others) into standard OpenAI Chat/Completions shapes. This rewrite modularizes the codebase, strengthens SSE parsing, and robustly reconstructs tool calls from upstream deltas and Kimi‑style reasoning markers — while staying client‑agnostic and safe by default.

Highlights
- OpenAI surface: `/v1/chat/completions`, `/v1/completions` (+ aliases without `/v1`).
- Providers: pluggable backends under `kimothy/providers` (Chutes, Kimi, OpenAI, Anthropic, etc.).
- Streaming: resilient SSE parser with CRLF normalization, multi‑line `data:` join, and final‑fragment flush.
- Tool calls: 
  - Parse upstream `choices[].delta.tool_calls` and assemble arguments incrementally.
  - Synthesize tool calls from Kimi reasoning markers (`<|tool_calls_section_begin|>`, `<|tool_call_begin|>`, `<|tool_call_argument_begin|>`, `<|tool_call_end|>`, `<|tool_calls_section_end|>`), ignoring markers in code fences and string literals.
  - End‑of‑stream flush emits pending tools if arguments are valid JSON.
- Client‑agnostic validation: 
  - If the client supplies tool schemas (OpenAI Tools JSON Schema), validate against them.
  - Otherwise, safe heuristics: strict for path‑sensitive tools (Read/Glob), permissive for schema‑less domain writes/edits.
- Logging & safety: structured logs, sanitized headers; avoids emitting invalid tool calls.


## Install

Requires Python 3.11+

```
pip install -r requirements.txt
```


## Run

CLI (recommended)

```
python kimothy_cli.py \
  --preset chutes \
  -k <UPSTREAM_API_KEY> \
  --default-model moonshotai/Kimi-K2-Thinking \
  -p 8928
```

Or via env + uvicorn

```
export UPSTREAM_BASE_URL=https://llm.chutes.ai/v1
export UPSTREAM_API_KEY=<YOUR_KEY>
export DEFAULT_MODEL=moonshotai/Kimi-K2-Thinking
uvicorn kimothy.main:app --host 0.0.0.0 --port 8928
```

## Call it like OpenAI

Non‑stream
```
curl -sS -X POST http://127.0.0.1:8928/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"moonshotai/Kimi-K2-Thinking",
    "messages":[{"role":"user","content":"Say hello"}],
    "stream":false
  }'
```
Stream (SSE)
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
- `GET /health` — Simple health check.


## Configuration

You may set via CLI flags or env vars (CLI takes precedence):

- `UPSTREAM_BASE_URL` / `--upstream-base-url` — Provider base URL (with or without `/v1`).
- `UPSTREAM_API_KEY` / `--upstream-api-key` — Provider Bearer token.
- `DEFAULT_MODEL` / `--default-model` — Used if client omits `model`.
- `ENSURE_STREAM` / `--ensure-stream {0|1}` — Force stream mode upstream (default: 1).
- `UPSTREAM_ENDPOINT` / `--upstream-endpoint` — Force specific upstream endpoint.
- `OVERRIDE_TEMPERATURE`, `OVERRIDE_MAX_TOKENS` — Optional overrides.

Logging & debugging
- `LOG_REQUESTS=1` — Logs incoming/outgoing meta.
- `LOG_UPSTREAM_LINES=1` — Log every upstream SSE line.
- `LOG_DOWNSTREAM_LINES=1` — Log every downstream SSE line.
- `EXPOSE_REASONING_AS_CONTENT=1` — Debug option to forward `reasoning_content` as normal `delta.content` text.

Tool-call compatibility
Retry on 429
- `RETRY_429_MAX_ATTEMPTS` / `--retry-429-attempts` (default: 6)
  - Max attempts when upstream returns 429.
- `RETRY_429_BASE_MS` / `--retry-429-base-ms` (default: 500)
  - Base backoff delay in ms (exponential).
- `RETRY_429_MAX_MS` / `--retry-429-max-ms` (default: 30000)
  - Max delay cap per attempt.
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


## Streaming and Tool Calls (v2 behavior)

- SSE parsing
  - Normalizes CRLF, joins multi‑line `data:` fields, and flushes trailing fragments when the upstream omits the final blank line.

- Upstream tool_calls
  - Incrementally assembles `function.arguments`, emits only newly appended slices, coalesces null ids, and robustly merges fragmented deltas.

- Kimi reasoning markers
  - Parses markers only outside code fences and JSON string literals; builds OpenAI‑style `tool_calls` with stable ids and streamed arguments.
  - If an explicit end marker never arrives, flushes once at end‑of‑stream when arguments are valid JSON.

- Validation
  - With tool schemas in the request, arguments are validated against JSON Schema (required keys, basic types, additionalProperties handling).
  - Without schemas, Read/Glob remain strict to avoid invalid paths, while write/edit remain permissive for non‑file domain payloads.


## Troubleshooting

- Not seeing final chunks:
  - Ensure upstream isn’t ending the SSE stream without a blank line; v2 flushes trailing fragments. Enable `LOG_UPSTREAM_LINES=1` to confirm.
- Tool calls missing or “invalid path” reads:
  - Check that tools schemas are provided; otherwise heuristics apply (strict for Read/Glob). See logs for “Dropping invalid tool call”.
- Markers in code/docs mis‑parsed as tools:
  - v2 masks triple‑backtick code fences and ignores markers inside JSON strings.


## Security

- Authorization headers are sanitized in logs; disable full line logging in production.


## Project Layout

- `kimothy/main.py` — FastAPI app and streaming pipeline.
- `kimothy/providers/*` — Upstream adapters and reasoning parser.
- `kimothy/utils/*` — SSE parsing, JSON helpers, tool call processing, sanitization, logging.
- `kimothy/streaming.py` — Low‑level incremental tool assembly.
- `kimothy_cli.py` — CLI entrypoint (sets env, runs uvicorn).


## Migration Notes (from v1 monolith)

- `kimothy.py` (monolith) has been split into modules under `kimothy/`.
- Use `python kimothy_cli.py` (or `uvicorn kimothy.main:app`) instead of `python kimothy.py`.
- Environment/config moved to `kimothy/config.py` (pydantic‑settings).
- New utils for SSE (`kimothy/utils/sse.py`), tool calls (`kimothy/utils/tool_call.py`), schemas/sanitizer.
