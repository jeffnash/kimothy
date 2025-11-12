"""Client-agnostic tool call sanitization and schema-based validation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .json_utils import JSONProcessor


def _python_type_for(schema_type: str):
    return {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }.get(schema_type, None)


def _schema_validate(name: str, args: dict, tool_schemas: dict | None) -> tuple[bool, dict]:
    """Validate args against a provided tool schema. Returns (is_valid, possibly_filtered_args)."""
    if not tool_schemas or name not in tool_schemas:
        return True, args

    schema = tool_schemas.get(name) or {}
    required = schema.get("required", []) or []
    properties = schema.get("properties", {}) or {}
    additional = schema.get("additionalProperties", True)

    # Must include all required keys (non-empty unless value can be falsy like 0 or False)
    for rk in required:
        if rk not in args or args.get(rk) in (None, ""):
            return False, args

    # Basic type checks for simple types
    for key, val in list(args.items()):
        prop = properties.get(key)
        if not prop:
            if additional is False:
                args.pop(key, None)
            continue
        expected_type = prop.get("type")
        if expected_type:
            py_t = _python_type_for(expected_type)
            if py_t and not isinstance(val, py_t):
                return False, args

    return True, args


def sanitize_tool_calls(
    tool_calls: List[Dict[str, Any]] | None,
    *,
    tool_schemas: Dict[str, Any] | None = None,
    enable_heuristics: bool = True,
    keep_invalid: bool = False,
) -> List[Dict[str, Any]]:
    """Validate and deduplicate tool_calls before emission in a client-agnostic way.

    Strategy:
    - If tool schemas are provided (OpenAI Tools JSON Schema), validate args against schema
      (required keys, basic type checks, filter unknown keys when additionalProperties=false).
    - If no schemas are available, optionally fall back to light heuristics for common tools
      (Read/Glob/Create/Edit) to avoid obviously invalid calls.
    - Deduplicate identical (name, arguments) pairs.
    """
    if not tool_calls:
        return []

    def coalesce(dst: str, normalized: dict, *srcs: str) -> None:
        if dst in normalized:
            return
        for s in srcs:
            if s in normalized and normalized[s] not in (None, ""):
                normalized[dst] = normalized[s]
                break

    seen: set[tuple[str, str]] = set()
    sanitized: List[Dict[str, Any]] = []

    for tc in tool_calls:
        try:
            fn = (tc or {}).get("function", {})
            name = fn.get("name")
            args_text = fn.get("arguments", "")
            args_obj = JSONProcessor.parse_object_safe(args_text) or {}

            # Canonicalize common alias keys in a copy (non-destructive to original)
            normalized = dict(args_obj)

            # 1) Schema validation (client-agnostic best-effort)
            valid, filtered_args = _schema_validate(name, dict(normalized), tool_schemas)

            # 2) Heuristics when no schema is provided
            if not tool_schemas and enable_heuristics:
                if name == "Read":
                    coalesce("file_path", normalized, "filePath", "path")
                    fp = normalized.get("file_path")
                    valid = isinstance(fp, str) and bool(fp.strip())
                elif name == "Glob":
                    coalesce("folder", normalized, "directory", "dir", "path")
                    pats = normalized.get("patterns")
                    folder = normalized.get("folder")
                    valid = (isinstance(pats, list) and bool(pats) and isinstance(folder, str) and bool(folder.strip()))
                elif name in {"Create", "CreateFile", "Write", "WriteFile"}:
                    # Without schemas, allow any non-empty JSON object for writes/creates.
                    # We still coalesce common keys if present, but do not require them.
                    coalesce("file_path", normalized, "filePath", "path")
                    coalesce("content", normalized, "text", "code", "data", "newContent")
                    valid = isinstance(normalized, dict) and bool(normalized)
                elif name in {"Edit", "ApplyPatch", "Patch", "Modify", "Change", "Append"}:
                    coalesce("file_path", normalized, "filePath", "path")
                    fp = normalized.get("file_path")
                    edits = normalized.get("edits")
                    patch = normalized.get("patch") or normalized.get("diff")
                    search = normalized.get("search") or normalized.get("pattern")
                    replace = normalized.get("replace") or normalized.get("replacement") or normalized.get("new_text")
                    content = normalized.get("content") or normalized.get("newContent")
                    # Allow non-empty JSON object edits when schemas are absent; prefer but not require classic patterns.
                    has_change = (
                        (isinstance(edits, list) and len(edits) > 0) or
                        (isinstance(patch, str) and patch.strip() != "") or
                        (isinstance(search, str) and isinstance(replace, str)) or
                        (isinstance(content, str) and content.strip() != "") or
                        (len(normalized.keys()) > 0)
                    )
                    valid = isinstance(normalized, dict) and has_change

            if not valid:
                if keep_invalid:
                    # Keep the original tool call without modification
                    sanitized.append(tc)
                    continue
                # Drop invalid tool call silently (callers can observe via surrounding logs)
                continue

            key = (name, args_text)
            if key in seen:
                # Deduplicate identical calls
                continue
            seen.add(key)

            # If we normalized or filtered args, update arguments string
            try:
                updated = filtered_args if tool_schemas and filtered_args is not None else normalized
                if updated and updated != args_obj:
                    tc = dict(tc)
                    fn_copy = dict(tc.get("function", {}))
                    fn_copy["arguments"] = json.dumps(updated)
                    tc["function"] = fn_copy
            except Exception:
                pass

            sanitized.append(tc)
        except Exception:
            # On unexpected errors, keep original to avoid data loss
            sanitized.append(tc)

    return sanitized
