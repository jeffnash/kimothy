"""Utilities for extracting tool schemas from requests."""

from typing import Dict, Any


def build_tool_schemas(request_data: Dict[str, Any] | None) -> Dict[str, Any]:
    """Extract a mapping of tool name -> JSON Schema from request data.

    Expects OpenAI-style tools entries: {"type": "function", "function": {"name": str, "parameters": {...}}}
    Returns an empty dict if no tools are present or schema is malformed.
    """
    schemas: Dict[str, Any] = {}
    if not isinstance(request_data, dict):
        return schemas

    try:
        tools = request_data.get("tools") or []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            params = fn.get("parameters") or {}
            if isinstance(name, str) and isinstance(params, dict) and name:
                schemas[name] = params
    except Exception:
        # Be tolerant of malformed requests
        return {}

    return schemas

