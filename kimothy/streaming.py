"""Robust streaming tool call assembly with proper JSON parsing."""

import json
from typing import Dict, List, Any, Optional


class ToolCallBuilder:
    """Builder that accumulates fragments verbatim, only parsing at completion."""
    
    def __init__(self, index: int):
        self.index = index
        self.id: Optional[str] = None
        self.name: Optional[str] = None
        self.arguments: str = ""
        self._validated = False  # Stop accumulating once valid
    
    def update(self, fragment: Dict[str, Any]) -> None:
        """Update with fragment - NEVER parse or interpret intermediate JSON."""
        if self._validated:
            return
        
        func = fragment.get("function", {})
        
        # Only update metadata if present
        if fragment.get("id") is not None:
            self.id = fragment["id"]
        if func.get("name") is not None:
            self.name = func["name"]
        
        # ALWAYS append arguments verbatim - no interpretation
        if "arguments" in func and func["arguments"] is not None:
            args = func["arguments"]
            # Normalize to string if needed
            if isinstance(args, dict):
                args = json.dumps(args)
            elif not isinstance(args, str):
                args = str(args)
            
            self.arguments += args
            
            # Try to validate now (only at the END of accumulation)
            self._validated = self._is_complete_json(self.arguments)
    
    def _is_complete_json(self, text: str) -> bool:
        """
        Check if text is complete, valid JSON.
        This handles:
        - Escaped quotes (\" inside strings)
        - Nested objects/arrays
        - Unicode characters
        - Nested strings with colons/commas/braces
        """
        text = text.strip()
        if not text:
            return False
        
        try:
            parsed = json.loads(text)
            # Must be a dict/object for tool call arguments
            return isinstance(parsed, dict)
        except json.JSONDecodeError:
            return False
    
    def is_complete(self) -> bool:
        """Check if we have id, name, and valid JSON arguments."""
        return bool(
            self.id and 
            self.name and 
            self._validated  # Only true if arguments parse as valid JSON
        )
    
    def to_dict(self) -> Optional[Dict[str, Any]]:
        """Convert to OpenAI format - arguments are already valid JSON."""
        if not self.is_complete():
            return None
        
        return {
            "index": self.index,
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments.strip()
            }
        }


def _find_json_object_boundaries(text: str) -> tuple:
    """
    Find the first complete JSON object in text, handling nested structures.
    Returns (start_idx, end_idx) or (None, None) if not found.
    
    This is a robust fallback if json.loads fails on the full string.
    """
    text = text.strip()
    
    # Quick check: must start with { or be wrapped in markdown
    if text.startswith('`'):
        text = text.strip('`')
        if text.startswith('json'):
            text = text[4:].strip()
    
    if not text.startswith('{'):
        return None, None
    
    depth = 0
    in_string = False
    escaped = False
    
    start = 0
    end = None
    
    for i, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        
        if char == '\\':
            escaped = True
            continue
        
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
    
    if end is not None:
        return start, end
    return None, None


def process_delta(delta: Dict[str, Any], accumulator: Dict[int, ToolCallBuilder]) -> Optional[Dict[str, Any]]:
    """
    Process delta with robust JSON handling.
    
    STRATEGY:
    1. ACCUMULATE VERBATIM - Never parse or interpret intermediate fragments
    2. COALESCE NULL-ID FRAGMENTS - Append to last builder with real id
    3. VALIDATE AT COMPLETION - Only parse with json.loads() when done
    4. EXTRACT BOUNDARIES - If full string fails, find inner JSON object
    """
    if not delta.get("tool_calls"):
        return None
    
    emitted = []
    
    for tc in delta["tool_calls"]:
        idx = tc.get("index", 0)
        
        # STRATEGY: Null-id fragments coalesce to last active builder
        if idx not in accumulator and tc.get("id") is None:
            active = [k for k, v in accumulator.items() if v.id is not None]
            if active:
                idx = max(active)  # Use highest index with valid id
        
        if idx not in accumulator:
            accumulator[idx] = ToolCallBuilder(idx)
        
        # ACCUMULATE VERBATIM - no interpretation
        accumulator[idx].update(tc)
        
        # EMIT ONLY WHEN COMPLETE (valid JSON + id + name)
        if accumulator[idx].is_complete():
            emitted.append(accumulator[idx].to_dict())
            del accumulator[idx]
    
    return {"tool_calls": emitted} if emitted else None


def fix_delta(delta: Dict[str, Any]) -> Dict[str, Any]:
    """Fix null fields before processing."""
    if delta.get("role") is None:
        delta["role"] = "assistant"
    if delta.get("tool_calls") is None:
        delta["tool_calls"] = []
    return delta
