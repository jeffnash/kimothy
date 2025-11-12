"""Handles Kimi-style reasoning content parsing."""

import re
import logging
from typing import List, Dict, Any, Optional, Iterator
from ..models import ChunkedReasoningEvent

_logger = logging.getLogger(__name__)


class ReasoningHandler:
    """Parses Kimi-style tool call reasoning markers from reasoning_content."""
    
    # Kimi marker patterns
    MARKERS = {
        "section_begin": r"<\|tool_calls_section_begin\|>",
        "section_end": r"<\|tool_calls_section_end\|>",
        "tool_call_begin": r"<\|tool_call_begin\|>(?:\s*([^<]+)\s*)?<\|tool_call_end\|>",
        "tool_call_start": r"<\|tool_call_begin\|>",
        "tool_call_end": r"<\|tool_call_end\|>",
        "args_begin": r"<\|tool_call_argument_begin\|>",
        "args_end": r"<\|tool_call_argument_end\|>",
    }
    
    def __init__(self) -> None:
        self.buffer = ""
        self.inside_section = False
        self.current_tool_index = 0
        self.tool_buffer = ""
        self.inside_tool = False
        self.tool_name = None
        self.inside_args = False
        self.args_buffer = ""
        
    def feed(self, text: str) -> List[ChunkedReasoningEvent]:
        """Feed reasoning content and return parsed events."""
        if not text:
            return []
            
        self.buffer += text
        events = []
        
        # Check for section begin/end
        if not self.inside_section:
            match = re.search(self.MARKERS["section_begin"], self.buffer)
            if match:
                self.inside_section = True
                self.buffer = self.buffer[match.end():]
                
        if self.inside_section:
            # Look for section end
            section_end_match = re.search(self.MARKERS["section_end"], self.buffer)
            working_text = self.buffer
            if section_end_match:
                working_text = self.buffer[:section_end_match.start()]
                
            # Parse tools within section
            pos = 0
            while pos < len(working_text):
                if not self.inside_tool:
                    # Look for tool start
                    tool_start_match = re.search(self.MARKERS["tool_call_start"], working_text[pos:])
                    if tool_start_match:
                        self.inside_tool = True
                        self.tool_start_pos = pos + tool_start_match.end()
                        pos = self.tool_start_pos
                        
                        # Look for tool name pattern
                        name_match = re.search(r"([^<]+)", working_text[pos:pos+200])
                        if name_match:
                            self.tool_name = name_match.group(1).strip()
                            # Skip the name content
                            name_end_match = re.search(self.MARKERS["tool_call_end"], working_text[pos:])
                            if name_end_match:
                                pos += name_end_match.end()
                                self.inside_tool = False
                                # Start a new tool with this name
                                events.append(ChunkedReasoningEvent(
                                    type="begin",
                                    index=self.current_tool_index,
                                    name=self.tool_name
                                ))
                                self.inside_args = True
                                self.args_buffer = ""
                                continue
                else:
                    # Find args boundaries within tool
                    if not self.inside_args:
                        args_begin_match = re.search(self.MARKERS["args_begin"], working_text[pos:])
                        if args_begin_match:
                            self.inside_args = True
                            pos += args_begin_match.end()
                            self.args_buffer = ""
                            continue
                    else:
                        # Looking for args end or more content
                        args_end_match = re.search(self.MARKERS["args_end"], working_text[pos:])
                        if args_end_match:
                            # Args section ended, capture what we have
                            if self.args_buffer:
                                events.append(ChunkedReasoningEvent(
                                    type="args",
                                    index=self.current_tool_index,
                                    text=self.args_buffer
                                ))
                                self.args_buffer = ""
                            self.inside_args = False
                            self.inside_tool = False
                            pos += args_end_match.end()
                            
                            events.append(ChunkedReasoningEvent(
                                type="end",
                                index=self.current_tool_index
                            ))
                            self.current_tool_index += 1
                            continue
                        else:
                            # Collect args content
                            next_marker_pos = len(working_text)
                            for marker_pattern in [self.MARKERS["args_end"]]:
                                marker_match = re.search(marker_pattern, working_text[pos:])
                                if marker_match:
                                    next_marker_pos = min(next_marker_pos, pos + marker_match.start())
                            
                            if next_marker_pos > pos:
                                self.args_buffer += working_text[pos:next_marker_pos]
                                pos = next_marker_pos
                            else:
                                self.args_buffer += working_text[pos:]
                                pos = len(working_text)
                                break
                
                pos += 1
            
            # Update buffer
            if section_end_match:
                self.buffer = self.buffer[section_end_match.end():]
                self.inside_section = False
                self.current_tool_index = 0
            else:
                self.buffer = ""
                
        return events
    
    def get_synthetic_tool_calls(self, completed_args: Dict[int, str]) -> List[Dict[str, Any]]:
        """Build tool calls from completed arguments."""
        tool_calls = []
        for idx, args in completed_args.items():
            tool_calls.append({
                "type": "function",
                "function": {
                    "name": f"tool_{idx}",
                    "arguments": args
                }
            })
        return tool_calls


class SimpleReasoningParser:
    """Lightweight reasoning parser for Kimi-style markers."""
    
    def __init__(self) -> None:
        self.complete_calls: Dict[int, str] = {}
        self.pending_calls: Dict[int, str] = {}
        self.buffer = ""
        self.inside_section = False
        self.inside_tool = False
        self.tool_name = None
        self.inside_args = False
        self.args_buffer = ""
        self.current_index = 0
        self.name_start = -1
        self._code_fence_re = re.compile(r"```.*?```", re.DOTALL)

    def _mask_code_fences(self, text: str) -> str:
        """Mask triple-backtick code fences to avoid treating literal markers as real."""
        if not text:
            return text
        # Replace code-fenced regions with spaces to preserve indices
        def _repl(m: re.Match) -> str:
            return " " * (m.end() - m.start())
        return self._code_fence_re.sub(_repl, text)
        
    def extract_tool_calls(self, reasoning: str) -> List[Dict[str, Any]]:
        """Extract tool calls from Kimi-style reasoning markers."""
        if not reasoning:
            return []
            
        self.buffer += reasoning
        tool_calls = []
        
        # Look for Kimi markers in the accumulated buffer
        import re
        
        marker_section_begin = r"<\|tool_calls_section_begin\|>"
        marker_begin = r"<\|tool_call_begin\|>"
        marker_args = r"<\|tool_call_argument_begin\|>"
        marker_tool_end = r"<\|tool_call_end\|>"
        marker_section_end = r"<\|tool_calls_section_end\|>"
        
        # Process buffer in a loop to handle multiple tools persisting across calls
        while self.buffer:
            scan_buffer = self._mask_code_fences(self.buffer)
            # Section gating: only parse tools inside a section
            if not self.inside_section:
                section_begin_match = re.search(marker_section_begin, scan_buffer)
                begin_match_probe = re.search(marker_begin, scan_buffer)
                if section_begin_match and (
                    not begin_match_probe or section_begin_match.start() <= begin_match_probe.start()
                ):
                    _logger.debug("Found section begin marker")
                    self.inside_section = True
                    self.buffer = self.buffer[section_begin_match.end():]
                    continue
                # Implicit section: if a tool_call_begin appears outside a section (and not in code), treat it as start
                if begin_match_probe:
                    _logger.debug("Implicit section start via tool_call_begin")
                    self.inside_section = True
                    self.inside_tool = True
                    self.buffer = self.buffer[begin_match_probe.end():]
                    self.name_start = 0
                    continue
                # No section begin or tool begin; stop parsing to avoid false positives in plans/docs
                break

            # If inside section, also allow closing it when not in a tool
            section_end_match_top = re.search(marker_section_end, scan_buffer)
            if section_end_match_top and not self.inside_tool:
                _logger.debug("Found section end marker (top-level)")
                self.inside_section = False
                self.buffer = self.buffer[section_end_match_top.end():]
                continue

            # Check if we're inside a tool call (building name)
            if not self.inside_tool:
                begin_match = re.search(marker_begin, scan_buffer)
                if begin_match:
                    _logger.debug("Found tool_call_begin marker")
                    self.inside_tool = True
                    self.buffer = self.buffer[begin_match.end():]
                    self.name_start = 0  # Name starts now
                    continue
            
            # Check if we're building the function name (between begin and args)
            if self.inside_tool and not self.inside_args:
                scan_buffer = self._mask_code_fences(self.buffer)
                args_match = re.search(marker_args, scan_buffer)
                if args_match:
                    _logger.debug("Found tool_call_argument_begin marker")
                    # Extract function name from buffer (give it the entire buffer so far)
                    name_part = self.buffer[:args_match.start()]
                    if name_part and len(name_part) > 1:
                        # Clean up the name (but keep dots and colons which are valid)
                        clean_name = re.sub(r'[^a-zA-Z0-9_.:]', '', name_part)
                        if clean_name:
                            # Derive a canonical function name that downstream executors expect.
                            # Upstream often formats names like "functions.Read:13" where:
                            # - prefix "functions." is constant
                            # - suffix ":<index>" is a call counter
                            # We keep the full cleaned string as a stable id, but normalize
                            # the function name to just the base (e.g., "Read").
                            base_name = clean_name
                            if base_name.startswith("functions."):
                                base_name = base_name.split(".", 1)[1]
                            if ":" in base_name:
                                base_name = base_name.split(":", 1)[0]

                            # Store both: the id (full) and the normalized function name (provider-specific)
                            # Do NOT remap provider names like TodoWrite; keep as-is
                            self.tool_name = base_name or clean_name
                            self.tool_id = clean_name
                            self.inside_args = True
                            # Move past the args marker
                            self.buffer = self.buffer[args_match.end():]
                            _logger.debug(f"Extracted tool name: {self.tool_name} (id: {self.tool_id})")
                            continue
            
            # Check if we're building the arguments
            if self.inside_args:
                # Look for end markers - both tool_call_end and section_end can end a tool call
                # But ignore markers that appear inside JSON string literals within the arguments.
                scan_buffer = self._mask_code_fences(self.buffer)

                def _find_token_outside_strings(text: str, token: str):
                    """Return index of token not inside a JSON string literal, or -1."""
                    start = 0
                    while True:
                        idx = text.find(token, start)
                        if idx == -1:
                            return -1
                        # Determine if inside string up to idx, using original buffer to keep indices
                        in_string = False
                        escaped = False
                        for ch in self.buffer[:idx]:
                            if escaped:
                                escaped = False
                                continue
                            if ch == '\\':
                                escaped = True
                                continue
                            if ch == '"':
                                in_string = not in_string
                        if not in_string:
                            return idx
                        start = idx + 1

                lit_tool_end = "<|tool_call_end|>"
                lit_section_end = "<|tool_calls_section_end|>"
                tool_end_idx = _find_token_outside_strings(scan_buffer, lit_tool_end)
                section_end_idx = _find_token_outside_strings(scan_buffer, lit_section_end)
                
                # Check if we have either end marker
                has_end_marker = (tool_end_idx != -1) or (section_end_idx != -1)
                
                if has_end_marker:
                    # Use whichever marker comes first
                    if tool_end_idx != -1 and section_end_idx != -1:
                        end_pos = min(tool_end_idx, section_end_idx)
                    else:
                        end_pos = tool_end_idx if tool_end_idx != -1 else section_end_idx
                    
                    _logger.debug("Found end marker, extracting tool call")
                    # We found the end, extract arguments
                    args_part = self.buffer[:end_pos]
                    self.args_buffer += args_part  # Append to existing buffer
                    raw_str = self.args_buffer.strip()

                    import json
                    from ..utils.json_utils import JSONProcessor
                    json_str = raw_str
                    try:
                        # Try direct parse; if it fails, attempt fence stripping and best-object extraction
                        try:
                            json.loads(json_str)
                        except json.JSONDecodeError:
                            if json_str.startswith("```"):
                                inner = json_str.strip("`")
                                if inner.lower().startswith("json"):
                                    inner = inner[4:].strip()
                                json_str = inner
                            extracted = JSONProcessor.extract_best_json(json_str)
                            if extracted:
                                json_str = extracted
                            json.loads(json_str)

                        # Create tool call
                        tool_calls.append({
                            "id": getattr(self, "tool_id", None) or self.tool_name or "call",
                            "index": self.current_index,
                            "type": "function",
                            "function": {
                                "name": self.tool_name or "unknown",
                                "arguments": json_str
                            }
                        })
                        _logger.debug(f"Extracted complete tool call: {self.tool_name}")

                        # Reset for next tool; skip past the consumed end marker token
                        # Determine which marker we consumed at end_pos
                        if tool_end_idx != -1 and tool_end_idx == end_pos:
                            consumed = len(lit_tool_end)
                        elif section_end_idx != -1 and section_end_idx == end_pos:
                            consumed = len(lit_section_end)
                        else:
                            consumed = 0
                        self.buffer = self.buffer[end_pos + consumed:]
                        self.inside_tool = False
                        self.tool_name = None
                        self.tool_id = None
                        self.inside_args = False
                        self.args_buffer = ""
                        self.current_index += 1
                        continue
                    except json.JSONDecodeError as e:
                        _logger.debug("Incomplete JSON in args", error=str(e), buffer=self.args_buffer[:100])
                        # Not complete yet, keep accumulating - clear processed part, keep rest
                        self.args_buffer += args_part
                        self.buffer = self.buffer[end_match.start():]  # Keep end marker for next call
                        return tool_calls  # Return what we have so far, keep state for next call
                else:
                    # No end marker yet, accumulate entire buffer into args_buffer
                    self.args_buffer += self.buffer
                    self.buffer = ""  # Clear buffer since we consumed it all
                    return tool_calls  # Return what we have so far, keep state for next call
            
            # If we get here, no more processing possible right now
            break

        return tool_calls

    def flush(self) -> List[Dict[str, Any]]:
        """Attempt to emit a pending tool call when stream ends without end markers."""
        tool_calls: List[Dict[str, Any]] = []
        if self.inside_args and (self.args_buffer or "").strip():
            raw_str = self.args_buffer.strip()
            import json
            from ..utils.json_utils import JSONProcessor
            json_str = raw_str
            try:
                json.loads(json_str)
            except json.JSONDecodeError:
                if json_str.startswith("```"):
                    inner = json_str.strip("`")
                    if inner.lower().startswith("json"):
                        inner = inner[4:].strip()
                    json_str = inner
                extracted = JSONProcessor.extract_best_json(json_str)
                if extracted:
                    json_str = extracted
                json.loads(json_str)

            tool_calls.append({
                "id": getattr(self, "tool_id", None) or self.tool_name or "call",
                "index": self.current_index,
                "type": "function",
                "function": {
                    "name": self.tool_name or "unknown",
                    "arguments": json_str
                }
            })
            # reset after flushing one
            self.inside_args = False
            self.inside_tool = False
            self.tool_name = None
            self.tool_id = None
            self.args_buffer = ""
            self.current_index += 1
        return tool_calls
