"""Robust tool call processing utilities."""

import json
import time
import sys
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from ..models import ToolCall
from ..utils.json_utils import ToolCallJSONParser, JSONProcessor


# Type alias for reasoning events
ChunkedReasoningEvent = Dict[str, Any]


@dataclass
class ToolCallState:
    """State for tracking tool calls during streaming."""
    
    id: Optional[str] = None
    name: Optional[str] = None
    type: str = "function"
    args_chunks: List[str] = field(default_factory=list)
    args_accumulated: str = ""
    emitted_len: int = 0
    complete: bool = False
    started_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    
    @property
    def args_text(self) -> str:
        """Get complete arguments text."""
        return self.args_accumulated
    
    def append_args(self, text: str) -> None:
        """Append arguments chunk."""
        if text:
            self.args_chunks.append(text)
            self.args_accumulated += text
            self.last_updated_at = time.time()
    
    def set_name_if_empty(self, name: Optional[str]) -> None:
        """Set name only if currently empty."""
        if name and not self.name:
            self.name = name
    
    def refresh_args_full(self, new_full: str) -> None:
        """Authoritative refresh of full arguments (when upstream sends complete JSON)."""
        self.args_accumulated = new_full or ""
        self.args_chunks = [new_full] if new_full else []
    
    def emit_suffix_if_parseable(self, strict_validation: bool = True) -> Optional[str]:
        """
        Emit only NEW suffix when accumulated args become valid JSON.
        
        This is the KEY method for proper incremental streaming:
        - Returns None if JSON is incomplete (don't emit yet)
        - Returns new suffix only when valid JSON is available
        - Tracks emission position to avoid duplicates
        """
        if not self.args_accumulated:
            self.complete = True
            return None
        
        # Try to get valid JSON from accumulated text
        valid_json = None
        
        # 1. Try direct parsing first
        if JSONProcessor.is_valid_json_object(self.args_accumulated):
            valid_json = self.args_accumulated
        
        # 2. If strict validation is off and we have content, let incomplete JSON through
        if not valid_json and not strict_validation and self.args_accumulated.strip():
            # For non-strict mode, emit if we have reasonable content
            # This is controlled by config.validate_json_before_emit
            if self.args_accumulated.strip().startswith('{'):
                valid_json = self.args_accumulated
        
        # 3. Try repair strategies if parsing failed
        if not valid_json:
            # Try to extract best valid JSON object from the text
            # This handles cases where accumulated text has extra characters
            valid_json = JSONProcessor.extract_best_json(self.args_accumulated)
        
        # If we still don't have valid JSON, don't emit yet
        if not valid_json:
            self.complete = False
            return None
        
        # We have valid JSON! Now emit only the NEW part
        if self.emitted_len >= len(valid_json):
            # Already emitted this or more
            return None
        
        # Emit only the suffix since last emission
        suffix = valid_json[self.emitted_len:]
        self.emitted_len = len(valid_json)
        self.complete = True
        
        # Update args to be the valid portion
        self.args_accumulated = valid_json
        
        return suffix if suffix.strip() else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI tool call format."""
        args_text = self.args_text
        
        # Check if we have a repairable JSON using the parser's parse method
        # which returns None for invalid JSON
        has_valid_json = bool(self.json_parser.parse(args_text))
        
        # Only emit if we have valid JSON and a name
        if has_valid_json and self.name:
            return {
                "id": self.id or f"call_{int(self.started_at * 1000)}",
                "type": self.type or "function",
                "function": {
                    "name": self.name or "auto_tool",
                    "arguments": args_text
                }
            }
        else:
            # For OpenAI compatibility during streaming, we MUST emit valid JSON
            # even if incomplete. But we keep track of completeness.
            args_final = args_text if args_text else "{}"
            name_final = self.name or "auto_tool"
            
            return {
                "id": self.id or f"call_{int(self.started_at * 1000)}",
                "type": self.type or "function",
                "function": {
                    "name": name_final,
                    "arguments": args_final
                }
            }


class ToolCallProcessor:
    """Process tool calls from various provider formats."""
    
    def __init__(
        self,
        attempt_json_repair: bool = True,
        use_external_json_libs: bool = True,
        emit_function_call_legacy: bool = False,
        synthesize_tool_calls: bool = True,
        filter_empty_args: bool = True,
        accumulate_min_chars: int = 5,
        validate_json_before_emit: bool = True
    ):
        self.json_parser = ToolCallJSONParser(
            attempt_repair=attempt_json_repair,
            use_external_libs=use_external_json_libs
        )
        self.emit_function_call_legacy = emit_function_call_legacy
        self.synthesize_tool_calls = synthesize_tool_calls
        self.filter_empty_args = filter_empty_args
        self.accumulate_min_chars = accumulate_min_chars
        self.validate_json_before_emit = validate_json_before_emit
        
        # State tracking
        self.tool_calls: Dict[int, ToolCallState] = {}
        self.last_index: Optional[int] = None
        
        # Bind JSON parser to ToolCallState class for use in emit_suffix_if_parseable
        ToolCallState.json_parser = self.json_parser
        
    def _is_spam_args(self, text: str) -> bool:
        """Detect spam/placeholder arguments that should be filtered."""
        if not isinstance(text, str):
            return True
        
        cleaned = text.strip()
        if not cleaned:
            return True
        
        # Filter common placeholder patterns
        spam_patterns = [
            "",  # Empty string
            "{",  # Lone opening brace
            "}",  # Lone closing brace
            "[]",  # Empty array
            "{}",  # Empty object
            "  ",  # Just whitespace
        ]
        
        if cleaned in spam_patterns:
            return True
        
        # Also filter control codes and obvious garbage
        if all(ord(c) < 32 for c in cleaned if c not in "\n\r\t"):
            return True
        
        return False
    
    def _is_valid_args_for_emission(self, text: str) -> bool:
        """Check if arguments are valid enough to emit."""
        if self._is_spam_args(text):
            return False
        
        if self.validate_json_before_emit:
            # Try to parse as JSON
            try:
                parsed = json.loads(text)
                # If it's a dict/object, it's valid
                return isinstance(parsed, dict)
            except json.JSONDecodeError:
                # If we have enough chars and don't strictly validate, emit anyway
                if len(text.strip()) >= self.accumulate_min_chars:
                    return True
                return False
        
        return len(text.strip()) >= self.accumulate_min_chars
    
    def reset(self) -> None:
        """Reset internal state."""
        self.tool_calls.clear()
        self.last_index = None
    
    def process_tool_calls_delta(
        self,
        tool_calls_delta: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process incremental tool calls delta."""
        if not tool_calls_delta:
            return []
        
        emitted = []
        
        for delta in tool_calls_delta:
            if not isinstance(delta, dict):
                continue
            
            index = delta.get("index", 0)
            self.last_index = index
            
            # Get or create tool call state
            if index not in self.tool_calls:
                self.tool_calls[index] = ToolCallState()
            
            state = self.tool_calls[index]
            
            # Update ID - only if not null (don't overwrite valid ID with null)
            if "id" in delta and delta["id"] is not None:
                state.id = delta["id"]
            
            # Update function details
            function_delta = delta.get("function", {})
            if function_delta:
                if isinstance(function_delta, dict):
                    # Update name - only if not null and not empty
                    if "name" in function_delta and function_delta["name"] is not None:
                        name = function_delta["name"]
                        if name and str(name).strip():
                            state.set_name_if_empty(name)
                    
                    # Update arguments - accumulate all fragments
                    if "arguments" in function_delta:
                        args_text = function_delta["arguments"]
                        if args_text is not None:
                            state.append_args(str(args_text))
            
            # Generate result ONLY when we have a COMPLETE tool call
            # (both name AND valid complete JSON)
            suffix = state.emit_suffix_if_parseable(strict_validation=self.validate_json_before_emit)
            
            # Emit ONLY when we have BOTH name and complete arguments
            # This prevents premature execution with incomplete/empty args
            if suffix is not None and state.name:
                result = {
                    "index": index,
                    "id": state.id,
                    "type": "function",
                    "function": {
                        "name": state.name,
                        "arguments": suffix
                    }
                }
                emitted.append(result)
        
        return emitted
    
    def process_function_call_delta(self, function_call_delta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process legacy function call delta."""
        if not function_call_delta:
            return []
        
        # Convert to tool_calls format
        tool_delta = {
            "index": self.last_index or 0,
            "function": function_call_delta
        }
        
        return self.process_tool_calls_delta([tool_delta])
    
    def process_reasoning_markers(
        self,
        reasoning_content: str
    ) -> Tuple[List[Dict[str, Any]], List[ChunkedReasoningEvent]]:
        """Process Kimi-style reasoning markers into tool calls."""
        if not reasoning_content or not self.synthesize_tool_calls:
            return [], []
        
        from ..providers.reasoning_handler import SimpleReasoningParser
        
        parser = SimpleReasoningParser()
        
        # Extract tool calls from reasoning
        tool_calls = parser.extract_tool_calls(reasoning_content)
        
        if not tool_calls:
            return [], []
        
        # Create ToolCallState entries to track these tool calls
        tool_calls_for_emission = []
        next_index = (self.last_index + 1) if self.last_index is not None else 0
        
        for idx, tc in enumerate(tool_calls):
            actual_index = next_index + idx
            
            # Create state for this tool call
            state = ToolCallState()
            state.id = f"call_{int(time.time() * 1000)}_{actual_index}"
            state.name = tc["function"]["name"]
            state.append_args(tc["function"]["arguments"])
            state.complete = True
            state.last_updated_at = time.time()
            
            # Store in tool_calls dict
            self.tool_calls[actual_index] = state
            self.last_index = actual_index
            
            # Create formatted tool call for emission
            tool_call = {
                "index": actual_index,
                "id": state.id,
                "type": "function",
                "function": {
                    "name": state.name,
                    "arguments": state.args_text
                }
            }
            tool_calls_for_emission.append(tool_call)
        
        return tool_calls_for_emission, []
    
    def finalize_tool_calls(self, validate: bool = True) -> List[Dict[str, Any]]:
        """Finalize all tool calls for output."""
        result = []
        
        for index in sorted(self.tool_calls.keys()):
            state = self.tool_calls[index]
            
            # Only emit tool calls with complete JSON arguments
            args_text = state.args_text
            if not self._is_valid_args_for_emission(args_text):
                continue
            
            tc_dict = state.to_dict()
            
            # Validate and repair arguments
            if validate:
                args = tc_dict["function"]["arguments"]
                repaired = self.json_parser.parse(args)
                if repaired:
                    tc_dict["function"]["arguments"] = repaired
                elif args and hasattr(self.json_parser, "to_raw_fallback"):
                    tc_dict["function"]["arguments"] = self.json_parser.to_raw_fallback(args)
            
            result.append(tc_dict)
        
        return result
    
    def get_incremental_delta(
        self,
        index: int,
        include_complete: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get incremental delta for a tool call."""
        if index not in self.tool_calls:
            return None
        
        state = self.tool_calls[index]
        
        # Calculate what's changed since last emit
        # For simplicity, emit latest args chunk
        delta = {
            "index": index,
            "id": state.id,
            "type": "function",
            "function": {}
        }
        
        if state.name:
            delta["function"]["name"] = state.name
        
        if state.args_chunks:
            delta["function"]["arguments"] = state.args_chunks[-1]
        
        return delta if delta["function"] else None


class ToolCallManager:
    """High-level manager for tool call processing throughout request lifecycle."""
    
    def __init__(
        self,
        settings: Any,
        log_enabled: bool = True
    ):
        self.settings = settings
        self.processor = ToolCallProcessor(
            attempt_json_repair=settings.toolcall_autocomplete_json or settings.json_repair_backup,
            use_external_json_libs=settings.json_repair_backup,
            emit_function_call_legacy=settings.toolcall_emit_function_call_legacy,
            synthesize_tool_calls=True
        )
        self.log_enabled = log_enabled
        
        # Auto-finish tracking
        self.auto_finish_enabled = settings.toolcall_auto_finish
        self.auto_finish_delay_ms = settings.toolcall_auto_finish_delay_ms
        self.finish_grace_ms = settings.toolcall_finish_grace_ms
        
        # State
        self.last_activity = time.time()
        self.pending_finish: Dict[int, float] = {}  # index -> timestamp
    
    def reset(self) -> None:
        """Reset manager state."""
        self.processor.reset()
        self.last_activity = time.time()
        self.pending_finish.clear()
    
    def process_upstream_delta(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single upstream delta."""
        self.last_activity = time.time()
        
        # Handle reasoning_content (Kimi-style)
        if "reasoning_content" in chunk_data:
            reasoning = chunk_data["reasoning_content"]
            tool_calls, markers = self.processor.process_reasoning_markers(reasoning)
            
            if tool_calls:
                return {
                    "tool_calls": tool_calls,
                    "reasoning_content": reasoning
                }
        
        # Handle tool_calls delta
        if "tool_calls" in chunk_data:
            deltas = self.processor.process_tool_calls_delta(chunk_data["tool_calls"])
            return {"tool_calls": deltas}
        
        # Handle function_call delta (legacy)
        if "function_call" in chunk_data:
            deltas = self.processor.process_function_call_delta(chunk_data["function_call"])
            return {"tool_calls": deltas}
        
        return {}
    
    def should_emit_finish(self) -> bool:
        """Check if we should emit tool_calls finish."""
        if not self.auto_finish_enabled:
            return False
        
        now = time.time()
        
        # Check if we've been idle long enough
        idle_time_ms = (now - self.last_activity) * 1000
        if idle_time_ms >= self.auto_finish_delay_ms:
            return True
        
        return False
    
    def generate_finish_chunk(self) -> Optional[Dict[str, Any]]:
        """Generate finish chunk if needed."""
        if not self.processor.tool_calls:
            return None
        
        # Check grace period
        now = time.time()
        
        # Find any tool call that has been idle long enough AND has complete JSON
        for index in list(self.processor.tool_calls.keys()):
            state = self.processor.tool_calls[index]
            
            # Skip if no name (not a complete tool call)
            if not state.name:
                continue
            
            # Check if idle time exceeded
            idle_time_ms = (now - state.last_updated_at) * 1000
            if idle_time_ms < self.finish_grace_ms:
                continue
            
            # Only emit when we have complete JSON
            args_text = state.args_text
            if not self.processor.json_parser.is_valid_json_object(args_text):
                continue
            
            # Valid complete tool call ready to finalize
            return {
                "finish_reason": "tool_calls",
                "tool_calls": self.processor.finalize_tool_calls()
            }
        
        # If no complete tool calls, don't emit finish yet
        return None
    
    def finalize(self) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Finalize tool calls at end of stream.
        
        Returns: (tool_calls, should_emit_finish)
        """
        tool_calls = self.processor.finalize_tool_calls()
        has_tool_calls = bool(tool_calls)
        
        should_finish = (
            has_tool_calls and
            self.settings.toolcall_finish_on_tool_end and
            not self.settings.toolcall_emit_incremental
        )
        
        return tool_calls, should_finish
