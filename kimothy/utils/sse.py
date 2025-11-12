"""Server-Sent Events (SSE) parsing utilities."""

import json
from typing import Iterator, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SSEEvent:
    """Server-Sent Event representation."""
    
    data: Optional[str] = None
    event: Optional[str] = None
    id: Optional[str] = None
    retry: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        if self.data is not None:
            result["data"] = self.data
        if self.event is not None:
            result["event"] = self.event
        if self.id is not None:
            result["id"] = self.id
        if self.retry is not None:
            result["retry"] = self.retry
        return result


class SSEParser:
    """Parse Server-Sent Events from text stream with JSON fragmentation support."""
    
    def __init__(self, strict: bool = False) -> None:
        self.buffer = ""
        self.strict = strict
        self._json_accumulator = ""  # Accumulate JSON fragments
        self._last_event_id = None
        self._last_event_type = None
    
    def feed(self, chunk: str) -> Iterator[SSEEvent]:
        """Feed chunk of data and yield complete SSE events.
        
        Handles JSON that is split across multiple data lines (e.g., chutes streaming).
        """
        # Normalize CRLF to LF to simplify parsing across providers
        self.buffer += chunk.replace("\r\n", "\n")
        
        # Process all complete events (separated by double newline)
        while "\n\n" in self.buffer:
            event_text, self.buffer = self.buffer.split("\n\n", 1)
            event = self._parse_event_fragment(event_text)
            if event:
                yield event
    
    def _parse_event_fragment(self, text: str) -> Optional[SSEEvent]:
        """Parse SSE event fragment and accumulate JSON."""
        if not text:
            return None
        
        data_lines = []
        event_type = None
        event_id = None
        
        # Parse all fields in this fragment
        for line in text.split("\n"):
            if not line:
                continue
            
            if ": " in line:
                field, value = line.split(": ", 1)
            elif line.startswith(":"):
                continue  # Comment line
            elif ":" in line:
                field, value = line.split(":", 1)
                value = value.lstrip()
            else:
                field, value = line, ""
            
            field = field.lower()
            
            if field == "data":
                data_lines.append(value)
            elif field == "event":
                event_type = value
                self._last_event_type = value
            elif field == "id":
                event_id = value
                self._last_event_id = value
            elif field == "retry":
                try:
                    int(value)
                except ValueError:
                    pass
        
        # Accumulate all data lines (handles JSON split across lines)
        if data_lines:
            # Join all data lines
            for line in data_lines:
                # For JSON fragments, accumulate. Per SSE spec, multiple
                # data lines are joined with a newline between them.
                stripped_line = line.strip()
                if stripped_line and stripped_line != "[DONE]":
                    if self._json_accumulator:
                        self._json_accumulator += "\n" + line
                    else:
                        self._json_accumulator += line
                elif stripped_line == "[DONE]":
                    # Special [DONE] marker
                    self._json_accumulator = "[DONE]"
                else:
                    # Empty line - might be end of fragment
                    pass
            
            # Check if we have a complete JSON object or [DONE]
            accumulated = self._json_accumulator.strip()
            
            # Determine if JSON is complete
            is_json_complete = False
            if accumulated == "[DONE]":
                is_json_complete = True
            elif accumulated.startswith('{') and accumulated.endswith('}'):
                # Check if it's valid JSON by trying to parse
                try:
                    json.loads(accumulated)
                    is_json_complete = True
                except json.JSONDecodeError:
                    # Might still be accumulating
                    pass
            elif accumulated.startswith('[') and accumulated.endswith(']'):
                is_json_complete = True
            
            # If complete, yield the event
            if is_json_complete and accumulated:
                event = SSEEvent(
                    data=self._json_accumulator,
                    event=event_type or self._last_event_type,
                    id=event_id or self._last_event_id
                )
                # Reset accumulator
                self._json_accumulator = ""
                return event
        
        return None

    def flush(self) -> Iterator[SSEEvent]:
        """Flush any remaining buffered data as a final SSE event if possible.

        Some providers may end the stream without a trailing blank line (\n\n).
        This attempts to parse whatever remains in the buffer as a single event.
        """
        if not self.buffer:
            return iter(())
        # Try to parse remaining fragment
        event = self._parse_event_fragment(self.buffer)
        self.buffer = ""
        if event:
            yield event
        return iter(())


class SSEGenerator:
    """Generate Server-Sent Events."""
    
    @staticmethod
    def generate(event: Optional[SSEEvent] = None, **kwargs: Any) -> str:
        """Generate SSE text from event or kwargs."""
        if event is None:
            event = SSEEvent(**kwargs)
        
        lines = []
        
        if event.id is not None:
            lines.append(f"id: {event.id}")
        
        if event.event is not None:
            lines.append(f"event: {event.event}")
        
        if event.retry is not None:
            lines.append(f"retry: {event.retry}")
        
        if event.data is not None:
            for line in event.data.split("\n"):
                lines.append(f"data: {line}")
        
        return "\n".join(lines) + "\n\n"
    
    @staticmethod
    def generate_json(data: Dict[str, Any]) -> str:
        """Generate SSE event with JSON data."""
        return SSEGenerator.generate(data=json.dumps(data))
    
    @staticmethod
    def generate_done() -> str:
        """Generate done marker event."""
        return SSEGenerator.generate(data="[DONE]")
