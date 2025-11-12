"""Structured logging utilities."""

import json
import sys
import time
from typing import Any, Dict, Optional, IO
from datetime import datetime
import inspect


class Logger:
    """Structured logger with context and levels."""
    
    LEVELS = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50,
    }
    
    def __init__(self, name: str, level: str = "INFO", output: IO = sys.stderr):
        self.name = name
        self.level = self.LEVELS.get(level.upper(), 20)
        self.output = output
        self._context: Dict[str, Any] = {}
    
    def set_level(self, level: str) -> None:
        """Set log level."""
        self.level = self.LEVELS.get(level.upper(), 20)
    
    def add_context(self, **kwargs: Any) -> None:
        """Add context to all subsequent logs."""
        self._context.update(kwargs)
    
    def remove_context(self, *keys: str) -> None:
        """Remove context keys."""
        for key in keys:
            self._context.pop(key, None)
    
    def clear_context(self) -> None:
        """Clear all context."""
        self._context.clear()
    
    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        """Internal logging method."""
        if self.LEVELS[level] < self.level:
            return
            
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            **self._context,
            **kwargs,
        }
        
        try:
            line = json.dumps(record, default=str)
            print(line, file=self.output, flush=True)
        except Exception:
            # Fallback if JSON serialization fails
            print(f"[{record['timestamp']}] {level}: {message}", file=self.output, flush=True)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._log("CRITICAL", message, **kwargs)
    
    def exception(self, message: str, exc: Exception, **kwargs: Any) -> None:
        """Log exception with stack trace."""
        import traceback
        tb = traceback.format_exc()
        self._log("ERROR", message, exception=str(exc), traceback=tb, **kwargs)


class Spinner:
    """Terminal spinner for showing progress."""
    
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    def __init__(self, text: str = "Processing", interval: float = 0.15):
        self.text = text
        self.interval = interval
        self._active = False
        self._frame_index = 0
    
    def start(self) -> None:
        """Start spinner."""
        self._active = True
        self._frame_index = 0
        self._update()
    
    def stop(self) -> None:
        """Stop spinner."""
        self._active = False
        print("\r" + " " * 80 + "\r", end="", flush=True)
    
    def _update(self) -> None:
        """Update spinner frame."""
        if not self._active:
            return
        
        import threading
        
        frame = self.FRAMES[self._frame_index % len(self.FRAMES)]
        print(f"\r{frame} {self.text}", end="", flush=True)
        
        self._frame_index += 1
        
        # Schedule next update
        timer = threading.Timer(self.interval, self._update)
        timer.daemon = True
        timer.start()


# Global logger instances
_loggers: Dict[str, Logger] = {}


def get_logger(name: str, level: Optional[str] = None) -> Logger:
    """Get or create a logger instance."""
    if name not in _loggers:
        _loggers[name] = Logger(name, level or "INFO")
    elif level:
        _loggers[name].set_level(level)
    return _loggers[name]
"""JSON processing utilities for tool calls."""

import json
import re
from typing import Any, Dict, List, Optional, Union


class JSONProcessor:
    """JSON processing utilities."""
    
    @staticmethod
    def parse_safe(text: str) -> Optional[Any]:
        """Safely parse JSON string."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    
    @staticmethod
    def parse_object_safe(text: str) -> Optional[Dict[str, Any]]:
        """Safely parse JSON string to dict."""
        result = JSONProcessor.parse_safe(text)
        if isinstance(result, dict):
            return result
        return None
    
    @staticmethod
    def is_valid_json(text: str) -> bool:
        """Check if string is valid JSON."""
        return JSONProcessor.parse_safe(text) is not None
    
    @staticmethod
    def is_valid_json_object(text: str) -> bool:
        """Check if string is valid JSON object."""
        return JSONProcessor.parse_object_safe(text) is not None
    
    @staticmethod
    def autocomplete_json(text: str) -> Optional[str]:
        """Auto-complete JSON by adding missing braces/brackets."""
        if not isinstance(text, str) or not text.strip().startswith("{"):
            return None
        
        # Count braces
        open_braces = text.count("{")
        close_braces = text.count("}")
        open_brackets = text.count("[")
        close_brackets = text.count("]")
        
        missing_braces = max(0, open_braces - close_braces)
        missing_brackets = max(0, open_brackets - close_brackets)
        
        if missing_braces == 0 and missing_brackets == 0:
            return None
        
        # Add missing closing characters
        completed = text + ("}" * missing_braces) + ("]" * missing_brackets)
        
        # Validate the result
        if JSONProcessor.is_valid_json_object(completed):
            return completed
        return None
    
    @staticmethod
    def repair_json_with_library(text: str) -> Optional[str]:
        """Try to repair JSON using external libraries."""
        if not text:
            return None
        
        # Try json_repair if available
        try:
            import importlib
            json_repair = importlib.import_module("json_repair")
            if hasattr(json_repair, "repair_json"):
                result = json_repair.repair_json(text)
                if result and JSONProcessor.is_valid_json_object(result):
                    return result
        except Exception:
            pass
        
        # Try jsonrepair if available
        try:
            json_repair = importlib.import_module("jsonrepair")
            if hasattr(json_repair, "repair_json"):
                result = json_repair.repair_json(text)
                if result:
                    result_str = json.dumps(result) if not isinstance(result, str) else result
                    if JSONProcessor.is_valid_json_object(result_str):
                        return result_str
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def extract_best_json(text: str, key_patterns: List[str] = None) -> Optional[str]:
        """Extract the best JSON object from text."""
        if not text or "{" not in text or "}" not in text:
            return None
        
        if key_patterns is None:
            key_patterns = ["command", "name", "explanation"]
        
        n = len(text)
        best_candidate: Optional[str] = None
        best_score = -1
        max_checks = 1000
        checks = 0
        
        for i in range(n):
            if text[i] != "{":
                continue
            
            # Try ending positions
            for j in range(i + 1, n + 1):
                if checks >= max_checks:
                    break
                if j < n and text[j] != "}":
                    continue
                
                checks += 1
                candidate = text[i:j]
                
                # Try to parse
                try:
                    parsed = json.loads(candidate)
                    if not isinstance(parsed, dict):
                        continue
                    
                    # Score based on key patterns
                    score = 0
                    keys = set(parsed.keys())
                    
                    for idx, pattern in enumerate(key_patterns):
                        if pattern in keys:
                            score += (len(key_patterns) - idx) * 3
                    
                    # Prefer longer candidates at same score
                    if score > best_score or (score == best_score and len(candidate) > len(best_candidate or "")):
                        best_score = score
                        best_candidate = candidate
                
                except json.JSONDecodeError:
                    continue
        
        return best_candidate


class ToolCallJSONParser:
    """Parse and repair tool call JSON arguments."""
    
    def __init__(self, attempt_repair: bool = True, use_external_libs: bool = True):
        self.attempt_repair = attempt_repair
        self.use_external_libs = use_external_libs
    
    def parse(self, text: str) -> Optional[str]:
        """Parse or repair tool call arguments."""
        if not text:
            return None
        
        # Already valid
        if JSONProcessor.is_valid_json_object(text):
            return text
        
        if not self.attempt_repair:
            return None
        
        # Try to autocomplete
        fixed = JSONProcessor.autocomplete_json(text)
        if fixed:
            return fixed
        
        # Extract best JSON object
        extracted = JSONProcessor.extract_best_json(text)
        if extracted:
            return extracted
        
        # Try external libraries
        if self.use_external_libs:
            repaired = JSONProcessor.repair_json_with_library(text)
            if repaired:
                return repaired
        
        return None
    
    def to_raw_fallback(self, text: str) -> str:
        """Create raw fallback JSON object."""
        return json.dumps({"_raw": text})
    
    def parse_tool_calls_list(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse and repair all tool calls in list."""
        if not tool_calls:
            return []
        
        result = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            
            function_spec = tc.get("function", {})
            if not isinstance(function_spec, dict):
                if hasattr(function_spec, "dict"):
                    function_spec = function_spec.dict()
                else:
                    continue
            
            args = function_spec.get("arguments", "")
            if not args:
                result.append(tc)
                continue
            
            # Try to parse args
            parsed = self.parse(args)
            if parsed:
                function_spec["arguments"] = parsed
            elif hasattr(self, "to_raw_fallback") and getattr(self, "emit_raw_fallback", True):
                function_spec["arguments"] = self.to_raw_fallback(args)
            
            tc["function"] = function_spec
            result.append(tc)
        
        return result
