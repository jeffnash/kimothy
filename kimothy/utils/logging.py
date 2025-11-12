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
