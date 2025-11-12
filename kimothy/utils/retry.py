"""Error types for the proxy."""


class ProxyError(Exception):
    """Base error for proxy operations."""
    
    def __init__(self, message: str, code: str = "PROXY_ERROR", status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class ProviderError(ProxyError):
    """Error from upstream provider."""
    
    def __init__(self, message: str, provider: str, status_code: int = 502) -> None:
        super().__init__(message, "PROVIDER_ERROR", status_code)
        self.provider = provider


class ValidationError(ProxyError):
    """Validation error for request/response data."""
    
    def __init__(self, message: str, field: str = None) -> None:
        super().__init__(message, "VALIDATION_ERROR", 400)
        self.field = field


class AuthenticationError(ProxyError):
    """Authentication error."""
    
    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message, "AUTH_ERROR", 401)


class RateLimitError(ProxyError):
    """Rate limit error."""
    
    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message, "RATE_LIMIT", 429)


class NotFoundError(ProxyError):
    """Resource not found error."""
    
    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, "NOT_FOUND", 404)


class ParseError(ProxyError):
    """Parse error."""
    
    def __init__(self, message: str = "Failed to parse response") -> None:
        super().__init__(message, "PARSE_ERROR", 400)
"""Retry utilities with exponential backoff."""

import asyncio
import random
import time
from typing import Callable, Optional, Type, Tuple, Any
from dataclasses import dataclass



@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    jitter_range: Tuple[float, float] = (0.8, 1.2)
    retry_on: Tuple[Type[Exception], ...] = (RateLimitError, ProviderError)
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for attempt."""
        delay = min(
            self.initial_delay * (self.backoff_factor ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            jitter_range = self.jitter_range
            delay *= random.uniform(*jitter_range)
        
        return delay


class RetryHandler:
    """Handles retry logic with exponential backoff."""
    
    def __init__(self, config: Optional[RetryConfig] = None) -> None:
        self.config = config or RetryConfig()
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        last_exception: Optional[Exception] = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                # Check if we should retry
                if not self._should_retry(e):
                    raise
                
                # Don't delay after last attempt
                if attempt >= self.config.max_attempts - 1:
                    break
                
                # Calculate delay
                delay = self.config.get_delay(attempt)
                
                # Log retry (can be passed via context)
                logger = kwargs.get("_logger")
                if logger:
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}",
                        retry_attempt=attempt + 1,
                        retry_delay=delay,
                        error_type=type(e).__name__
                    )
                
                await asyncio.sleep(delay)
        
        # All attempts failed
        if last_exception:
            raise last_exception
        raise Exception("Max retries exceeded")
    
    def _should_retry(self, exc: Exception) -> bool:
        """Check if exception should trigger retry."""
        for retry_exc in self.config.retry_on:
            if isinstance(exc, retry_exc):
                return True
        return False
