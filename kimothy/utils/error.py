"""Exception classes for Kimothy proxy."""


class ProxyError(Exception):
    """Base exception for proxy errors."""
    
    def __init__(self, message: str, status_code: int = 500, code: str = "proxy_error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class ValidationError(ProxyError):
    """Request validation error."""
    
    def __init__(self, message: str, field: str = None):
        super().__init__(message, status_code=400, code="validation_error")
        self.field = field


class ProviderError(ProxyError):
    """Upstream provider error."""
    
    def __init__(self, message: str, status_code: int = 502, provider: str = None):
        super().__init__(message, status_code=status_code, code="provider_error")
        self.provider = provider


class RateLimitError(ProviderError):
    """Rate limit error from provider."""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)


class AuthenticationError(ProxyError):
    """Authentication error."""
    
    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(message, status_code=401, code="invalid_api_key")
