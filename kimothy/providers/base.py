from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, Optional, Union
from ..models import ChatCompletionRequest, CompletionRequest, ChatCompletionResponse, ChatCompletionChunk


class ProviderRequest:
    """Container for provider request data."""
    
    def __init__(self, data: Dict[str, Any], headers: Dict[str, str]) -> None:
        self.data = data
        self.headers = headers
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from request data."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in request data."""
        self.data[key] = value


class ProviderResponse:
    """Container for provider response data."""
    
    def __init__(self, status_code: int, headers: Dict[str, str], body: Any) -> None:
        self.status_code = status_code
        self.headers = headers
        self.body = body
        
    def get_body(self) -> Any:
        """Get response body."""
        return self.body


class BaseProvider(ABC):
    """Abstract base class for all providers."""
    
    def __init__(self, name: str, settings: Any) -> None:
        self.name = name
        self.settings = settings
        
    @abstractmethod
    async def chat_completions(
        self, 
        request: ProviderRequest
    ) -> AsyncIterator[Union[ProviderResponse, ChatCompletionResponse, ChatCompletionChunk]]:
        """
        Handle chat completions requests.
        
        Returns either:
        - ProviderResponse for non-streaming
        - ChatCompletionResponse for parsed non-streaming
        - ChatCompletionChunk for streaming
        """
        pass
    
    @abstractmethod
    async def completions(
        self, 
        request: ProviderRequest
    ) -> AsyncIterator[Union[ProviderResponse, ChatCompletionResponse, ChatCompletionChunk]]:
        """
        Handle text completions requests.
        
        Returns either:
        - ProviderResponse for non-streaming
        - ChatCompletionResponse for parsed non-streaming
        - ChatCompletionChunk for streaming
        """
        pass
    
    def transform_request(self, request: ProviderRequest, endpoint: str) -> ProviderRequest:
        """
        Transform incoming request to provider-specific format.
        Called before sending to provider.
        """
        return request
    
    def transform_response(self, response: ProviderResponse, endpoint: str) -> ProviderResponse:
        """
        Transform provider response to OpenAI format.
        Called after receiving from provider.
        """
        return response
    
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        return True
    
    def get_model_name(self, request: ProviderRequest) -> str:
        """Extract model name from request."""
        model = request.get("model") or self.settings.default_model
        if not model:
            raise ValueError("Model not specified in request and no default model configured")
        return model
    
    def build_headers(self, api_key: Optional[str] = None) -> Dict[str, str]:
        """Build headers for provider request."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        key = api_key or self.settings.upstream_api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
            
        return headers
    
    def detect_stream_mode(self, request: ProviderRequest) -> bool:
        """Detect if request should be streamed."""
        return bool(request.get("stream", False))
        
    def normalize_chunk(self, chunk: Dict[str, Any]) -> Optional[ChatCompletionChunk]:
        """
        Normalize a single streaming chunk to OpenAI format.
        Override for provider-specific normalization.
        """
        try:
            return ChatCompletionChunk(**chunk)
        except Exception:
            return None
