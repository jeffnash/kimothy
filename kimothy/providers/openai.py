"""OpenAI provider - standard OpenAI-compatible endpoints."""

import logging
from typing import Dict, Any, Optional
from .base import BaseProvider
from ..config import Settings

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """Provider for standard OpenAI API endpoints."""
    
    PROVIDER_NAME = "openai"
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        logger.info(f"Initialized OpenAIProvider for {settings.upstream_base_url}")
    
    def prepare_upstream_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """No special transformation needed for OpenAI."""
        # Remove reasoning_effort if present (not supported by standard OpenAI yet)
        request.pop("reasoning_effort", None)
        return request
    
    def handle_upstream_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """No special handling needed for OpenAI."""
        return response
    
    def supports_reasoning(self) -> bool:
        """Check if provider supports reasoning content."""
        return False
    
    def get_default_model(self) -> Optional[str]:
        """Get default model for this provider."""
        return self.settings.default_model or "gpt-4o"
