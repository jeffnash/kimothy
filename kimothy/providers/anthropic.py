"""Anthropic provider - handles Claude models."""

import logging
from typing import Dict, Any, Optional
from .base import BaseProvider
from ..config import Settings

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude endpoints."""
    
    PROVIDER_NAME = "anthropic"
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        logger.info(f"Initialized AnthropicProvider for {settings.upstream_base_url}")
    
    def prepare_upstream_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform request for Anthropic API."""
        # Anthropic uses different parameter structure
        # This is a placeholder - real implementation would map OpenAI format to Anthropic
        request.pop("reasoning_effort", None)
        return request
    
    def handle_upstream_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Anthropic response to OpenAI format."""
        # This is a placeholder - real implementation would map Anthropic format to OpenAI
        return response
    
    def supports_reasoning(self) -> bool:
        """Check if provider supports reasoning content."""
        # Claude has reasoning in some models
        return True
    
    def get_default_model(self) -> Optional[str]:
        """Get default model for this provider."""
        return self.settings.default_model or "claude-3-5-sonnet-latest"
