"""Nahcrof provider - handles Factory Droid client with specific tool call patterns."""

import logging
from typing import Dict, Any, Optional
from .base import BaseProvider
from ..config import Settings

logger = logging.getLogger(__name__)


class NahcrofProvider(BaseProvider):
    """Provider for Nahcrof/Factory Droid endpoints."""
    
    PROVIDER_NAME = "nahcrof"
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        logger.info(f"Initialized NahcrofProvider for {settings.upstream_base_url}")
    
    def prepare_upstream_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform request for Nahcrof API."""
        # Nagcrof doesn't accept reasoning_effort parameter
        request.pop("reasoning_effort", None)
        
        # Handle model name mapping if needed
        model = request.get("model")
        if model == "kimi-k2-thinking":
            # Convert to full model path if needed
            request["model"] = "moonshotai/Kimi-K2-Thinking"
        
        return request
    
    def handle_upstream_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Nahcrof response for OpenAI compatibility."""
        # Nahcrof sends tool calls with empty arguments, let the tool processor handle it
        return response
    
    def supports_reasoning(self) -> bool:
        """Check if provider supports reasoning content."""
        return True
    
    def get_default_model(self) -> Optional[str]:
        """Get default model for this provider."""
        return "moonshotai/Kimi-K2-Thinking"
