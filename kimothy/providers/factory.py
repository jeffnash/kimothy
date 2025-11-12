"""Provider factory for creating provider instances."""

from typing import Dict, Type
from .base import BaseProvider
from ..config import Settings


class ProviderFactory:
    """Factory for creating provider instances based on settings."""
    
    _providers: Dict[str, Type[BaseProvider]] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[BaseProvider]) -> None:
        """Register a provider class."""
        cls._providers[name] = provider_class
    
    @classmethod
    def create(cls, settings: Settings) -> BaseProvider:
        """Create a provider instance based on settings."""
        # Determine provider based on base URL
        base_url = settings.upstream_base_url.lower()
        
        # Auto-detect provider from URL patterns
        if "nahcrof" in base_url:
            from .nahcrof import NahcrofProvider
            provider_class = NahcrofProvider
        elif "moonshot" in base_url or "kimi" in base_url:
            from .kimi import KimiProvider
            provider_class = KimiProvider
        elif "chutes" in base_url:
            from .chutes import ChutesProvider
            provider_class = ChutesProvider
        elif "openai" in base_url:
            from .openai import OpenAIProvider
            provider_class = OpenAIProvider
        elif "anthropic" in base_url or "claude" in base_url:
            from .anthropic import AnthropicProvider
            provider_class = AnthropicProvider
        elif "google" in base_url or "vertex" in base_url:
            from .google import GoogleProvider
            provider_class = GoogleProvider
        else:
            # Default to generic provider for unknown endpoints
            from .generic import GenericProvider
            provider_class = GenericProvider
        
        return provider_class(settings)


def create_provider(settings: Settings) -> BaseProvider:
    """Convenience function to create a provider."""
    return ProviderFactory.create(settings)
