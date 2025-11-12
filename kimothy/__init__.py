"""Kimothy: OpenAI-Compatible Streaming Proxy for Kimi-like Providers."""

__version__ = "2.0.0"
__author__ = "Kimothy Team"

from .config import get_settings, Settings
from .main import app

__all__ = ["get_settings", "Settings", "app"]
