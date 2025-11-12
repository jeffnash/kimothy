"""Configuration management for Kimothy proxy."""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional, Any
from enum import Enum


class ModelPlacement(str, Enum):
    """Where to place the model in upstream requests."""
    BODY = "body"
    PATH = "path"


class UpstreamEndpoint(str, Enum):
    """Upstream endpoint options."""
    CHAT_COMPLETIONS = "chat/completions"
    COMPLETIONS = "completions"
    AUTO = "auto"


class Settings(BaseSettings):
    """Application settings with validation and defaults."""
    
    # Core settings
    upstream_base_url: Optional[str] = Field(None, env="UPSTREAM_BASE_URL", description="Provider base URL")
    upstream_api_key: Optional[str] = Field(None, env="UPSTREAM_API_KEY", description="Provider API key")
    default_model: Optional[str] = Field(None, env="DEFAULT_MODEL", description="Default model when not specified")
    
    # Request shaping
    ensure_stream: bool = Field(True, env="ENSURE_STREAM", description="Force stream: true upstream")
    url_model_placement: ModelPlacement = Field(ModelPlacement.BODY, env="URL_MODEL_PLACEMENT")
    upstream_endpoint: UpstreamEndpoint = Field(UpstreamEndpoint.AUTO, env="UPSTREAM_ENDPOINT")
    override_temperature: Optional[float] = Field(None, env="OVERRIDE_TEMPERATURE")
    override_max_tokens: Optional[int] = Field(None, env="OVERRIDE_MAX_TOKENS")
    
    # Tool call handling
    toolcall_fallback_name: str = Field("auto_tool", env="TOOLCALL_FALLBACK_NAME")
    toolcall_emit_function_call_legacy: bool = Field(False, env="EMIT_FUNCTION_CALL_LEGACY")
    toolcall_emit_incremental: bool = Field(False, env="TOOLCALL_EMIT_INCREMENTAL")
    toolcall_emit_raw_fallback: bool = Field(True, env="TOOLCALL_EMIT_RAW_FALLBACK")
    toolcall_finish_on_tool_end: bool = Field(True, env="FINISH_ON_TOOL_END")
    toolcall_auto_finish: bool = Field(True, env="AUTO_FINISH_TOOLCALLS")
    toolcall_auto_finish_delay_ms: int = Field(800, env="AUTO_FINISH_DELAY_MS")
    toolcall_finish_grace_ms: int = Field(600, env="TOOLCALL_FINISH_GRACE_MS")
    toolcall_infer_name: bool = Field(True, env="TOOLCALL_INFER_NAME")
    toolcall_warn_on_empty_name: bool = Field(True, env="TOOLCALL_WARN_ON_EMPTY_NAME")
    toolcall_strict_json: bool = Field(True, env="STRICT_TOOLCALL_JSON")
    toolcall_autocomplete_json: bool = Field(True, env="TOOLCALL_AUTOCOMPLETE_JSON")
    toolcall_min_chars: int = Field(4, env="TOOLCALL_MIN_CHARS")
    toolcall_first_emit_delay_ms: int = Field(200, env="TOOLCALL_FIRST_EMIT_DELAY_MS")
    full_toolcall_summary_on_finish: bool = Field(True, env="FULL_TOOLCALL_SUMMARY_ON_FINISH")
    simple_toolcall_assembly: bool = Field(True, env="SIMPLE_TOOLCALL_ASSEMBLY")
    json_repair_backup: bool = Field(True, env="JSON_REPAIR_BACKUP")
    
    # Retry configuration
    retry_429_max_attempts: int = Field(6, env="RETRY_429_MAX_ATTEMPTS")
    retry_429_base_ms: int = Field(1000, env="RETRY_429_BASE_MS")
    retry_429_max_ms: int = Field(30000, env="RETRY_429_MAX_MS")
    early_close_retry_attempts: int = Field(3, env="EARLY_CLOSE_RETRY_ATTEMPTS")
    early_close_window_ms: int = Field(2500, env="EARLY_CLOSE_WINDOW_MS")
    empty_finish_retry_attempts: int = Field(3, env="EMPTY_FINISH_RETRY_ATTEMPTS")
    empty_stream_fallback: bool = Field(True, env="EMPTY_STREAM_FALLBACK")
    
    # Logging and debugging
    log_requests: bool = Field(False, env="LOG_REQUESTS")
    log_upstream_lines: bool = Field(False, env="LOG_UPSTREAM_LINES")
    log_downstream_lines: bool = Field(False, env="LOG_DOWNSTREAM_LINES")
    tty_spinner: bool = Field(True, env="TTY_SPINNER")
    spinner_interval_ms: int = Field(150, env="SPINNER_INTERVAL_MS")
    heartbeat_ms: int = Field(0, env="HEARTBEAT_MS")
    status_comments: bool = Field(True, env="STATUS_COMMENTS")
    coalesce_ms: int = Field(120, env="COALESCE_MS")

    # Reasoning visibility
    expose_reasoning_as_content: bool = Field(
        True,
        env="EXPOSE_REASONING_AS_CONTENT",
        description="If true, forwards reasoning_content as delta.content for debugging"
    )
    
    # Server settings
    port: int = Field(8928, env="PORT")
    host: str = Field("0.0.0.0", env="HOST")
    
    # Preset configurations
    preset: Optional[str] = Field(None, env="PRESET")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation."""
        # Apply preset first if present
        if self.preset:
            _apply_preset(self)
        
        super().model_post_init(__context)
        
        # Validate upstream_base_url after all initialization including presets
        if not self.upstream_base_url:
            raise ValueError('upstream_base_url is required and cannot be None. Use --upstream-base-url or a preset (chutes, nahcrof).')
    
    def get_upstream_url(self, endpoint: str = "chat/completions", model: Optional[str] = None, incoming_path: Optional[str] = None) -> str:
        """Generate upstream URL based on configuration - MATCHES ORIGINAL LOGIC EXACTLY."""
        base = self.upstream_base_url.rstrip("/")
        
        # Determine endpoint like original: env override or mirror incoming
        if self.upstream_endpoint and self.upstream_endpoint != UpstreamEndpoint.AUTO:
            endpoint = self.upstream_endpoint.value
        elif incoming_path:
            if incoming_path.startswith("/v1/"):
                endpoint = incoming_path[len("/v1/"):].strip("/")
            else:
                endpoint = incoming_path.lstrip("/")
        
        # Fallback sanity check (matches original)
        if endpoint not in {"chat/completions", "completions"}:
            endpoint = "chat/completions"
        
        if self.url_model_placement == ModelPlacement.PATH and model:
            # Respect whether the upstream base already includes /v1 (MATCHES ORIGINAL)
            has_version = base.endswith("/v1")
            version_prefix = "" if has_version else "/v1"
            return f"{base}{version_prefix}/models/{model}/{endpoint}"
        else:
            # Respect whether the upstream base already includes /v1 (MATCHES ORIGINAL)
            has_version = base.endswith("/v1")
            version_prefix = "" if has_version else "/v1"
            return f"{base}{version_prefix}/{endpoint}"
    
    def apply_retry_policy(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter for 429 retries."""
        import random
        delay = min(
            self.retry_429_base_ms * (2 ** attempt),
            self.retry_429_max_ms
        )
        jitter = random.uniform(0.8, 1.2)
        return delay * jitter / 1000.0  # Convert to seconds


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        if _settings.preset:
            _apply_preset(_settings)
    return _settings


def set_settings(settings: Settings) -> None:
    """Set global settings (useful for testing)."""
    global _settings
    _settings = settings


def _apply_preset(settings: Settings) -> None:
    """Apply preset configurations."""
    presets = {
        "chutes": {
            "upstream_base_url": "https://llm.chutes.ai/v1",
            "upstream_endpoint": UpstreamEndpoint.CHAT_COMPLETIONS,
            "ensure_stream": True,
            "url_model_placement": ModelPlacement.BODY,
            "toolcall_emit_function_call_legacy": True,
        },
        "nahcrof": {
            "upstream_base_url": "https://ai.nahcrof.com/v2",
            "upstream_endpoint": UpstreamEndpoint.CHAT_COMPLETIONS,
            "ensure_stream": True,
            "url_model_placement": ModelPlacement.BODY,
            "default_model": "moonshotai/Kimi-K2-Thinking",
        }
    }
    
    preset_config = presets.get(settings.preset)
    if preset_config:
        for key, value in preset_config.items():
            setattr(settings, key, value)
