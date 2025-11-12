"""Core data models for the proxy."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import json


class Role(str, Enum):
    """Message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """Chat message model."""
    role: Optional[str] = None
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = None
    
    @validator("role", pre=True)
    def validate_role(cls, v):
        """Validate role is one of the allowed values."""
        if v is None:
            return "assistant"  # Default role for streaming deltas
        if v not in [r.value for r in Role] + ["function"]:  # Some providers use "function"
            raise ValueError(f"Invalid role: {v}")
        return v
    
    @validator("content", pre=True)
    def validate_content(cls, v):
        """Convert multimodal content format to string if needed."""
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            # Multimodal format: extract text content
            text_parts = []
            for block in v:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if text:
                        text_parts.append(text)
            # Join all text parts or return first one if single
            if len(text_parts) == 1:
                return text_parts[0]
            elif len(text_parts) > 1:
                return "\n\n".join(text_parts)
            # If no text blocks found, try to convert list to string
            return str(v)
        return str(v)


class FunctionDefinition(BaseModel):
    """Function definition for tool calls."""
    name: str
    arguments: str  # JSON string
    description: Optional[str] = None
    
    def parse_arguments(self) -> Dict[str, Any]:
        """Parse arguments JSON string to dict."""
        try:
            return json.loads(self.arguments) if self.arguments else {}
        except json.JSONDecodeError:
            return {}


class ToolCall(BaseModel):
    """Tool call model."""
    id: Optional[str] = None
    type: str = "function"
    function: Union[FunctionDefinition, Dict[str, Any]]  # Allow dict for flexibility


class Choice(BaseModel):
    """Response choice model."""
    index: int = 0
    message: Optional[Message] = None
    delta: Optional[Message] = None
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Chat completion response model."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    
    class Config:
        extra = "allow"  # Allow provider-specific fields


class ChatCompletionChunk(BaseModel):
    """Streaming chunk response model."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    
    class Config:
        extra = "allow"


class ChunkedReasoningEvent(BaseModel):
    """Intermediate chunk emitted from tool call reasoning parser."""
    type: str  # "begin", "args", "end"
    index: Optional[int] = None
    name: Optional[str] = None
    text: Optional[str] = None
    
    class Config:
        extra = "allow"


class FunctionSpec(BaseModel):
    """Function specification for tool definitions."""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolSpec(BaseModel):
    """Tool specification."""
    type: str = "function"
    function: FunctionSpec


class FunctionParameter(BaseModel):
    """Function parameter specification."""
    name: str
    value: Optional[Any] = None
    description: Optional[str] = None
    required: bool = False
    
    def validate_type(self, value: Any) -> bool:
        """Basic type validation."""
        if value is None and self.required:
            return False
        return True


class StreamOptions(BaseModel):
    """Stream options for chat completions."""
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""
    model: Optional[str] = None
    messages: List[Message]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[ToolSpec]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    stream_options: Optional[StreamOptions] = None
    
    class Config:
        extra = "allow"


class CompletionRequest(BaseModel):
    """Text completion request model."""
    model: Optional[str] = None
    prompt: Union[str, List[Union[str, List[int]]]]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    echo: Optional[bool] = None
    best_of: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    seed: Optional[int] = None
    
    class Config:
        extra = "allow"
