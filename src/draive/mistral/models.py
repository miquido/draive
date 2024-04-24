from typing import Any, Literal, NotRequired, Required, TypedDict

from draive.types import Model

__all__ = [
    "UsageInfo",
    "EmbeddingObject",
    "EmbeddingResponse",
    "FunctionCall",
    "ToolCall",
    "ChatMessage",
    "FunctionCallResponse",
    "ToolCallResponse",
    "ChatMessageResponse",
    "ChatCompletionResponseChoice",
    "ChatCompletionResponse",
    "ChatDeltaMessageResponse",
    "ChatCompletionResponseStreamChoice",
    "ChatCompletionStreamResponse",
]


class UsageInfo(Model):
    prompt_tokens: int
    total_tokens: int
    completion_tokens: int | None = None


class EmbeddingObject(Model):
    object: str
    embedding: list[float]
    index: int


class EmbeddingResponse(Model):
    id: str
    object: str
    data: list[EmbeddingObject]
    model: str
    usage: UsageInfo


class FunctionCall(TypedDict, total=False):
    name: Required[str]
    arguments: Required[str]


class ToolCall(TypedDict, total=False):
    id: Required[str]
    type: Required[Literal["function"]]
    function: Required[FunctionCall]


class ChatMessage(TypedDict, total=False):
    role: Required[str]
    content: Required[str | list[str]]
    name: NotRequired[str]
    tool_calls: NotRequired[list[ToolCall]]


class FunctionCallResponse(Model):
    name: str
    arguments: dict[str, Any] | str


class ToolCallResponse(Model):
    id: str
    type: Literal["function"]
    function: FunctionCallResponse


class ChatDeltaMessageResponse(Model):
    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCallResponse] | None = None


class ChatCompletionResponseStreamChoice(Model):
    index: int
    delta: ChatDeltaMessageResponse
    finish_reason: Literal["stop", "length", "error", "tool_calls"] | None = None


class ChatCompletionStreamResponse(Model):
    id: str
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    created: int | None = None
    usage: UsageInfo | None = None


class ChatMessageResponse(Model):
    role: str
    content: list[str] | str | None = None
    tool_calls: list[ToolCallResponse] | None = None


class ChatCompletionResponseChoice(Model):
    index: int
    message: ChatMessageResponse
    finish_reason: Literal["stop", "length", "error", "tool_calls"] | None = None


class ChatCompletionResponse(Model):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo
