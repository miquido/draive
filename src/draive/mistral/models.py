from collections.abc import Mapping, Sequence
from typing import Any, Literal, NotRequired, Required, TypedDict

from draive.parameters import DataModel

__all__ = [
    "ChatCompletionResponse",
    "ChatCompletionResponseChoice",
    "ChatCompletionResponseStreamChoice",
    "ChatCompletionStreamResponse",
    "ChatDeltaMessageResponse",
    "ChatFunctionCall",
    "ChatFunctionCallResponse",
    "ChatMessage",
    "ChatMessageResponse",
    "ChatToolCallRequest",
    "ChatToolCallResponse",
    "EmbeddingObject",
    "EmbeddingResponse",
    "UsageInfo",
]


class UsageInfo(DataModel):
    prompt_tokens: int
    total_tokens: int
    completion_tokens: int | None = None


class EmbeddingObject(DataModel):
    object: str
    embedding: Sequence[float]
    index: int


class EmbeddingResponse(DataModel):
    id: str
    object: str
    data: Sequence[EmbeddingObject]
    model: str
    usage: UsageInfo


class ChatFunctionCall(TypedDict, total=False):
    name: Required[str]
    arguments: Required[str]


class ChatToolCallRequest(TypedDict, total=False):
    id: Required[str]
    function: Required[ChatFunctionCall]


class ChatMessage(TypedDict, total=False):
    role: Required[str]
    tool_call_id: NotRequired[str]
    content: Required[str | list[str]]
    name: NotRequired[str]
    tool_calls: NotRequired[list[ChatToolCallRequest]]
    prefix: NotRequired[bool]


class ChatFunctionCallResponse(DataModel):
    name: str
    arguments: Mapping[str, Any] | str


class ChatToolCallResponse(DataModel):
    id: str
    function: ChatFunctionCallResponse


class ChatDeltaMessageResponse(DataModel):
    role: str | None = None
    content: str | None = None
    tool_calls: Sequence[ChatToolCallResponse] | None = None


class ChatCompletionResponseStreamChoice(DataModel):
    index: int
    delta: ChatDeltaMessageResponse
    finish_reason: Literal["stop", "length", "error", "tool_calls"] | None = None


class ChatCompletionStreamResponse(DataModel):
    id: str
    model: str
    choices: Sequence[ChatCompletionResponseStreamChoice]
    created: int | None = None
    usage: UsageInfo | None = None


class ChatMessageResponse(DataModel):
    role: str
    content: Sequence[str] | str | None = None
    tool_calls: Sequence[ChatToolCallResponse] | None = None


class ChatCompletionResponseChoice(DataModel):
    index: int
    message: ChatMessageResponse
    finish_reason: Literal["stop", "length", "error", "tool_calls"] | None = None


class ChatCompletionResponse(DataModel):
    id: str
    object: str
    created: int
    model: str
    choices: Sequence[ChatCompletionResponseChoice]
    usage: UsageInfo
