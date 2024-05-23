from typing import Any, Literal, NotRequired, Required, TypedDict

from draive.parameters import DataModel

__all__ = [
    "UsageInfo",
    "EmbeddingObject",
    "EmbeddingResponse",
    "ChatFunctionCall",
    "ChatToolCallRequest",
    "ChatMessage",
    "ChatFunctionCallResponse",
    "ChatToolCallResponse",
    "ChatMessageResponse",
    "ChatCompletionResponseChoice",
    "ChatCompletionResponse",
    "ChatDeltaMessageResponse",
    "ChatCompletionResponseStreamChoice",
    "ChatCompletionStreamResponse",
]


class UsageInfo(DataModel):
    prompt_tokens: int
    total_tokens: int
    completion_tokens: int | None = None


class EmbeddingObject(DataModel):
    object: str
    embedding: list[float]
    index: int


class EmbeddingResponse(DataModel):
    id: str
    object: str
    data: list[EmbeddingObject]
    model: str
    usage: UsageInfo


class ChatFunctionCall(TypedDict, total=False):
    name: Required[str]
    arguments: Required[str]


class ChatToolCallRequest(TypedDict, total=False):
    id: Required[str]
    type: Required[Literal["function"]]
    function: Required[ChatFunctionCall]


class ChatMessage(TypedDict, total=False):
    role: Required[str]
    content: Required[str | list[str]]
    name: NotRequired[str]
    tool_calls: NotRequired[list[ChatToolCallRequest]]


class ChatFunctionCallResponse(DataModel):
    name: str
    arguments: dict[str, Any] | str


class ChatToolCallResponse(DataModel):
    id: str
    type: Literal["function"]
    function: ChatFunctionCallResponse


class ChatDeltaMessageResponse(DataModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[ChatToolCallResponse] | None = None


class ChatCompletionResponseStreamChoice(DataModel):
    index: int
    delta: ChatDeltaMessageResponse
    finish_reason: Literal["stop", "length", "error", "tool_calls"] | None = None


class ChatCompletionStreamResponse(DataModel):
    id: str
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    created: int | None = None
    usage: UsageInfo | None = None


class ChatMessageResponse(DataModel):
    role: str
    content: list[str] | str | None = None
    tool_calls: list[ChatToolCallResponse] | None = None


class ChatCompletionResponseChoice(DataModel):
    index: int
    message: ChatMessageResponse
    finish_reason: Literal["stop", "length", "error", "tool_calls"] | None = None


class ChatCompletionResponse(DataModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo
