from typing import Any, Literal, TypedDict

__all__ = (
    "ChatCompletionResponse",
    "ChatCompletionResponseOutput",
    "ChatCompletionResponseUsage",
    "ChatMessage",
    "ChatMessageContent",
    "ChatMessageImage",
    "ChatMessageImageContent",
    "ChatMessageImageContentSource",
    "ChatMessageText",
    "ChatMessageToolCall",
    "ChatMessageToolCallContent",
    "ChatMessageToolResult",
    "ChatMessageToolResultContent",
    "ChatTool",
)


class ChatMessageText(TypedDict):
    text: str


class ChatMessageImageContentSource(TypedDict):
    bytes: bytes


class ChatMessageImageContent(TypedDict):
    format: Literal["png", "jpeg", "gif"]
    source: ChatMessageImageContentSource


class ChatMessageImage(TypedDict):
    image: ChatMessageImageContent


class ChatMessageToolCallContent(TypedDict):
    toolUseId: str
    name: str
    input: Any


class ChatMessageToolCall(TypedDict):
    toolUse: ChatMessageToolCallContent


class ChatMessageToolResultContent(TypedDict):
    toolUseId: str
    content: list[ChatMessageText | ChatMessageImage]
    status: Literal["success", "error"]


class ChatMessageToolResult(TypedDict):
    toolResult: ChatMessageToolResultContent


type ChatMessageContent = (
    ChatMessageText | ChatMessageImage | ChatMessageToolCall | ChatMessageToolResult
)


class ChatMessage(TypedDict):
    role: str
    content: list[ChatMessageContent]


class ChatCompletionResponseUsage(TypedDict):
    inputTokens: int
    outputTokens: int
    totalTokens: int


class ChatCompletionResponseOutput(TypedDict):
    message: ChatMessage


class ChatCompletionResponse(TypedDict):
    output: ChatCompletionResponseOutput
    stopReason: Literal[
        "end_turn",
        "tool_use",
        "max_tokens",
        "stop_sequence",
        "guardrail_intervened",
        "content_filtered",
    ]
    usage: ChatCompletionResponseUsage


class ChatTool(TypedDict):
    name: str
    description: str
    inputSchema: Any
