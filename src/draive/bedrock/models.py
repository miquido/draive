from typing import Any, Literal, NotRequired, TypedDict

__all__ = (
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
    format: Literal["png", "jpeg", "gif", "webp"]
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


class ChatTool(TypedDict):
    name: str
    # empty description is not allowed, it has to be skipped when unavailable
    description: NotRequired[str]
    inputSchema: Any
