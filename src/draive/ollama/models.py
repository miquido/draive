from draive.parameters import DataModel

__all__ = [
    "ChatCompletionResponse",
    "ChatMessage",
]


class ChatMessage(DataModel):
    role: str
    content: str
    images: list[str] | None = None


class ChatCompletionResponse(DataModel):
    model: str
    message: ChatMessage
    prompt_eval_count: int | None = None
    eval_count: int | None = None
