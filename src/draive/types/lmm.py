from collections.abc import AsyncIterable
from typing import Any, Literal, Self

from draive.parameters import DataModel, Field
from draive.types.multimodal import MultimodalContent, MultimodalContentConvertible

__all__ = [
    "LMMCompletion",
    "LMMCompletionChunk",
    "LMMContextElement",
    "LMMInput",
    "LMMOutput",
    "LMMOutputStream",
    "LMMOutputStreamChunk",
    "LMMToolRequest",
    "LMMToolRequests",
    "LMMToolResponse",
    "LMMToolStatus",
]


class LMMInput(DataModel):
    @classmethod
    def of(
        cls,
        content: MultimodalContent | MultimodalContentConvertible,
        /,
    ) -> Self:
        return cls(content=MultimodalContent.of(content))

    content: MultimodalContent

    def __bool__(self) -> bool:
        return bool(self.content)


class LMMCompletion(DataModel):
    @classmethod
    def of(
        cls,
        content: MultimodalContent | MultimodalContentConvertible,
        /,
    ) -> Self:
        return cls(content=MultimodalContent.of(content))

    content: MultimodalContent

    def __bool__(self) -> bool:
        return bool(self.content)


class LMMCompletionChunk(DataModel):
    @classmethod
    def of(
        cls,
        content: MultimodalContent | MultimodalContentConvertible,
        /,
    ) -> Self:
        return cls(content=MultimodalContent.of(content))

    content: MultimodalContent

    def __bool__(self) -> bool:
        return bool(self.content)


class LMMToolStatus(DataModel):
    identifier: str
    tool: str
    status: Literal[
        "STARTED",
        "RUNNING",
        "FINISHED",
        "FAILED",
    ]
    content: MultimodalContent | None = None


class LMMToolResponse(DataModel):
    identifier: str
    tool: str
    content: MultimodalContent
    direct: bool
    error: bool


class LMMToolRequest(DataModel):
    identifier: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LMMToolRequests(DataModel):
    requests: list[LMMToolRequest]


LMMContextElement = LMMInput | LMMCompletion | LMMToolRequests | LMMToolResponse
LMMOutput = LMMCompletion | LMMToolRequests
LMMOutputStreamChunk = LMMCompletionChunk | LMMToolRequests
LMMOutputStream = AsyncIterable[LMMOutputStreamChunk]
