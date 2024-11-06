from typing import Any, Self

from draive.parameters import DataModel, Field
from draive.types.multimodal import Multimodal, MultimodalContent

__all__ = [
    "LMMCompletion",
    "LMMContextElement",
    "LMMInput",
    "LMMOutput",
    "LMMToolRequest",
    "LMMToolRequests",
    "LMMToolResponse",
    "LMMStreamChunk",
    "LMMStreamInput",
    "LMMStreamOutput",
]


class LMMInput(DataModel):
    @classmethod
    def of(
        cls,
        content: Multimodal,
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
        content: Multimodal,
        /,
    ) -> Self:
        return cls(content=MultimodalContent.of(content))

    content: MultimodalContent

    def __bool__(self) -> bool:
        return bool(self.content)


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


class LMMStreamChunk(DataModel):
    @classmethod
    def of(
        cls,
        content: Multimodal,
        /,
        final: bool = False,
    ) -> Self:
        return cls(
            content=MultimodalContent.of(content),
            is_final=final,
        )

    content: MultimodalContent
    is_final: bool

    def __bool__(self) -> bool:
        return bool(self.content)


LMMStreamInput = LMMStreamChunk | LMMToolResponse
LMMStreamOutput = LMMStreamChunk | LMMToolRequest
