from collections.abc import AsyncIterator
from typing import Any, Self

from draive.parameters import Field
from draive.parameters.model import DataModel
from draive.types.instruction import Instruction
from draive.types.multimodal import MultimodalContent, MultimodalContentElement

__all__ = [
    "LMMCompletion",
    "LMMCompletionChunk",
    "LMMContextElement",
    "LMMInput",
    "LMMInstruction",
    "LMMOutput",
    "LMMOutputStream",
    "LMMOutputStreamChunk",
    "LMMToolRequest",
    "LMMToolRequests",
    "LMMToolResponse",
]


class LMMInstruction(DataModel):
    @classmethod
    def of(
        cls,
        instruction: Instruction | str,
        /,
        **variables: object,
    ) -> Self:
        match instruction:
            case str(content):
                return cls(content=content.format_map(variables) if variables else content)

            case instruction:
                return cls(content=instruction.format(**variables))

    content: str

    def __bool__(self) -> bool:
        return bool(self.content)


class LMMInput(DataModel):
    @classmethod
    def of(
        cls,
        content: MultimodalContent | MultimodalContentElement,
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
        content: MultimodalContent | MultimodalContentElement,
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
        content: MultimodalContent | MultimodalContentElement,
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


class LMMToolRequest(DataModel):
    identifier: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LMMToolRequests(DataModel):
    requests: list[LMMToolRequest]


LMMContextElement = LMMInstruction | LMMInput | LMMCompletion | LMMToolRequests | LMMToolResponse
LMMOutput = LMMCompletion | LMMToolRequests
LMMOutputStreamChunk = LMMCompletionChunk | LMMToolRequests
LMMOutputStream = AsyncIterator[LMMOutputStreamChunk]
