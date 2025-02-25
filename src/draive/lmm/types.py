from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from typing import (
    Any,
    Literal,
    Protocol,
    Self,
    TypedDict,
    runtime_checkable,
)

from haiway import State

from draive.instructions import Instruction
from draive.multimodal import Multimodal, MultimodalContent
from draive.parameters import DataModel, Field, ParametersSpecification

__all__ = [
    "LMMCompletion",
    "LMMContext",
    "LMMContextElement",
    "LMMException",
    "LMMInput",
    "LMMInvocating",
    "LMMOutput",
    "LMMStreamChunk",
    "LMMStreamInput",
    "LMMStreamOutput",
    "LMMStreamProperties",
    "LMMStreaming",
    "LMMToolError",
    "LMMToolFunctionSpecification",
    "LMMToolRequest",
    "LMMToolRequests",
    "LMMToolResponse",
    "LMMToolResponses",
    "LMMToolSelection",
    "LMMToolSpecification",
]


class LMMToolFunctionSpecification(TypedDict):
    name: str
    description: str | None
    parameters: ParametersSpecification | None


type LMMToolSpecification = LMMToolFunctionSpecification
type LMMOutputSelection = (
    Literal["auto", "text", "json", "image", "audio", "video"] | type[DataModel]
)
type LMMToolSelection = Literal["auto", "required", "none"] | LMMToolSpecification


class LMMException(Exception):
    pass


class LMMToolError(LMMException):
    def __init__(
        self,
        *args: object,
        content: MultimodalContent,
    ) -> None:
        super().__init__(*args)
        self.content: MultimodalContent = content


class LMMInput(DataModel):
    @classmethod
    def of(
        cls,
        content: Multimodal,
        /,
        meta: Mapping[str, str | float | int | bool | None] | None = None,
    ) -> Self:
        return cls(
            content=MultimodalContent.of(content),
            meta=meta,
        )

    content: MultimodalContent
    meta: Mapping[str, str | float | int | bool | None] | None = None

    def __bool__(self) -> bool:
        return bool(self.content)


class LMMCompletion(DataModel):
    @classmethod
    def of(
        cls,
        content: Multimodal,
        /,
        meta: Mapping[str, str | float | int | bool | None] | None = None,
    ) -> Self:
        return cls(
            content=MultimodalContent.of(content),
            meta=meta,
        )

    content: MultimodalContent
    meta: Mapping[str, str | float | int | bool | None] | None = None

    def __bool__(self) -> bool:
        return bool(self.content)


class LMMToolResponse(DataModel):
    identifier: str
    tool: str
    content: MultimodalContent
    direct: bool
    error: bool


class LMMToolResponses(DataModel):
    responses: Sequence[LMMToolResponse]


class LMMToolRequest(DataModel):
    identifier: str
    tool: str
    arguments: Mapping[str, Any] = Field(default_factory=dict)


class LMMToolRequests(DataModel):
    completion: LMMCompletion | None = None
    requests: Sequence[LMMToolRequest]


LMMContextElement = LMMInput | LMMCompletion | LMMToolRequests | LMMToolResponses
LMMContext = Sequence[LMMContextElement]
LMMOutput = LMMCompletion | LMMToolRequests


class LMMStreamChunk(DataModel):
    @classmethod
    def of(
        cls,
        content: Multimodal,
        /,
        eod: bool = False,
    ) -> Self:
        return cls(
            content=MultimodalContent.of(content),
            eod=eod,
        )

    content: MultimodalContent
    eod: bool

    def __bool__(self) -> bool:
        return bool(self.content)


LMMStreamInput = LMMStreamChunk | LMMToolResponse
LMMStreamOutput = LMMStreamChunk | LMMToolRequest


@runtime_checkable
class LMMInvocating(Protocol):
    async def __call__(
        self,
        *,
        instruction: Instruction | str | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        output: LMMOutputSelection,
        **extra: Any,
    ) -> LMMOutput: ...


class LMMStreamProperties(State):
    instruction: Instruction | str | None = None
    tools: Sequence[LMMToolSpecification] | None
    tool_selection: LMMToolSelection = "auto"


@runtime_checkable
class LMMStreaming(Protocol):
    async def __call__(
        self,
        *,
        properties: AsyncIterator[LMMStreamProperties],
        input: AsyncIterator[LMMStreamInput],  # noqa: A002
        context: LMMContext | None,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]: ...
