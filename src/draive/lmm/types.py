from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from typing import (
    Any,
    Literal,
    Protocol,
    Self,
    TypedDict,
    runtime_checkable,
)

from haiway import Default, State

from draive.commons import META_EMPTY, Meta
from draive.instructions import Instruction
from draive.multimodal import Multimodal, MultimodalContent
from draive.parameters import DataModel, ParametersSpecification

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
    Sequence[Literal["text", "image", "audio", "video"]]
    | Literal["auto", "text", "json", "image", "audio", "video"]
    | type[DataModel]
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
        meta: Meta | None = None,
    ) -> Self:
        return cls(
            content=MultimodalContent.of(content),
            meta=meta if meta is not None else META_EMPTY,
        )

    content: MultimodalContent
    meta: Meta = Default(META_EMPTY)

    def __bool__(self) -> bool:
        return bool(self.content)


class LMMCompletion(DataModel):
    @classmethod
    def of(
        cls,
        content: Multimodal,
        /,
        meta: Meta | None = None,
    ) -> Self:
        return cls(
            content=MultimodalContent.of(content),
            meta=meta if meta is not None else META_EMPTY,
        )

    content: MultimodalContent
    meta: Meta = Default(META_EMPTY)

    def __bool__(self) -> bool:
        return bool(self.content)


class LMMToolResponse(DataModel):
    identifier: str
    tool: str
    content: MultimodalContent
    direct: bool
    error: bool


class LMMToolResponses(DataModel):
    @classmethod
    def of(
        cls,
        responses: Sequence[LMMToolResponse],
        /,
        meta: Meta | None = None,
    ) -> Self:
        return cls(
            responses=responses,
            meta=meta if meta is not None else META_EMPTY,
        )

    responses: Sequence[LMMToolResponse]
    meta: Meta = Default(META_EMPTY)


class LMMToolRequest(DataModel):
    @classmethod
    def of(
        cls,
        identifier: str,
        /,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        meta: Meta | None = None,
    ) -> Self:
        return cls(
            identifier=identifier,
            tool=tool,
            arguments=arguments if arguments is not None else {},
            meta=meta if meta is not None else META_EMPTY,
        )

    identifier: str
    tool: str
    arguments: Mapping[str, Any] = Default(factory=dict)
    meta: Meta = Default(META_EMPTY)


class LMMToolRequests(DataModel):
    @classmethod
    def of(
        cls,
        requests: Sequence[LMMToolRequest],
        /,
        content: MultimodalContent | None = None,
        meta: Meta | None = None,
    ) -> Self:
        return cls(
            requests=requests,
            content=content,
            meta=meta if meta is not None else META_EMPTY,
        )

    requests: Sequence[LMMToolRequest]
    content: MultimodalContent | None = None
    meta: Meta = Default(META_EMPTY)


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
        meta: Meta | None = None,
    ) -> Self:
        return cls(
            content=MultimodalContent.of(content),
            eod=eod,
            meta=meta if meta is not None else META_EMPTY,
        )

    content: MultimodalContent
    eod: bool
    meta: Meta = Default(META_EMPTY)

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
