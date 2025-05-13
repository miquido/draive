from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from typing import (
    Any,
    Literal,
    Protocol,
    Self,
    TypedDict,
    overload,
    runtime_checkable,
)

from haiway import Default

from draive.commons import META_EMPTY, Meta
from draive.instructions import Instruction
from draive.multimodal import Multimodal, MultimodalContent
from draive.parameters import DataModel, ParametersSpecification

__all__ = (
    "LMMCompleting",
    "LMMCompletion",
    "LMMContext",
    "LMMContextElement",
    "LMMException",
    "LMMInput",
    "LMMOutput",
    "LMMSessionEvent",
    "LMMSessionOutput",
    "LMMSessionOutputSelection",
    "LMMStreamChunk",
    "LMMStreamInput",
    "LMMStreamOutput",
    "LMMToolError",
    "LMMToolFunctionSpecification",
    "LMMToolRequest",
    "LMMToolRequests",
    "LMMToolResponse",
    "LMMToolResponseHandling",
    "LMMToolResponses",
    "LMMToolSelection",
    "LMMToolSpecification",
)


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

    type: Literal["input"] = "input"
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

    type: Literal["completion"] = "completion"
    content: MultimodalContent
    meta: Meta = Default(META_EMPTY)

    def __bool__(self) -> bool:
        return bool(self.content)


LMMToolResponseHandling = Literal["error", "result", "direct_result"]


class LMMToolResponse(DataModel):
    type: Literal["tool_response"] = "tool_response"
    identifier: str
    tool: str
    content: MultimodalContent
    handling: LMMToolResponseHandling


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

    type: Literal["tool_responses"] = "tool_responses"
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

    type: Literal["tool_request"] = "tool_request"
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

    type: Literal["tool_requests"] = "tool_requests"
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

    type: Literal["stream_chunk"] = "stream_chunk"
    content: MultimodalContent
    eod: bool
    meta: Meta = Default(META_EMPTY)

    def __bool__(self) -> bool:
        return bool(self.content)


LMMStreamInput = LMMStreamChunk | LMMToolResponse
LMMStreamOutput = LMMStreamChunk | LMMToolRequest


class LMMSessionEvent(DataModel):
    @classmethod
    def of(
        cls,
        category: Literal["completed", "interrupted"] | str,
        /,
        *,
        meta: Meta | None = None,
    ) -> Self:
        return cls(
            category=category,
            meta=meta if meta is not None else META_EMPTY,
        )

    type: Literal["session_event"] = "session_event"
    category: Literal["completed", "interrupted"] | str
    meta: Meta


LMMSessionOutput = LMMStreamChunk | LMMToolRequest | LMMSessionEvent


@runtime_checkable
class LMMCompleting(Protocol):
    @overload
    async def __call__(
        self,
        *,
        instruction: Instruction | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        output: LMMOutputSelection,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> LMMOutput: ...

    @overload
    async def __call__(
        self,
        *,
        instruction: Instruction | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        output: LMMOutputSelection,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]: ...

    async def __call__(
        self,
        *,
        instruction: Instruction | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        output: LMMOutputSelection,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput] | LMMOutput: ...


type LMMSessionOutputSelection = (
    Sequence[Literal["text", "audio"]] | Literal["auto", "text", "audio"]
)


@runtime_checkable
class LMMSessionPreparing(Protocol):
    async def __call__(
        self,
        *,
        instruction: Instruction | None,
        initial_context: LMMContext | None,
        input_stream: AsyncIterator[LMMStreamInput],
        output: LMMSessionOutputSelection,
        tools: Sequence[LMMToolSpecification],
        tool_selection: LMMToolSelection,
        **extra: Any,
    ) -> AsyncIterator[LMMSessionOutput]: ...
