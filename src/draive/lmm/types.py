from collections.abc import (
    AsyncIterator,
    Iterable,
    Mapping,
    Sequence,
)
from types import TracebackType
from typing import (
    Any,
    ClassVar,
    Literal,
    Protocol,
    Self,
    TypedDict,
    overload,
    runtime_checkable,
)

from haiway import Default, State, as_tuple, ctx

from draive.commons import META_EMPTY, Meta, MetaValues
from draive.multimodal import Multimodal, MultimodalContent
from draive.parameters import DataModel, ParametersSpecification
from draive.utils import Memory

__all__ = (
    "LMMCompleting",
    "LMMCompletion",
    "LMMContext",
    "LMMContextElement",
    "LMMException",
    "LMMInput",
    "LMMInstruction",
    "LMMMemory",
    "LMMOutput",
    "LMMOutputDecoder",
    "LMMOutputInvalid",
    "LMMOutputLimit",
    "LMMSession",
    "LMMSessionClosing",
    "LMMSessionEvent",
    "LMMSessionInput",
    "LMMSessionOpening",
    "LMMSessionOutput",
    "LMMSessionOutputSelection",
    "LMMSessionReading",
    "LMMSessionScope",
    "LMMSessionWriting",
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
    "LMMTools",
)

LMMInstruction = str


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


class LMMTools(State):
    none: ClassVar[Self]  # defined after the class

    @classmethod
    def of(
        cls,
        tools: Iterable[LMMToolSpecification] | None,
        /,
        *,
        selection: LMMToolSelection = "auto",
    ) -> Self:
        specifications: tuple[LMMToolSpecification, ...] | None = as_tuple(tools)
        if not specifications:
            return cls(
                selection="none",
                specifications=(),
            )

        if not isinstance(selection, str) and selection not in specifications:
            return cls(
                selection="auto",
                specifications=specifications,
            )

        return cls(
            selection=selection,
            specifications=specifications,
        )

    selection: LMMToolSelection
    specifications: Sequence[LMMToolSpecification]

    def __bool__(self) -> bool:
        return bool(self.specifications)


LMMTools.none = LMMTools.of((), selection="none")


class LMMException(Exception):
    pass


class LMMOutputLimit(LMMException):
    pass


class LMMOutputInvalid(LMMException):
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
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            content=MultimodalContent.of(content),
            meta=Meta.of(meta),
        )

    content: MultimodalContent
    meta: Meta = META_EMPTY

    def __bool__(self) -> bool:
        return bool(self.content)


class LMMCompletion(DataModel):
    @classmethod
    def of(
        cls,
        content: Multimodal,
        /,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            content=MultimodalContent.of(content),
            meta=Meta.of(meta),
        )

    content: MultimodalContent
    meta: Meta = META_EMPTY

    def __bool__(self) -> bool:
        return bool(self.content)


@runtime_checkable
class LMMOutputDecoder(Protocol):
    def __call__(
        self,
        content: MultimodalContent,
    ) -> MultimodalContent: ...


LMMToolResponseHandling = Literal["error", "result", "direct_result"]


class LMMToolResponse(DataModel):
    @classmethod
    def of(
        cls,
        identifier: str,
        /,
        *,
        tool: str,
        content: MultimodalContent,
        handling: LMMToolResponseHandling,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            identifier=identifier,
            tool=tool,
            content=content,
            handling=handling,
            meta=Meta.of(meta),
        )

    identifier: str
    tool: str
    content: MultimodalContent
    handling: LMMToolResponseHandling
    meta: Meta = META_EMPTY


class LMMToolResponses(DataModel):
    @classmethod
    def of(
        cls,
        responses: Sequence[LMMToolResponse],
        /,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            responses=responses,
            meta=Meta.of(meta),
        )

    responses: Sequence[LMMToolResponse]
    meta: Meta = META_EMPTY


class LMMToolRequest(DataModel):
    @classmethod
    def of(
        cls,
        identifier: str,
        /,
        *,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            identifier=identifier,
            tool=tool,
            arguments=arguments if arguments is not None else {},
            meta=Meta.of(meta),
        )

    identifier: str
    tool: str
    arguments: Mapping[str, Any] = Default(factory=dict)
    meta: Meta = META_EMPTY


class LMMToolRequests(DataModel):
    @classmethod
    def of(
        cls,
        requests: Sequence[LMMToolRequest],
        /,
        content: MultimodalContent | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            requests=requests,
            content=content,
            meta=Meta.of(meta),
        )

    requests: Sequence[LMMToolRequest]
    content: MultimodalContent | None = None
    meta: Meta = META_EMPTY


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
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            content=MultimodalContent.of(content),
            eod=eod,
            meta=Meta.of(meta),
        )

    content: MultimodalContent
    eod: bool
    meta: Meta = META_EMPTY

    def __bool__(self) -> bool:
        return bool(self.content)


LMMStreamInput = LMMStreamChunk | LMMToolResponse
LMMStreamOutput = LMMStreamChunk | LMMToolRequest
LMMMemory = Memory[LMMContext, LMMContextElement]


@runtime_checkable
class LMMCompleting(Protocol):
    @overload
    async def __call__(
        self,
        *,
        instruction: LMMInstruction | None,
        context: LMMContext,
        tools: LMMTools | None,
        output: LMMOutputSelection,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> LMMOutput: ...

    @overload
    async def __call__(
        self,
        *,
        instruction: LMMInstruction | None,
        context: LMMContext,
        tools: LMMTools | None,
        output: LMMOutputSelection,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]: ...

    async def __call__(
        self,
        *,
        instruction: LMMInstruction | None,
        context: LMMContext,
        tools: LMMTools | None,
        output: LMMOutputSelection,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput] | LMMOutput: ...


class LMMSessionEvent(DataModel):
    @classmethod
    def of(
        cls,
        category: str,
        /,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            category=category,
            meta=Meta.of(meta),
        )

    category: str
    meta: Meta


LMMSessionInput = LMMStreamChunk | LMMToolResponse | LMMSessionEvent
LMMSessionOutput = LMMStreamChunk | LMMToolRequest | LMMSessionEvent
type LMMSessionOutputSelection = (
    Sequence[Literal["text", "audio"]] | Literal["auto", "text", "audio"]
)


@runtime_checkable
class LMMSessionReading(Protocol):
    async def __call__(self) -> LMMSessionOutput: ...


@runtime_checkable
class LMMSessionWriting(Protocol):
    async def __call__(
        self,
        input: LMMSessionInput,  # noqa: A002
    ) -> None: ...


class LMMSession(State):
    @classmethod
    async def read(cls) -> LMMSessionOutput:
        return await ctx.state(cls).reading()

    @classmethod
    async def reader(cls) -> AsyncIterator[LMMSessionOutput]:
        session: Self = ctx.state(cls)
        while True:  # breaks on exception
            yield await session.reading()

    @classmethod
    async def write(
        cls,
        input: LMMSessionInput,  # noqa: A002
    ) -> None:
        await ctx.state(cls).writing(input=input)

    @classmethod
    async def writer(
        cls,
        input: AsyncIterator[LMMSessionInput],  # noqa: A002
    ) -> None:
        session: Self = ctx.state(cls)
        while True:  # breaks on exception
            await session.writing(input=await anext(input))

    reading: LMMSessionReading
    writing: LMMSessionWriting


@runtime_checkable
class LMMSessionOpening(Protocol):
    async def __call__(self) -> LMMSession: ...


@runtime_checkable
class LMMSessionClosing(Protocol):
    async def __call__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...


class LMMSessionScope(State):
    opening: LMMSessionOpening
    closing: LMMSessionClosing

    async def __aenter__(self) -> LMMSession:
        return await self.opening()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.closing(
            exc_type=exc_type,
            exc_val=exc_val,
            exc_tb=exc_tb,
        )


@runtime_checkable
class LMMSessionPreparing(Protocol):
    async def __call__(
        self,
        *,
        instruction: LMMInstruction | None,
        memory: LMMMemory | None,
        tools: LMMTools | None,
        output: LMMSessionOutputSelection,
        **extra: Any,
    ) -> LMMSessionScope: ...
