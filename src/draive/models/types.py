from asyncio import sleep
from collections.abc import (
    AsyncIterable,
    Collection,
    Iterable,
    Mapping,
    MutableSequence,
    Sequence,
)
from types import TracebackType
from typing import (
    Any,
    ClassVar,
    Literal,
    NotRequired,
    Protocol,
    Required,
    Self,
    TypedDict,
    final,
    runtime_checkable,
)

from haiway import (
    BasicValue,
    Meta,
    MetaValues,
    State,
    TypeSpecification,
    ctx,
)

from draive.multimodal import Multimodal, MultimodalContent, TextContent
from draive.multimodal.content import MultimodalContentPart

__all__ = (
    "ModelContext",
    "ModelContextElement",
    "ModelException",
    "ModelGenerating",
    "ModelInput",
    "ModelInputBlock",
    "ModelInputBlocks",
    "ModelInputChunk",
    "ModelInputInvalid",
    "ModelInstructions",
    "ModelOutput",
    "ModelOutputBlock",
    "ModelOutputBlocks",
    "ModelOutputChunk",
    "ModelOutputFailed",
    "ModelOutputInvalid",
    "ModelOutputLimit",
    "ModelOutputSelection",
    "ModelOutputStream",
    "ModelRateLimit",
    "ModelReasoning",
    "ModelReasoningChunk",
    "ModelSession",
    "ModelSessionClosing",
    "ModelSessionEvent",
    "ModelSessionInputChunk",
    "ModelSessionInputStream",
    "ModelSessionOpening",
    "ModelSessionOutputChunk",
    "ModelSessionOutputSelection",
    "ModelSessionOutputStream",
    "ModelSessionPreparing",
    "ModelSessionReading",
    "ModelSessionScope",
    "ModelSessionWriting",
    "ModelToolDetachedHandling",
    "ModelToolFunctionSpecification",
    "ModelToolHandling",
    "ModelToolParametersSpecification",
    "ModelToolRequest",
    "ModelToolResponse",
    "ModelToolSpecification",
    "ModelToolStatus",
    "ModelTools",
    "ModelToolsSelection",
)


class ModelException(Exception):
    """Base exception raised by model providers.

    Parameters
    ----------
    *args
        Exception message parts passed to ``Exception``.
    provider
        Provider identifier that produced the failure.
    model
        Provider model identifier related to the failure.
    """

    __slots__ = (
        "model",
        "provider",
    )

    def __init__(
        self,
        *args: Any,
        provider: str,
        model: str,
    ) -> None:
        super().__init__(*args)
        self.provider: str = provider
        self.model: str = model


@final
class ModelInputInvalid(ModelException):
    """Raised when a provider rejects request input as invalid.

    Parameters
    ----------
    provider
        Provider identifier that rejected the input.
    model
        Provider model identifier used for the request.
    """

    __slots__ = ("reason",)

    def __init__(
        self,
        *,
        provider: str,
        model: str,
    ) -> None:
        super().__init__(
            f"Invalid input for {model} by {provider}",
            provider=provider,
            model=model,
        )


@final
class ModelOutputInvalid(ModelException):
    """Raised when a provider returns malformed or unsupported output.

    Parameters
    ----------
    provider
        Provider identifier that produced invalid output.
    model
        Provider model identifier used for generation.
    reason
        Human-readable validation failure reason.
    """

    __slots__ = ("reason",)

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        reason: str,
    ) -> None:
        super().__init__(
            f"Invalid output from {model} by {provider}, reason: {reason}",
            provider=provider,
            model=model,
        )
        self.reason: str = reason


@final
class ModelRateLimit(ModelException):
    """Raised when a provider applies rate limiting.

    Parameters
    ----------
    provider
        Provider identifier that applied the limit.
    model
        Provider model identifier affected by the limit.
    retry_after
        Delay in seconds before retry should be attempted.
    """

    __slots__ = ("retry_after",)

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        retry_after: float,
    ) -> None:
        super().__init__(
            f"Rate limit for {model} by {provider}, retry after {retry_after:.2f}s",
            provider=provider,
            model=model,
        )
        self.retry_after: float = retry_after

    async def wait(self) -> None:
        """Sleep for the configured retry delay.

        Returns
        -------
        None
        """
        await sleep(self.retry_after)


@final
class ModelOutputFailed(ModelException):
    """Raised when provider output ends with a terminal failure state.

    Parameters
    ----------
    provider
        Provider identifier that failed to produce output.
    model
        Provider model identifier used for generation.
    reason
        Human-readable failure reason.
    """

    __slots__ = ("reason",)

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        reason: str,
    ) -> None:
        super().__init__(
            f"Output failed for {model} by {provider}, reason: {reason}",
            provider=provider,
            model=model,
        )
        self.reason: str = reason


ModelInstructions = str
"""Model-level instruction text consumed by a generation call."""


@final
class ModelToolParametersSpecification(TypedDict, total=False):
    """Schema description used to validate model tool call arguments.

    Keys
    ----
    type
        JSON Schema root type. Must be ``"object"``.
    properties
        Mapping of argument names to typed specifications.
    description
        Optional description of the parameter object.
    required
        Optional list of required argument names.
    additionalProperties
        Must be ``False`` to enforce strict argument validation.
    """

    type: Required[Literal["object"]]
    properties: Required[Mapping[str, TypeSpecification]]
    required: NotRequired[Sequence[str]]
    additionalProperties: Required[Literal[False]]


@final
class ModelToolFunctionSpecification(State, serializable=True):
    """Serializable tool declaration exposed to a provider.

    Attributes
    ----------
    name
        Stable tool name used in model tool requests.
    description
        Optional model-facing description of tool behavior.
    parameters
        Optional argument schema expected by the tool.
    meta
        Additional metadata propagated with this declaration.
    """

    @classmethod
    def of(
        cls,
        *,
        name: str,
        description: str | None = None,
        parameters: ModelToolParametersSpecification | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a tool function specification with normalized metadata.

        Parameters
        ----------
        name
            Stable tool name used during dispatch.
        description
            Optional model-facing tool description.
        parameters
            Optional argument schema for tool invocation.
        meta
            Optional metadata merged into the resulting state.

        Returns
        -------
        Self
            New tool specification instance.
        """
        return cls(
            name=name,
            description=description,
            parameters=parameters,
            meta=Meta.of(meta),
        )

    name: str
    description: str | None = None
    parameters: ModelToolParametersSpecification | None = None
    meta: Meta = Meta.empty


ModelToolSpecification = ModelToolFunctionSpecification
"""Alias for provider tool specification state."""
ModelToolsSelection = Literal["auto", "required", "none"] | ModelToolSpecification
"""Strategy controlling whether and how a model may call tools."""


@final
class ModelTools(State, serializable=True):
    """Toolset and selection strategy passed to model generation.

    Attributes
    ----------
    specification
        Ordered collection of tool declarations available to the model.
    selection
        Provider-specific tool choice policy.
    """

    none: ClassVar[Self]  # defined after the class

    @classmethod
    def of(
        cls,
        *specification: ModelToolSpecification,
        selection: ModelToolsSelection = "auto",
    ) -> Self:
        """Create a toolset with the provided specifications.

        Parameters
        ----------
        *specification
            Tool specifications exposed to the model.
        selection
            Tool-choice strategy used by provider adapters.

        Returns
        -------
        Self
            Toolset state object.
        """
        return cls(
            specification=specification,
            selection=selection,
        )

    specification: Sequence[ModelToolSpecification]
    selection: ModelToolsSelection

    def __bool__(self) -> bool:
        return bool(self.specification)


ModelTools.none = ModelTools(
    specification=(),
    selection="none",
)


@final
class ModelToolRequest(State, serializable=True):
    """Model-emitted request to execute a concrete tool.

    Attributes
    ----------
    identifier
        Provider-generated call identifier.
    tool
        Requested tool name.
    arguments
        Typed argument payload for the tool call.
    meta
        Metadata attached by provider adapters.
    """

    @classmethod
    def of(
        cls,
        identifier: str,
        /,
        *,
        tool: str,
        arguments: Mapping[str, BasicValue] | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a tool request with normalized defaults.

        Parameters
        ----------
        identifier
            Provider-generated tool call identifier.
        tool
            Requested tool name.
        arguments
            Optional argument mapping. Defaults to an empty mapping.
        meta
            Optional metadata attached to the request.

        Returns
        -------
        Self
            Tool request instance.
        """
        return cls(
            identifier=identifier,
            tool=tool,
            arguments=arguments if arguments is not None else {},
            meta=Meta.of(meta),
        )

    identifier: str
    tool: str
    arguments: Mapping[str, BasicValue]
    meta: Meta = Meta.empty


class ModelToolDetachedHandling(State):
    detach_message: Multimodal


ModelToolHandling = Literal["response", "output"] | ModelToolDetachedHandling
"""Routing policy describing how successful tool results are handled."""

ModelToolStatus = Literal["success", "error"]
"""Outcome status describing whether tool execution succeeded."""


@final
class ModelToolResponse(State, serializable=True):
    """Application-produced tool execution result.

    Attributes
    ----------
    identifier
        Identifier matching the originating tool request.
    tool
        Tool name for the reported result.
    result
        Multimodal result payload returned by the tool.
    handling
        Routing policy describing how successful tool results are handled.
    status
        Outcome status of the tool execution.
    meta
        Metadata attached to the response.
    """

    @classmethod
    def of(
        cls,
        identifier: str,
        /,
        *,
        tool: str,
        result: Multimodal,
        handling: ModelToolHandling = "response",
        status: ModelToolStatus = "success",
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a tool response with normalized content and metadata.

        Parameters
        ----------
        identifier
            Identifier of the related tool request.
        tool
            Tool name that produced the result.
        result
            Tool output content.
        handling
            Routing policy for successful tool output.
        status
            Outcome status used by providers and orchestration.
        meta
            Optional metadata attached to the response.

        Returns
        -------
        Self
            Tool response instance.
        """
        return cls(
            identifier=identifier,
            tool=tool,
            result=MultimodalContent.of(result),
            handling=handling,
            status=status,
            meta=Meta.of(meta),
        )

    identifier: str
    tool: str
    result: MultimodalContent
    handling: ModelToolHandling = "response"
    status: ModelToolStatus = "success"
    meta: Meta = Meta.empty


ModelInputBlock = MultimodalContent | ModelToolResponse
"""Single input block accepted by model generation."""
ModelInputBlocks = Sequence[ModelInputBlock]
"""Ordered model input blocks."""


@final
class ModelInput(State, serializable=True):
    """Input payload passed into model generation.

    Attributes
    ----------
    input
        Ordered sequence of content and tool response blocks.
    meta
        Metadata associated with this input payload.
    """

    @classmethod
    def of(
        cls,
        /,
        *blocks: ModelInputBlock,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create an input payload from blocks.

        Parameters
        ----------
        *blocks
            Input blocks in delivery order.
        meta
            Optional metadata attached to the payload.

        Returns
        -------
        Self
            Input state object.
        """
        return cls(
            input=blocks,
            meta=Meta.of(meta),
        )

    input: ModelInputBlocks
    meta: Meta = Meta.empty

    @property
    def content(self) -> MultimodalContent:
        """Return only multimodal content blocks."""
        return MultimodalContent.of(
            *(block for block in self.input if isinstance(block, MultimodalContent))
        )

    @property
    def contains_tools(self) -> bool:
        return any(isinstance(block, ModelToolResponse) for block in self.input)

    @property
    def tool_responses(self) -> Sequence[ModelToolResponse]:
        """Return only tool response blocks."""
        return tuple(block for block in self.input if isinstance(block, ModelToolResponse))

    def without_tools(self) -> Self:
        """Return a copy that excludes all tool response blocks."""
        return self.__class__(
            input=tuple(block for block in self.input if not isinstance(block, ModelToolResponse)),
            meta=self.meta,
        )

    def __bool__(self) -> bool:
        return bool(self.input)


ModelOutputSelection = (
    Collection[Literal["text", "image", "audio", "video"]]
    | Literal["auto", "text", "json", "image", "audio", "video"]
    | type[State]
)
"""Requested output modality selection for model generation."""


@final
class ModelReasoningChunk(State, serializable=True):
    """Incremental reasoning fragment produced by providers.

    Attributes
    ----------
    reasoning_part
        Text fragment with model reasoning content.
    meta
        Metadata associated with the fragment.
    """

    @classmethod
    def of(
        cls,
        /,
        reasoning_chunk: TextContent,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a reasoning fragment.

        Parameters
        ----------
        reasoning
            Reasoning text fragment.
        meta
            Optional metadata attached to the fragment.

        Returns
        -------
        Self
            Reasoning fragment instance.
        """
        return cls(
            reasoning_chunk=reasoning_chunk,
            meta=Meta.of(meta),
        )

    reasoning_chunk: TextContent
    meta: Meta = Meta.empty


@final
class ModelReasoning(State, serializable=True):
    """Aggregated reasoning content emitted by a model.

    Attributes
    ----------
    reasoning
        Combined multimodal representation of reasoning parts.
    meta
        Metadata merged across reasoning fragments.
    """

    @classmethod
    def of(
        cls,
        reasoning: Iterable[ModelReasoningChunk] | Multimodal,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create aggregated reasoning from fragments or multimodal input.

        Parameters
        ----------
        reasoning
            Either reasoning parts or already-formed multimodal content.
        meta
            Optional metadata used when ``reasoning`` is multimodal.

        Returns
        -------
        Self
            Aggregated reasoning state.
        """
        if isinstance(reasoning, Multimodal):
            return cls(
                reasoning=MultimodalContent.of(reasoning),
                meta=Meta.of(meta),
            )

        else:
            content_parts: MutableSequence[MultimodalContentPart] = []
            combined_meta: Meta = Meta.empty

            for element in reasoning:
                content_parts.append(element.reasoning_chunk)
                combined_meta = combined_meta.merged_with(element.meta)

            return cls(
                reasoning=MultimodalContent.of(*content_parts),
                meta=combined_meta,
            )

    reasoning: MultimodalContent
    meta: Meta = Meta.empty


ModelOutputBlock = MultimodalContent | ModelReasoning | ModelToolRequest
"""Single output block emitted by model generation."""
ModelOutputBlocks = Sequence[ModelOutputBlock]
"""Ordered collection of model output blocks."""


@final
class ModelOutputLimit(ModelException):
    """Raised when provider output exceeds a configured token budget.

    Parameters
    ----------
    provider
        Provider identifier producing the output.
    model
        Provider model identifier used for generation.
    max_output_tokens
        Maximum output token budget configured for the call.
    content
        Partial output captured before the limit was reached.
    """

    __slots__ = (
        "content",
        "max_output_tokens",
    )

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        max_output_tokens: int,
    ) -> None:
        super().__init__(
            f"Exceeded output limit of {max_output_tokens} tokens"
            f" for {model} model provided by {provider}",
            provider=provider,
            model=model,
        )
        self.max_output_tokens: int = max_output_tokens


@final
class ModelOutput(State, serializable=True):
    """Output payload produced by model generation.

    Attributes
    ----------
    output
        Ordered sequence of content, reasoning, and tool requests.
    meta
        Metadata associated with this output payload.
    """

    @classmethod
    def of(
        cls,
        /,
        *blocks: ModelOutputBlock,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create an output payload from blocks.

        Parameters
        ----------
        *blocks
            Output blocks in stream order.
        meta
            Optional metadata attached to the payload.

        Returns
        -------
        Self
            Output state object.
        """
        return cls(
            output=blocks,
            meta=Meta.of(meta),
        )

    output: ModelOutputBlocks
    meta: Meta = Meta.empty

    @property
    def content(self) -> MultimodalContent:
        """Return only multimodal content blocks."""
        return MultimodalContent.of(
            *(block for block in self.output if isinstance(block, MultimodalContent))
        )

    @property
    def contains_tools(self) -> bool:
        return any(isinstance(block, ModelToolRequest) for block in self.output)

    @property
    def tool_requests(self) -> Sequence[ModelToolRequest]:
        """Return only tool request blocks."""
        return tuple(block for block in self.output if isinstance(block, ModelToolRequest))

    def without_tools(self) -> Self:
        """Return a copy that excludes all tool request blocks."""
        return self.__class__(
            output=tuple(block for block in self.output if not isinstance(block, ModelToolRequest)),
            meta=self.meta,
        )

    def without_reasoning(self) -> Self:
        return self.__class__(
            output=tuple(block for block in self.output if not isinstance(block, ModelReasoning)),
            meta=self.meta,
        )

    def __bool__(self) -> bool:
        return bool(self.output)


ModelContextElement = ModelInput | ModelOutput
"""Single element of generation context history."""
ModelContext = Sequence[ModelContextElement]
"""Ordered model conversation context."""


ModelInputChunk = MultimodalContentPart | ModelToolResponse
"""Single incremental input chunk for realtime sessions."""
ModelOutputChunk = MultimodalContentPart | ModelReasoningChunk | ModelToolRequest
"""Single incremental output chunk emitted by providers."""
ModelOutputStream = AsyncIterable[ModelOutputChunk]
"""Asynchronous stream of model output chunks."""


@runtime_checkable
class ModelGenerating(Protocol):
    """Protocol for async model generation callables."""

    def __call__(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelTools,
        context: ModelContext,
        output: ModelOutputSelection,
        **extra: Any,
    ) -> ModelOutputStream: ...


@final
class ModelSessionEvent(State, serializable=True):
    """Typed control/event message used in realtime model sessions.

    Attributes
    ----------
    event
        Event type identifier.
    content
        Optional event content payload.
    meta
        Metadata attached to the event.
    """

    @classmethod
    def of(
        cls,
        event: str,
        *,
        content: State | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            event=event,
            content=content,
            meta=Meta.of(meta),
        )

    @classmethod
    def turn_started(
        cls,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            event="turn_started",
            content=None,
            meta=Meta.of(meta),
        )

    @classmethod
    def turn_commited(
        cls,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            event="turn_commited",
            content=None,
            meta=Meta.of(meta),
        )

    @classmethod
    def turn_finished(
        cls,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            event="turn_finished",
            content=None,
            meta=Meta.of(meta),
        )

    @classmethod
    def turn_completed(
        cls,
        content: State,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            event="turn_completed",
            content=content,
            meta=Meta.of(meta),
        )

    @classmethod
    def context_updated(
        cls,
        context: ModelContext,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            event="context_updated",
            content=context,
            meta=Meta.of(meta),
        )

    event: (
        Literal[
            "turn_started",
            "turn_commited",
            "turn_finished",
            "turn_completed",
            "context_updated",
        ]
        | str
    )
    content: State | ModelContext | None = None
    meta: Meta = Meta.empty


ModelSessionOutputSelection = (
    Collection[Literal["text", "audio"]] | Literal["auto", "text", "audio"]
)
"""Requested modality selection for realtime session output."""


ModelSessionInputChunk = ModelInputChunk | ModelSessionEvent
"""Single chunk accepted by realtime session write operations."""
ModelSessionInputStream = AsyncIterable[ModelSessionInputChunk]
"""Asynchronous input stream for realtime sessions."""
ModelSessionOutputChunk = ModelOutputChunk | ModelSessionEvent
"""Single chunk returned by realtime session read operations."""
ModelSessionOutputStream = AsyncIterable[ModelSessionOutputChunk]
"""Asynchronous output stream for realtime sessions."""


@runtime_checkable
class ModelSessionReading(Protocol):
    """Protocol describing a single realtime session read operation."""

    async def __call__(self) -> ModelSessionOutputChunk: ...


@runtime_checkable
class ModelSessionWriting(Protocol):
    """Protocol describing a single realtime session write operation."""

    async def __call__(
        self,
        input: ModelSessionInputChunk,  # noqa: A002
    ) -> None: ...


@final
class ModelSession(State):
    """Bound realtime model session with read/write callables."""

    @classmethod
    async def read(cls) -> ModelSessionOutputChunk:
        """Read a single chunk from the active scoped session."""
        return await ctx.state(cls)._reading()

    @classmethod
    async def stream_read(cls) -> ModelSessionOutputStream:
        """Continuously read chunks from the active scoped session."""
        session: Self = ctx.state(cls)
        while True:  # breaks on exception
            yield await session._reading()

    @classmethod
    async def write(
        cls,
        input: ModelSessionInputChunk,  # noqa: A002
    ) -> None:
        """Write a single chunk to the active scoped session."""
        await ctx.state(cls)._writing(input=input)

    @classmethod
    async def stream_write(
        cls,
        input: AsyncIterable[ModelSessionInputChunk],  # noqa: A002
    ) -> None:
        """Write all chunks from an async iterable to the active session."""
        session: Self = ctx.state(cls)
        async for element in input:
            await session._writing(input=element)

    _reading: ModelSessionReading
    _writing: ModelSessionWriting

    def __init__(
        self,
        reading: ModelSessionReading,
        writing: ModelSessionWriting,
    ) -> None:
        super().__init__(
            _reading=reading,
            _writing=writing,
        )


@runtime_checkable
class ModelSessionOpening(Protocol):
    """Protocol for opening a realtime session resource."""

    async def __call__(self) -> ModelSession: ...


@runtime_checkable
class ModelSessionClosing(Protocol):
    """Protocol for closing a realtime session resource."""

    async def __call__(
        self,
        exception: BaseException | None,
    ) -> None: ...


@final
class ModelSessionScope(State):
    """Async context manager wrapping realtime session lifecycle hooks."""

    _opening: ModelSessionOpening
    _closing: ModelSessionClosing

    def __init__(
        self,
        opening: ModelSessionOpening,
        closing: ModelSessionClosing,
    ) -> None:
        super().__init__(
            _opening=opening,
            _closing=closing,
        )

    async def __aenter__(self) -> ModelSession:
        """Open and return a realtime session."""
        return await self._opening()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the realtime session with an optional terminal exception."""
        await self._closing(exception=exc_val)


@runtime_checkable
class ModelSessionPreparing(Protocol):
    """Protocol for preparing scoped realtime model sessions."""

    async def __call__(
        self,
        *,
        instructions: ModelInstructions,
        toolbox: ModelTools,
        context: ModelContext,
        output: ModelSessionOutputSelection,
        **extra: Any,
    ) -> ModelSessionScope: ...
