from asyncio import sleep
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    Collection,
    Coroutine,
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
    final,
    overload,
    runtime_checkable,
)

from haiway import META_EMPTY, Meta, MetaValues, State, ctx

from draive.multimodal import ArtifactContent, Multimodal, MultimodalContent, MultimodalContentPart
from draive.parameters import DataModel, ToolParametersSpecification
from draive.utils import Memory

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
    "ModelMemory",
    "ModelMemoryRecall",
    "ModelOutput",
    "ModelOutputBlock",
    "ModelOutputBlocks",
    "ModelOutputChunk",
    "ModelOutputFailed",
    "ModelOutputInvalid",
    "ModelOutputLimit",
    "ModelOutputSelection",
    "ModelRateLimit",
    "ModelReasoning",
    "ModelSession",
    "ModelSessionClosing",
    "ModelSessionEvent",
    "ModelSessionInput",
    "ModelSessionOpening",
    "ModelSessionOutput",
    "ModelSessionOutputSelection",
    "ModelSessionPreparing",
    "ModelSessionReading",
    "ModelSessionScope",
    "ModelSessionWriting",
    "ModelStreamOutput",
    "ModelToolFunctionSpecification",
    "ModelToolHandling",
    "ModelToolRequest",
    "ModelToolResponse",
    "ModelToolSpecification",
    "ModelToolsDeclaration",
    "ModelToolsSelection",
)

ModelInstructions = str
"""System or task instructions provided to guide model behavior.

Currently represented as a string, but may be extended to support structured
instruction objects in the future depending on provider capabilities.
"""


class ModelException(Exception):
    """Base exception for model-related errors.

    Carries the provider and model identifiers for better diagnostics in higher
    layers and logs.

    Attributes
    ----------
    provider : str
        Name of the provider that produced the error.
    model : str
        Name of the model that produced the error.
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


class ModelInputInvalid(ModelException):
    """Raised when prepared input is invalid for a given provider/model."""

    __slots__ = ()

    def __init__(
        self,
        *,
        provider: str,
        model: str,
    ) -> None:
        super().__init__(
            f"Invalid input prepared for {model} model provided by {provider}",
            provider=provider,
            model=model,
        )


class ModelOutputInvalid(ModelException):
    """Raised when a provider/model produces an invalid output.

    Attributes
    ----------
    reason : str
        Human-readable reason describing why the output is considered invalid.
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
            f"Invalid output produced by {model} model provided by {provider}, reason: {reason}",
            provider=provider,
            model=model,
        )
        self.reason: str = reason


@final
class ModelRateLimit(ModelException):
    """Raised when a provider enforces a rate limit for a model.

    Attributes
    ----------
    retry_after : float
        Number of seconds to wait before retrying.
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
            f"Rate limit for {model} model provided by {provider}, retry after {retry_after:.2f}s",
            provider=provider,
            model=model,
        )
        self.retry_after: float = retry_after

    async def wait(self) -> None:
        """Sleep for the recommended ``retry_after`` interval.

        Useful shorthand to respect provider throttling without duplicating
        sleep logic at call sites.
        """
        await sleep(self.retry_after)


class ModelOutputFailed(ModelException):
    """Raised when provider signals a generation failure.

    Attributes
    ----------
    reason : str
        Human-readable reason describing the failure.
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
            f"Output for {model} model provided by {provider} failed, reason: {reason}",
            provider=provider,
            model=model,
        )
        self.reason: str = reason


@final
class ModelToolFunctionSpecification(TypedDict, total=True):
    """JSON Schema-like function specification exposed to the model.

    Keys mirror OpenAI/JSON Schema conventions for function tools.
    """

    name: str
    description: str | None
    parameters: ToolParametersSpecification | None
    additionalProperties: Literal[False]


ModelToolSpecification = ModelToolFunctionSpecification
"""Type alias for tool specifications exposed to models."""

ModelToolsSelection = Literal["auto", "required", "none"] | str
"""Selection strategy for tool use during generation.

- "auto": Let the model decide when to use tools
- "required": Force the model to use at least one tool
- "none": Disable tool use for this turn
- Provider-specific strings for custom selection modes
"""


@final
class ModelToolsDeclaration(State):
    """Declares tools available to a generation turn.

    Use ``ModelToolsDeclaration.of(...)`` to build an instance. ``none`` is a
    predefined constant indicating that no tools are available.

    Attributes
    ----------
    specifications : Sequence[ModelToolSpecification]
        Function tool specifications exposed to the model.
    selection : ModelToolsSelection
        Selection strategy hint (e.g., ``"auto"``, ``"required"``, ``"none"`` or provider-specific).
    """

    none: ClassVar[Self]  # defined after the class

    @classmethod
    def of(
        cls,
        *specifications: ModelToolSpecification,
        selection: ModelToolsSelection = "auto",
    ) -> Self:
        """Build a tools declaration.

        Parameters
        ----------
        specifications : Sequence[ModelToolSpecification]
            Tool function specifications to expose.
        selection : ModelToolsSelection, optional
            Tool selection mode hint. Defaults to ``"auto"``.

        Returns
        -------
        ModelToolsDeclaration
            The constructed declaration.
        """
        return cls(
            specifications=specifications,
            selection=selection,
        )

    specifications: Sequence[ModelToolSpecification]
    selection: ModelToolsSelection

    def __bool__(self) -> bool:
        """Return ``True`` when any tool specifications are present."""
        return bool(self.specifications)


ModelToolsDeclaration.none = ModelToolsDeclaration(
    selection="none",
    specifications=(),
)


@final
class ModelToolRequest(State):
    """A request emitted by the model to call a tool function."""

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
        """Create a tool request.

        Parameters
        ----------
        identifier : str
            Correlation identifier to match responses to requests.
        tool : str
            Name of the tool/function to invoke.
        arguments : Mapping[str, Any], optional
            JSON-serializable arguments for the tool.
        meta : Meta | MetaValues | None, optional
            Additional metadata.

        Returns
        -------
        ModelToolRequest
            Constructed tool request.
        """
        return cls(
            identifier=identifier,
            tool=tool,
            arguments=arguments if arguments is not None else {},
            meta=Meta.of(meta),
        )

    identifier: str
    tool: str
    arguments: Mapping[str, Any]
    meta: Meta


ModelToolHandling = Literal["response", "error", "output", "output_extension", "detached"]
"""Defines how tool responses should be handled in the conversation flow.

- "response": Include the response in the conversation context for the next turn
- "error": Treat the response as an error and include it in context
- "output": Use this response as the final output, ending the conversation loop
- "output_extension": Include content in final output while continuing the loop
- "detached": Handle the response outside the normal conversation flow
"""


@final
class ModelToolResponse(State):
    """A response produced by a tool invocation.

    Content may be included in the model conversation depending on ``handling``.
    """

    @classmethod
    def of(
        cls,
        identifier: str,
        /,
        *,
        tool: str,
        content: MultimodalContent,
        handling: ModelToolHandling = "response",
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a tool response.

        Parameters
        ----------
        identifier : str
            Correlation identifier that matches a previous request.
        tool : str
            Tool/function name that produced the response.
        content : MultimodalContent
            Multimodal content produced by the tool.
        handling : ModelToolHandling, optional
            How the response should be used (``"response"``, ``"output"``, etc.).
        meta : Meta | MetaValues | None, optional
            Additional metadata.

        Returns
        -------
        ModelToolResponse
            Constructed tool response.
        """
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
    handling: ModelToolHandling
    meta: Meta


ModelInputBlock = MultimodalContent | ModelToolResponse
"""Content blocks that can be included in model input."""

ModelInputBlocks = Sequence[ModelInputBlock]
"""Ordered sequence of input blocks for a model turn."""


@final
class ModelInput(State):
    """Structured input to a model turn.

    Combines multimodal content and tool responses that become part of the
    conversation context.

    Attributes
    ----------
    blocks : ModelInputBlocks
        Ordered sequence of input blocks.
    meta : Meta
        Additional metadata.
    """

    @classmethod
    def of(
        cls,
        /,
        *blocks: ModelInputBlock,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a model input from blocks.

        Parameters
        ----------
        blocks : ModelInputBlock
            Multimodal content and/or tool responses.
        meta : Meta | MetaValues | None, optional
            Additional metadata.

        Returns
        -------
        ModelInput
            Constructed model input.
        """
        return cls(
            blocks=blocks,
            meta=Meta.of(meta),
        )

    blocks: ModelInputBlocks
    meta: Meta

    @property
    def content(self) -> MultimodalContent:
        """Return only the multimodal content blocks merged into one."""
        return MultimodalContent.of(
            *(block for block in self.blocks if isinstance(block, MultimodalContent))
        )

    @property
    def tools(self) -> Sequence[ModelToolResponse]:
        """Return only the tool responses included in the input."""
        return [block for block in self.blocks if isinstance(block, ModelToolResponse)]

    def __bool__(self) -> bool:
        """Return ``True`` when any input blocks are present."""
        return bool(self.blocks)


ModelOutputSelection = (
    Collection[Literal["text", "image", "audio", "video"]]
    | Literal["auto", "text", "json", "image", "audio", "video"]
    | type[DataModel]
)


@final
class ModelReasoning(State):
    """Represents model reasoning or chain-of-thought content.

    This class encapsulates reasoning steps, thoughts, or explanations generated
    by a model before producing the final output. It's distinct from regular
    multimodal content as it represents the model's internal thought process.

    Attributes
    ----------
    content : MultimodalContent
        The reasoning content produced by the model.
    meta : Meta
        Additional metadata associated with the reasoning.
    """

    @classmethod
    def of(
        cls,
        content: Multimodal,
        /,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a reasoning block from multimodal content.

        Parameters
        ----------
        content : Multimodal
            The reasoning content to encapsulate.
        meta : Meta | MetaValues | None, optional
            Additional metadata for the reasoning block.

        Returns
        -------
        ModelReasoning
            Constructed reasoning block.
        """
        return cls(
            content=MultimodalContent.of(content),
            meta=Meta.of(meta),
        )

    content: MultimodalContent
    meta: Meta


ModelOutputBlock = MultimodalContent | ModelReasoning | ModelToolRequest
"""Content blocks that can be included in model output."""

ModelOutputBlocks = Sequence[ModelOutputBlock]
"""Ordered sequence of output blocks from a model turn."""


@final
class ModelOutput(State):
    """Structured output of a model turn.

    Contains multimodal content blocks and potential tool requests.

    Attributes
    ----------
    blocks : ModelOutputBlocks
        Ordered sequence of output blocks produced by the model.
    meta : Meta
        Additional metadata.
    """

    @classmethod
    def of(
        cls,
        /,
        *blocks: ModelOutputBlock,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a model output from blocks.

        Parameters
        ----------
        blocks : ModelOutputBlock
            Multimodal content and/or tool requests.

        meta : Meta | MetaValues | None, optional
            Additional metadata.

        Returns
        -------
        ModelOutput
            Constructed model output.
        """
        return cls(
            blocks=blocks,
            meta=Meta.of(meta),
        )

    blocks: ModelOutputBlocks
    meta: Meta

    @property
    def content(self) -> MultimodalContent:
        """Return only the multimodal content blocks merged into one."""
        return MultimodalContent.of(
            *(block for block in self.blocks if isinstance(block, MultimodalContent))
        )

    @property
    def content_with_reasoning(self) -> MultimodalContent:
        """\
        Return the multimodal content blocks merged into one and including reasoning as MetaContent
        """
        parts: list[MultimodalContentPart] = []
        for block in self.blocks:
            if isinstance(block, MultimodalContent):
                parts.extend(block.parts)

            elif isinstance(block, ModelReasoning):
                parts.append(
                    ArtifactContent.of(
                        block,
                        category="reasoning",
                        hidden=True,
                    )
                )

        return MultimodalContent.of(*parts)

    @property
    def reasoning(self) -> Sequence[ModelReasoning]:
        """Return only the reasoning included in the output."""
        return [block for block in self.blocks if isinstance(block, ModelReasoning)]

    @property
    def tools(self) -> Sequence[ModelToolRequest]:
        """Return only the tool requests included in the output."""
        return [block for block in self.blocks if isinstance(block, ModelToolRequest)]

    def __bool__(self) -> bool:
        """Return ``True`` when any output blocks are present."""
        return bool(self.blocks)


ModelStreamOutput = MultimodalContentPart | ModelReasoning | ModelToolRequest
"""Individual chunks that can be streamed as model output."""


@final
class ModelOutputLimit(ModelException):
    """Raised when the model exceeds its configured output token limit.

    Attributes
    ----------
    max_output_tokens : int
        Maximum number of output tokens that was exceeded.
    content : ModelOutputBlocks
        Partial content produced before the limit was hit.
    """

    __slots__ = ("content", "max_output")

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        max_output_tokens: int,
        content: ModelOutputBlocks,
    ) -> None:
        super().__init__(
            f"Exceeded output limit of {max_output_tokens} tokens"
            f" for {model} model provided by {provider}",
            provider=provider,
            model=model,
        )
        self.max_output_tokens: int = max_output_tokens
        self.content: ModelOutputBlocks = content


ModelContextElement = ModelInput | ModelOutput
"""Elements that can be included in conversation context."""

ModelContext = Sequence[ModelContextElement]
"""Conversation context as a sequence of inputs and outputs."""


@final
class ModelMemoryRecall(State):
    """Immutable snapshot of recalled memory for a session/turn.

    Attributes
    ----------
    context : ModelContext
        Recalled context elements (inputs/outputs) to include.
    variables : Mapping[str, str]
        Key-value variables associated with the recall.
    meta : Meta
        Additional metadata.
    """

    empty: ClassVar[Self]  # defined after the class

    @classmethod
    def of(
        cls,
        element: ModelContextElement,
        *elements: ModelContextElement,
        variables: Mapping[str, str] | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a recall from one or more context elements.

        Parameters
        ----------
        element, elements : ModelContextElement
            Context elements (inputs/outputs) to include in the recall.
        variables : Mapping[str, str] | None, optional
            Variables associated with the recall.
        meta : Meta | MetaValues | None, optional
            Additional metadata.

        Returns
        -------
        ModelMemoryRecall
            Constructed memory recall snapshot.
        """
        return cls(
            context=(element, *elements),
            variables=variables if variables is not None else {},
            meta=Meta.of(meta),
        )

    context: ModelContext
    variables: Mapping[str, str]
    meta: Meta


ModelMemoryRecall.empty = ModelMemoryRecall(
    context=(),
    variables={},
    meta=META_EMPTY,
)

ModelMemory = Memory[ModelMemoryRecall, ModelContextElement]
"""Memory abstraction for managing conversation context and variables."""


@final
class ModelInputChunk(State):
    """Streaming input chunk for realtime sessions.

    Attributes
    ----------
    content : MultimodalContent
        Incremental multimodal content payload.
    eod : bool
        End-of-data marker for the current input stream.
    meta : Meta
        Additional metadata.
    """

    @classmethod
    def of(
        cls,
        *content: Multimodal,
        eod: bool = False,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a streaming input chunk.

        Parameters
        ----------
        content : Multimodal
            One or more multimodal elements to include.
        eod : bool, optional
            Marks the end of the input stream when ``True``.
        meta : Meta | MetaValues | None, optional
            Additional metadata.

        Returns
        -------
        ModelInputChunk
            Constructed input chunk.
        """
        return cls(
            content=MultimodalContent.of(*content),
            eod=eod,
            meta=Meta.of(meta),
        )

    content: MultimodalContent
    eod: bool
    meta: Meta

    def __bool__(self) -> bool:
        """Return ``True`` when any content is present."""
        return bool(self.content)


@final
class ModelOutputChunk(State):
    """Streaming output chunk produced by a model.

    Attributes
    ----------
    content : MultimodalContent
        Incremental multimodal content payload.
    eod : bool
        End-of-data marker for the current output stream.
    meta : Meta
        Additional metadata.
    """

    @classmethod
    def of(
        cls,
        /,
        *content: Multimodal,
        eod: bool = False,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a streaming output chunk.

        Parameters
        ----------
        content : Multimodal
            One or more multimodal elements to include.
        eod : bool, optional
            Marks the end of the output stream when ``True``.
        meta : Meta | MetaValues | None, optional
            Additional metadata.

        Returns
        -------
        ModelOutputChunk
            Constructed output chunk.
        """
        return cls(
            content=MultimodalContent.of(*content),
            eod=eod,
            meta=Meta.of(meta),
        )

    content: MultimodalContent
    eod: bool
    meta: Meta

    def __bool__(self) -> bool:
        """Return ``True`` when any content is present."""
        return bool(self.content)


@runtime_checkable
class ModelGenerating(Protocol):
    """Provider interface for a single-turn generation call.

    This protocol defines the contract that provider implementations must follow
    for performing model generation. Implementations must handle both streaming
    and non-streaming modes and support tool use through the tools parameter.

    The protocol supports overloaded signatures to provide proper type hints
    based on the stream parameter value.
    """

    @overload
    def __call__(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        context: ModelContext,
        output: ModelOutputSelection,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> Coroutine[None, None, ModelOutput]: ...

    @overload
    def __call__(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        context: ModelContext,
        output: ModelOutputSelection,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput]: ...

    def __call__(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        context: ModelContext,
        output: ModelOutputSelection,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput] | Coroutine[None, None, ModelOutput]: ...


@final
class ModelSessionEvent(State):
    """Session event used in realtime flows (e.g., status, typing)."""

    @classmethod
    def of(
        cls,
        category: str,
        /,
        content: Multimodal = MultimodalContent.empty,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a session event.

        Parameters
        ----------
        category : str
            Event category or type.
        content : Multimodal, optional
            Optional multimodal payload carried with the event.
        meta : Meta | MetaValues | None, optional
            Additional metadata.

        Returns
        -------
        ModelSessionEvent
            Constructed session event.
        """
        return cls(
            category=category,
            content=MultimodalContent.of(content),
            meta=Meta.of(meta),
        )

    category: str
    content: MultimodalContent
    meta: Meta


ModelSessionInput = ModelInputChunk | ModelToolResponse | ModelSessionEvent
"""Input types that can be sent to a realtime model session."""

ModelSessionOutput = ModelOutputChunk | ModelToolRequest | ModelSessionEvent
"""Output types that can be received from a realtime model session."""

ModelSessionOutputSelection = (
    Collection[Literal["text", "audio"]] | Literal["auto", "text", "audio"]
)
"""Output selection policy for realtime session events.

Limited to text and audio modalities for realtime use cases.
"""


@runtime_checkable
class ModelSessionReading(Protocol):
    """Callable that reads the next session output event."""

    async def __call__(self) -> ModelSessionOutput: ...


@runtime_checkable
class ModelSessionWriting(Protocol):
    """Callable that writes an input event to the session."""

    async def __call__(
        self,
        input: ModelSessionInput,  # noqa: A002
    ) -> None: ...


@final
class ModelSession(State):
    """Static helpers for interacting with the active realtime session state.

    Delegates to the provider-implemented ``reading``/``writing`` callables.
    """

    @classmethod
    async def read(cls) -> ModelSessionOutput:
        """Read a single ``ModelSessionOutput`` event from the session."""
        return await ctx.state(cls).reading()

    @classmethod
    async def reader(cls) -> AsyncIterable[ModelSessionOutput]:
        """Async iterator that continuously reads session outputs until error."""
        session: Self = ctx.state(cls)
        while True:  # breaks on exception
            yield await session.reading()

    @classmethod
    async def write(
        cls,
        input: ModelSessionInput,  # noqa: A002
    ) -> None:
        """Write a single input event into the session."""
        await ctx.state(cls).writing(input=input)

    @classmethod
    async def writer(
        cls,
        input: AsyncIterable[ModelSessionInput],  # noqa: A002
    ) -> None:
        """Consume an async iterable and write each element into the session."""
        session: Self = ctx.state(cls)
        async for element in input:
            await session.writing(input=element)

    reading: ModelSessionReading
    writing: ModelSessionWriting


@runtime_checkable
class ModelSessionOpening(Protocol):
    """Callable that opens a realtime session and returns ``ModelSession``."""

    async def __call__(self) -> ModelSession: ...


@runtime_checkable
class ModelSessionClosing(Protocol):
    """Callable that closes a realtime session, optionally receiving an exception."""

    async def __call__(
        self,
        exception: BaseException | None,
    ) -> None: ...


@final
class ModelSessionScope(State):
    """Async context manager that opens and closes a ``ModelSession``.

    Attributes
    ----------
    opening : ModelSessionOpening
        Callable that opens the session.
    closing : ModelSessionClosing
        Callable that closes the session.
    """

    opening: ModelSessionOpening
    closing: ModelSessionClosing

    async def __aenter__(self) -> ModelSession:
        """Open and return the underlying ``ModelSession``."""
        return await self.opening()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the session, forwarding any exception to the ``closing`` hook."""
        await self.closing(exception=exc_val)


@runtime_checkable
class ModelSessionPreparing(Protocol):
    """Provider interface for preparing realtime model sessions.

    This protocol defines how providers should implement session preparation
    for realtime, bidirectional communication with models. The implementation
    should return a ``ModelSessionScope`` that can be used as an async context
    manager for session lifecycle management.

    Parameters
    ----------
    instructions : ModelInstructions
        System/task instructions for the session.
    tools : ModelToolsDeclaration
        Tools available throughout the session.
    memory : ModelMemory
        Initial memory context for the session.
    output : ModelSessionOutputSelection
        Desired output selection policy for session events.
    **extra : Any
        Provider-specific configuration parameters.

    Returns
    -------
    ModelSessionScope
        Prepared session scope ready for realtime interaction.
    """

    async def __call__(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        memory: ModelMemory,
        output: ModelSessionOutputSelection,
        **extra: Any,
    ) -> ModelSessionScope: ...
