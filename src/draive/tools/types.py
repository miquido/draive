from collections.abc import AsyncIterable, Sequence
from typing import (
    Any,
    Protocol,
    Self,
    runtime_checkable,
)

from haiway import BasicValue, Meta, MetaValues, State, ctx

from draive.models import (
    ModelToolHandling,
    ModelToolParametersSpecification,
    ModelToolRequest,
    ModelToolSpecification,
)
from draive.multimodal import Multimodal, MultimodalContent, MultimodalContentPart

__all__ = (
    "Tool",
    "ToolEvent",
    "ToolException",
    "ToolOutputChunk",
    "ToolsLoading",
    "ToolsSuggesting",
)


class ToolException(Exception):
    """Base exception raised by tool execution helpers."""


_placeholder_request: ModelToolRequest = ModelToolRequest(
    identifier="",
    tool="",
    arguments={},
)


class ToolEvent(State, serializable=True):
    """Structured event emitted while a tool is running."""

    @classmethod
    def of(
        cls,
        event: str,
        /,
        content: Multimodal = MultimodalContent.empty,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a tool event bound to the current tool request context.

        Parameters
        ----------
        event : str
            Event name describing the emitted state transition or progress update.
        content : Multimodal, default=MultimodalContent.empty
            Event payload converted to multimodal content.
        meta : Meta | MetaValues | None, default=None
            Additional metadata merged with the current request identifier and tool
            name when available in context.

        Returns
        -------
        Self
            Event instance ready to be streamed alongside tool output.
        """
        request: ModelToolRequest = ctx.state(
            ModelToolRequest,
            default=_placeholder_request,
        )
        return cls(
            event=event,
            content=MultimodalContent.of(content),
            meta=Meta.of(meta).merged_with(
                {
                    "tool": request.tool,
                    "identifier": request.identifier,
                }
            )
            if request is not _placeholder_request
            else Meta.of(meta),
        )

    event: str
    content: MultimodalContent
    meta: Meta = Meta.empty


ToolOutputChunk = MultimodalContentPart | ToolEvent


@runtime_checkable
class Tool(Protocol):
    """Runtime contract implemented by all tool adapters."""

    @property
    def name(self) -> str:
        """Tool identifier used by model requests and toolbox lookup."""
        ...

    @property
    def description(self) -> str | None:
        """Optional human-readable behavior description."""
        ...

    @property
    def parameters(self) -> ModelToolParametersSpecification | None:
        """Parameter schema accepted by this tool, when declared."""
        ...

    @property
    def specification(self) -> ModelToolSpecification:
        """Complete model-facing specification for this tool."""
        ...

    @property
    def handling(self) -> ModelToolHandling:
        """Handling mode defining how results are surfaced."""
        ...

    def call(
        self,
        **arguments: BasicValue,
    ) -> AsyncIterable[ToolOutputChunk]:
        """Execute the tool with basic-value arguments.

        Parameters
        ----------
        **arguments : BasicValue
            Tool arguments matching the declared parameter schema.

        Returns
        -------
        AsyncIterable[ToolOutputChunk]
            Stream of tool output parts and optional tool events.
        """
        ...


@runtime_checkable
class ToolsSuggesting(Protocol):
    """Strategy selecting required tool usage for a model call."""

    def __call__(
        self,
        tools: Sequence[ModelToolSpecification],
        iteration: int,
    ) -> ModelToolSpecification | bool:
        """Suggest required tool usage for current iteration.

        Parameters
        ----------
        iteration : int
            Current execution or conversation iteration.
        tools : Sequence[ModelToolSpecification]
            Currently available tool specifications.

        Returns
        -------
        ModelToolSpecification | bool
            ``False`` to use automatic selection, ``True`` to require any tool,
            or a concrete specification to require a specific tool.
        """
        ...


@runtime_checkable
class ToolsLoading(Protocol):
    """Async provider returning tools for runtime context."""

    async def __call__(
        self,
        **extra: Any,
    ) -> Sequence[Tool]:
        """Load tool instances.

        Parameters
        ----------
        **extra : Any
            Loader-specific contextual arguments.

        Returns
        -------
        Sequence[Tool]
            Loaded tools available for execution.
        """
        ...
