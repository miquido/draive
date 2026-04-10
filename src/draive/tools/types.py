from collections.abc import AsyncIterable, Sequence
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

from haiway import BasicValue, Meta, MetaValues

from draive.models import (
    ModelToolHandling,
    ModelToolParametersSpecification,
    ModelToolSpecification,
)
from draive.multimodal import MultimodalContentPart
from draive.utils import ProcessingEvent

__all__ = (
    "Tool",
    "ToolException",
    "ToolOutputChunk",
    "ToolsLoading",
    "ToolsSuggesting",
)


class ToolException(Exception):
    """Base exception raised by tool execution helpers."""

    __slots__ = ("meta", "tool")

    def __init__(
        self,
        *args: object,
        tool: str,
        meta: Meta | MetaValues | None = None,
    ) -> None:
        super().__init__(*args)
        self.tool: str = tool
        self.meta: Meta = Meta.of(meta)


ToolOutputChunk = MultimodalContentPart | ProcessingEvent


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
