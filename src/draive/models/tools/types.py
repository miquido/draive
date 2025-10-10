from collections.abc import Sequence
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

from draive.models.types import (
    ModelToolHandling,
    ModelToolParametersSpecification,
    ModelToolSpecification,
)
from draive.multimodal import MultimodalContent

__all__ = (
    "Tool",
    "ToolAvailabilityChecking",
    "ToolError",
    "ToolErrorFormatting",
    "ToolResultFormatting",
    "ToolsLoading",
    "ToolsSuggesting",
)


class ToolError(Exception):
    """Tool execution error carrying formatted content.

    Attributes
    ----------
    content : MultimodalContent
        Pre-formatted content describing the error to be surfaced to the model.
    """

    def __init__(
        self,
        *args: object,
        content: MultimodalContent,
    ) -> None:
        super().__init__(*args)
        self.content: MultimodalContent = content


@runtime_checkable
class ToolAvailabilityChecking(Protocol):
    """Callable that decides if a tool is available for a given turn."""

    def __call__(
        self,
        tools_turn: int,
        specification: ModelToolSpecification,
    ) -> bool: ...


@runtime_checkable
class ToolResultFormatting[Result](Protocol):
    """Callable that converts a tool result to ``MultimodalContent``."""

    def __call__(
        self,
        result: Result,
    ) -> MultimodalContent: ...


@runtime_checkable
class ToolErrorFormatting(Protocol):
    """Callable that converts an exception to fallback ``MultimodalContent``."""

    def __call__(
        self,
        error: Exception,
    ) -> MultimodalContent: ...


@runtime_checkable
class Tool(Protocol):
    """Public interface every tool must implement."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str | None: ...

    @property
    def parameters(self) -> ModelToolParametersSpecification | None: ...

    @property
    def specification(self) -> ModelToolSpecification: ...

    @property
    def handling(self) -> ModelToolHandling: ...

    def available(
        self,
        tools_turn: int,
    ) -> bool: ...

    async def call(
        self,
        call_id: str,
        /,
        **arguments: Any,
    ) -> MultimodalContent: ...


@runtime_checkable
class ToolsSuggesting(Protocol):
    """Callable that suggests a single tool or policy for a loop turn."""

    def __call__(
        self,
        tools_turn: int,
        tools: Sequence[ModelToolSpecification],
    ) -> ModelToolSpecification | bool: ...


@runtime_checkable
class ToolsLoading(Protocol):
    """Callable that loads a sequence of tools, possibly using extra kwargs."""

    async def __call__(
        self,
        **extra: Any,
    ) -> Sequence[Tool]: ...
