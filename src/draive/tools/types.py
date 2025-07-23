from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from haiway import Meta

from draive.lmm import LMMToolSpecification
from draive.multimodal import MultimodalContent
from draive.parameters import ParametersSpecification

__all__ = (
    "Tool",
    "ToolAvailabilityChecking",
    "ToolError",
    "ToolErrorFormatting",
    "ToolHandling",
    "ToolResultFormatting",
    "ToolsFetching",
)


type ToolHandling = Literal["auto", "extend", "direct", "spawn"]


class ToolError(Exception):
    def __init__(
        self,
        *args: object,
        content: MultimodalContent,
    ) -> None:
        super().__init__(*args)
        self.content: MultimodalContent = content


@runtime_checkable
class ToolAvailabilityChecking(Protocol):
    def __call__(
        self,
        tools_turn: int,
        meta: Meta,
    ) -> bool: ...


@runtime_checkable
class ToolResultFormatting[Result](Protocol):
    def __call__(
        self,
        result: Result,
    ) -> MultimodalContent: ...


@runtime_checkable
class ToolErrorFormatting(Protocol):
    def __call__(
        self,
        error: Exception,
    ) -> MultimodalContent: ...


@runtime_checkable
class Tool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str | None: ...

    @property
    def parameters(self) -> ParametersSpecification | None: ...

    @property
    def specification(self) -> LMMToolSpecification: ...

    @property
    def meta(self) -> Meta: ...

    @property
    def handling(self) -> ToolHandling: ...

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
class ToolsFetching(Protocol):
    async def __call__(
        self,
        **extra: Any,
    ) -> Sequence[Tool]: ...
