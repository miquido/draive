from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from draive.commons import Meta
from draive.lmm import LMMToolSpecification
from draive.multimodal import Multimodal, MultimodalContent
from draive.parameters import ParametersSpecification

__all__ = (
    "Tool",
    "ToolAvailabilityChecking",
    "ToolErrorFormatting",
    "ToolException",
    "ToolHandling",
    "ToolResultFormatting",
    "ToolsFetching",
)


ToolHandling = Literal["auto", "direct"]


class ToolException(Exception):
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
        meta: Meta,
    ) -> bool: ...


@runtime_checkable
class ToolResultFormatting[Result](Protocol):
    def __call__(
        self,
        result: Result,
    ) -> Multimodal: ...


@runtime_checkable
class ToolErrorFormatting[Result](Protocol):
    def __call__(
        self,
        error: Exception,
    ) -> Multimodal: ...


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
    def available(self) -> bool: ...

    @property
    def handling(self) -> ToolHandling: ...

    async def tool_call(
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
