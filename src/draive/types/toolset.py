from typing import Protocol, runtime_checkable

from draive.types.model import Model
from draive.types.tool import ToolSpecification

__all__ = [
    "Toolset",
]


@runtime_checkable
class Toolset(Protocol):
    @property
    def available_tools(self) -> list[ToolSpecification]:
        ...

    async def call_tool(
        self,
        name: str,
        /,
        call_id: str,
        arguments: str | bytes | None,
    ) -> Model | str:
        ...
