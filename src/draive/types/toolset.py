from typing import Protocol

from draive.types.string import StringConvertible
from draive.types.tool import ToolSpecification

__all__ = [
    "Toolset",
]


class Toolset(Protocol):
    @property
    def available_tools(self) -> list[ToolSpecification]:
        ...

    async def call_tool(
        self,
        name: str,
        *,
        arguments: str | bytes | None,
    ) -> StringConvertible:
        ...
