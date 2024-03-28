from json import loads
from typing import Any, final

from draive.tools import Tool
from draive.tools.errors import ToolException
from draive.types import ToolSpecification

__all__ = [
    "Toolbox",
]

AnyTool = Tool[Any, Any]


@final
class Toolbox:
    def __init__(
        self,
        *tools: AnyTool,
    ) -> None:
        self._tools: dict[str, AnyTool] = {tool.name: tool for tool in tools}

    @property
    def available_tools(self) -> list[ToolSpecification]:
        return [tool.specification for tool in self._tools.values() if tool.available]

    async def call_tool(
        self,
        name: str,
        /,
        call_id: str,
        arguments: str | bytes | None,
    ) -> Any:
        if tool := self._tools[name]:
            return await tool(
                tool_call_id=call_id,
                **loads(arguments) if arguments else {},
            )
        else:
            raise ToolException("Requested tool is not defined", name)
