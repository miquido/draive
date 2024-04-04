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
        suggested: AnyTool | None = None,
    ) -> None:
        self._suggested_tool: AnyTool | None = suggested
        self._tools: dict[str, AnyTool] = {tool.name: tool for tool in tools}
        if suggested := suggested:
            self._tools[suggested.name] = suggested

    @property
    def suggested_tool_name(self) -> str | None:
        if self._suggested_tool is None or not self._suggested_tool.available:
            return None
        elif self._suggested_tool.available:
            return self._suggested_tool.name

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
