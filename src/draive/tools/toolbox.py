from json import loads
from typing import Any, Literal, final

from draive.helpers import freeze
from draive.parameters import ToolSpecification
from draive.tools import Tool
from draive.tools.errors import ToolException

__all__ = [
    "Toolbox",
]

AnyTool = Tool[Any, Any]


@final
class Toolbox:
    def __init__(
        self,
        *tools: AnyTool,
        suggest: AnyTool | Literal[True] | None = None,
    ) -> None:
        self._tools: dict[str, AnyTool] = {tool.name: tool for tool in tools}
        self.suggest_tools: bool
        self._suggested_tool: AnyTool | None
        match suggest:
            case None:
                self.suggest_tools = False
                self._suggested_tool = None
            case True:
                self.suggest_tools = True if self._tools else False
                self._suggested_tool = None
            case tool:
                self.suggest_tools = True
                self._suggested_tool = tool
                self._tools[tool.name] = tool

        freeze(self)

    @property
    def suggested_tool_name(self) -> str | None:
        if self._suggested_tool is not None and self._suggested_tool.available:
            return self._suggested_tool.name
        else:
            return None

    @property
    def suggested_tool(self) -> ToolSpecification | None:
        if self._suggested_tool is not None and self._suggested_tool.available:
            return self._suggested_tool.specification
        else:
            return None

    @property
    def available_tools(self) -> list[ToolSpecification]:
        return [tool.specification for tool in self._tools.values() if tool.available]

    def requires_direct_result(
        self,
        tool_name: str,
    ) -> bool:
        if tool := self._tools.get(tool_name):
            return tool.requires_direct_result
        else:
            return False

    async def call_tool(
        self,
        name: str,
        /,
        call_id: str,
        arguments: dict[str, Any] | str | bytes | None,
    ) -> Any:
        if tool := self._tools.get(name):
            return await tool(
                tool_call_id=call_id,
                **loads(arguments) if isinstance(arguments, str | bytes) else arguments or {},
            )
        else:
            raise ToolException("Requested tool is not defined", name)
