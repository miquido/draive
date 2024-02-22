from json import loads
from typing import Any, final

from draive.tools import Tool, ToolException
from draive.types import StringConvertible, ToolSpecification

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
        *,
        arguments: str | bytes | None,
    ) -> StringConvertible:
        if tool := self._tools[name]:
            return await tool(
                **self._validated_call_arguments(
                    tool,
                    **loads(arguments) if arguments else {},
                ),
            )

        else:
            raise ToolException("Requested tool is not defined", name)

    def _validated_call_arguments(
        self,
        tool: AnyTool,
        /,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # TODO: validate call arguments
        # those are decoded from json and require type validation
        # may also require name changes
        return kwargs
