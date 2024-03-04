from json import loads
from typing import Any, final

from draive.scope import ctx
from draive.tools import Tool, ToolException
from draive.tools.state import ToolCallContext
from draive.types import (
    Model,
    StreamingProgressUpdate,
    StringConvertible,
    ToolSpecification,
)

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
        progress: StreamingProgressUpdate[Model] | None = None,
    ) -> StringConvertible:
        if tool := self._tools[name]:
            with ctx.updated(
                ToolCallContext(
                    call_id=call_id,
                    progress=progress or (lambda update: None),
                )
            ):
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
