from asyncio import gather
from collections.abc import Callable, Coroutine, Iterable
from typing import Any, Literal, Self, final

from haiway import async_noop, ctx, freeze

from draive.lmm.tools.specification import ToolSpecification
from draive.lmm.tools.status import ToolStatus
from draive.lmm.tools.tool import AnyTool
from draive.lmm.tools.types import ToolError, ToolException
from draive.types import (
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    MultimodalContent,
)

__all__ = [
    "Toolbox",
]


@final
class Toolbox:
    @classmethod
    def of(
        cls,
        tools: Self | Iterable[AnyTool] | None,
        /,
    ) -> Self:
        match tools:
            case None:
                return cls()

            case Toolbox() as tools:
                return tools

            case tools:
                return cls(*tools)

    def __init__(
        self,
        *tools: AnyTool,
        suggest: AnyTool | Literal[True] | None = None,
        recursive_calls_limit: int | None = None,
    ) -> None:
        self._tools: dict[str, AnyTool] = {tool.name: tool for tool in tools}
        self.recursion_limit: int = recursive_calls_limit or 1
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

    def tool_selection(
        self,
        *,
        recursion_level: int = 0,
    ) -> ToolSpecification | Literal["auto", "required", "none"]:
        if recursion_level >= self.recursion_limit:
            return "none"  # require no tools if reached the limit

        elif recursion_level != 0:
            return "auto"  # require tools only for the first call, use auto otherwise

        elif self._suggested_tool is not None and self._suggested_tool.available:
            return self._suggested_tool.specification  # use suggested tool if able

        else:  # use suggestion mode if no specific tool was available
            return "required" if self.suggest_tools else "auto"

    def available_tools(self) -> list[ToolSpecification]:
        return [tool.specification for tool in self._tools.values() if tool.available]

    async def call_tool(
        self,
        name: str,
        /,
        call_id: str,
        arguments: dict[str, Any],
    ) -> MultimodalContent:
        if tool := self._tools.get(name):
            return await tool._toolbox_call(  # pyright: ignore[reportPrivateUsage]
                call_id,
                arguments=arguments,
                report_status=async_noop,
            )

        else:
            raise ToolException("Requested tool (%s) is not defined", name)

    async def respond(
        self,
        requests: LMMToolRequests,
        /,
    ) -> list[LMMToolResponse]:
        return await gather(
            *[self._respond(request) for request in requests.requests],
            return_exceptions=False,  # toolbox calls handle errors if able
        )

    async def _respond(
        self,
        request: LMMToolRequest,
        /,
        report_status: Callable[[ToolStatus], Coroutine[None, None, None]] | None = None,
    ) -> LMMToolResponse:
        if tool := self._tools.get(request.tool):
            try:
                return LMMToolResponse(
                    identifier=request.identifier,
                    tool=request.tool,
                    content=await tool._toolbox_call(  # pyright: ignore[reportPrivateUsage]
                        request.identifier,
                        arguments=request.arguments or {},
                        report_status=report_status or async_noop,
                    ),
                    direct=tool.requires_direct_result,
                    error=False,
                )

            except ToolError as error:  # use formatted error, blow up on other exception
                ctx.log_error(
                    "Tool (%s) returned an error",
                    request.tool,
                    exception=error,
                )
                return LMMToolResponse(
                    identifier=request.identifier,
                    tool=request.tool,
                    content=error.content,
                    direct=False,
                    error=True,
                )

        else:
            # log error and provide fallback result to avoid blowing out the execution
            ctx.log_error("Requested tool (%s) is not defined", request.tool)
            return LMMToolResponse(
                identifier=request.identifier,
                tool=request.tool,
                content=MultimodalContent.of("ERROR"),
                direct=False,
                error=True,
            )
