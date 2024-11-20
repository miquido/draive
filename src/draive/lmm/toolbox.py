from asyncio import gather
from collections.abc import Iterable
from typing import Any, Literal, Self, final

from haiway import ctx, freeze

from draive.lmm.tool import AnyTool
from draive.lmm.types import (
    LMMToolError,
    LMMToolException,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    ToolSpecification,
)
from draive.multimodal import MultimodalContent

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
        repeated_calls_limit: int | None = None,
    ) -> None:
        self._tools: dict[str, AnyTool] = {tool.name: tool for tool in tools}
        self.repeated_calls_limit: int = repeated_calls_limit or 1
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
        repetition_level: int = 0,
    ) -> ToolSpecification | Literal["auto", "required", "none"]:
        if repetition_level >= self.repeated_calls_limit:
            return "none"  # require no tools if reached the limit

        elif repetition_level != 0:
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
            )

        else:
            raise LMMToolException("Requested tool (%s) is not defined", name)

    async def respond_all(
        self,
        requests: LMMToolRequests,
        /,
    ) -> list[LMMToolResponse]:
        return await gather(
            *[self.respond(request) for request in requests.requests],
            return_exceptions=False,  # toolbox calls handle errors if able
        )

    async def respond(
        self,
        request: LMMToolRequest,
        /,
    ) -> LMMToolResponse:
        if tool := self._tools.get(request.tool):
            try:
                return LMMToolResponse(
                    identifier=request.identifier,
                    tool=request.tool,
                    content=await tool._toolbox_call(  # pyright: ignore[reportPrivateUsage]
                        request.identifier,
                        arguments=request.arguments or {},
                    ),
                    direct=tool.requires_direct_result,
                    error=False,
                )

            except LMMToolError as error:  # use formatted error, blow up on other exception
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
