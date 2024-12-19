from asyncio import gather
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Self, final

from haiway import State, ctx

from draive.lmm.tool import AnyTool
from draive.lmm.types import (
    LMMToolError,
    LMMToolException,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    LMMToolSpecification,
)
from draive.multimodal import MultimodalContent

__all__ = [
    "Toolbox",
]


@final
class Toolbox(State):
    @classmethod
    def of(
        cls,
        *tools: AnyTool,
        suggest: AnyTool | Literal[True] | None = None,
        repeated_calls_limit: int | None = None,
    ) -> Self:
        match suggest:
            case None:
                return cls(
                    tools={tool.name: tool for tool in tools},
                    suggest_tool=False,
                    repeated_calls_limit=repeated_calls_limit or 1,
                )

            case True:
                return cls(
                    tools={tool.name: tool for tool in tools},
                    suggest_tool=True,
                    repeated_calls_limit=repeated_calls_limit or 1,
                )

            case tool:
                return cls(
                    tools={**{tool.name: tool for tool in tools}, tool.name: tool},
                    suggest_tool=tool,
                    repeated_calls_limit=repeated_calls_limit or 1,
                )

    @classmethod
    def out_of(
        cls,
        tools: Self | Iterable[AnyTool] | None,
        /,
    ) -> Self:
        match tools:
            case None:
                return cls(
                    tools={},
                    suggest_tool=False,
                    repeated_calls_limit=1,
                )

            case Toolbox() as toolbox:
                return toolbox

            case tools:
                return cls(
                    tools={tool.name: tool for tool in tools},
                    suggest_tool=False,
                    repeated_calls_limit=1,
                )

    tools: Mapping[str, AnyTool]
    suggest_tool: AnyTool | bool
    repeated_calls_limit: int

    def tool_selection(
        self,
        *,
        repetition_level: int = 0,
    ) -> LMMToolSpecification | Literal["auto", "required", "none"]:
        if repetition_level >= self.repeated_calls_limit:
            return "none"  # require no tools if reached the limit

        elif repetition_level != 0:
            return "auto"  # require tools only for the first call, use auto otherwise

        elif self.suggest_tool is False:
            return "auto"

        elif self.suggest_tool is True:
            return "required"

        elif self.suggest_tool.available:
            return self.suggest_tool.specification  # use suggested tool if able

        else:
            return "auto"

    def available_tools(self) -> list[LMMToolSpecification]:
        return [tool.specification for tool in self.tools.values() if tool.available]

    async def call_tool(
        self,
        name: str,
        /,
        call_id: str,
        arguments: dict[str, Any],
    ) -> MultimodalContent:
        if tool := self.tools.get(name):
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
        if tool := self.tools.get(request.tool):
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
