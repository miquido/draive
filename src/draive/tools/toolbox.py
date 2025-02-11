from asyncio import gather
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Protocol, Self, cast, final, runtime_checkable

from haiway import State, ctx

from draive.lmm.types import (
    LMMException,
    LMMToolError,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    LMMToolSpecification,
)
from draive.multimodal import MultimodalContent
from draive.tools.tool import AnyTool

__all__ = [
    "ExternalToolbox",
    "Toolbox",
    "ToolboxFetching",
]


@final
class Toolbox(State):
    @classmethod
    def of(
        cls,
        *tools: AnyTool,
        suggest: AnyTool | bool | None = None,
        repeated_calls_limit: int | None = None,
    ) -> Self:
        match suggest:
            case None | False:
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

    @classmethod
    async def external(
        cls,
        suggest: bool | None = None,
        repeated_calls_limit: int | None = None,
        other_tools: Iterable[AnyTool] | None = None,
        **extra: Any,
    ) -> Self:
        external_toolbox: Self = cast(
            Self,
            await ctx.state(ExternalToolbox).fetch(
                suggest_tool=suggest,
                repeated_calls_limit=repeated_calls_limit,
                **extra,
            ),
        )

        if other_tools:
            return external_toolbox.updated(
                tools={**external_toolbox.tools, **{tool.name: tool for tool in other_tools}}
            )

        else:
            return external_toolbox

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

    def available_tools(self) -> Sequence[LMMToolSpecification]:
        return [tool.specification for tool in self.tools.values() if tool.available]

    async def call_tool(
        self,
        name: str,
        /,
        call_id: str,
        arguments: Mapping[str, Any],
    ) -> MultimodalContent:
        if tool := self.tools.get(name):
            return await tool._toolbox_call(  # pyright: ignore[reportPrivateUsage]
                call_id,
                arguments=arguments,
            )

        else:
            raise LMMException(f"Requested tool ({name}) is not defined")

    async def respond_all(
        self,
        requests: LMMToolRequests,
        /,
    ) -> Sequence[LMMToolResponse]:
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


@runtime_checkable
class ToolboxFetching(Protocol):
    async def __call__(
        self,
        *,
        suggest: bool | None = None,
        repeated_calls_limit: int | None = None,
        **extra: Any,
    ) -> Toolbox: ...


class ExternalToolbox(State):
    fetch: ToolboxFetching
