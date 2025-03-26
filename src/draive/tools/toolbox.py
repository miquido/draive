from asyncio import gather
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Protocol, Self, final, runtime_checkable

from haiway import State, ctx

from draive.lmm.types import (
    LMMException,
    LMMToolError,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    LMMToolResponses,
    LMMToolSpecification,
)
from draive.multimodal import MultimodalContent
from draive.tools.tool import AnyTool, Tool

__all__ = [
    "ExternalTools",
    "Toolbox",
    "ToolsFetching",
]


@final
class Toolbox(State):
    @classmethod
    def of(
        cls,
        tool_or_toolbox: Self | AnyTool | Iterable[AnyTool] | None = None,
        /,
        *tools: AnyTool,
        suggest: AnyTool | bool | None = None,
        repeated_calls_limit: int | None = None,
    ) -> Self:
        match tool_or_toolbox:
            case None:
                assert not suggest  # nosec: B101
                return cls(
                    tools={},
                    suggest_call=False,
                    repeated_calls_limit=repeated_calls_limit or 3,
                )

            case Toolbox() as toolbox:
                assert not isinstance(suggest, Tool) or suggest in tools  # nosec: B101
                return cls(
                    tools={
                        **toolbox.tools,
                        **{tool.name: tool for tool in tools},
                    },
                    suggest_call=suggest or toolbox.suggest_call,
                    repeated_calls_limit=repeated_calls_limit or toolbox.repeated_calls_limit,
                )

            case Tool() as tool:
                merged_tools: Sequence[AnyTool] = (tool, *tools)
                assert not isinstance(suggest, Tool) or suggest in merged_tools  # nosec: B101
                return cls(
                    tools={tool.name: tool for tool in merged_tools},
                    suggest_call=suggest or False,
                    repeated_calls_limit=repeated_calls_limit or 3,
                )

            case iterable_tools:
                merged_tools: Sequence[AnyTool] = (*iterable_tools, *tools)
                assert not isinstance(suggest, Tool) or suggest in merged_tools  # nosec: B101
                return cls(
                    tools={tool.name: tool for tool in merged_tools},
                    suggest_call=suggest or False,
                    repeated_calls_limit=repeated_calls_limit or 3,
                )

    @classmethod
    async def external(
        cls,
        repeated_calls_limit: int | None = None,
        suggest_call: AnyTool | bool = False,
        other_tools: Iterable[AnyTool] | None = None,
        **extra: Any,
    ) -> Self:
        external_tools: Sequence[AnyTool] = await ctx.state(ExternalTools).fetch(**extra)
        return cls.of(
            *(*external_tools, *(other_tools or ())),
            suggest=suggest_call or False,
            repeated_calls_limit=repeated_calls_limit or 3,
        )

    tools: Mapping[str, AnyTool]
    suggest_call: AnyTool | bool
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

        elif self.suggest_call is False:
            return "auto"

        elif self.suggest_call is True:
            return "required"

        elif self.suggest_call.available:
            return self.suggest_call.specification  # use suggested tool if able

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
            return await tool._toolbox_call(
                call_id,
                arguments=arguments,
            )

        else:
            raise LMMException(f"Requested tool ({name}) is not defined")

    async def respond_all(
        self,
        requests: LMMToolRequests,
        /,
    ) -> LMMToolResponses:
        return LMMToolResponses(
            responses=await gather(
                *[self.respond(request) for request in requests.requests],
                return_exceptions=False,  # toolbox calls handle errors if able
            )
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

    def with_tools(
        self,
        tool: AnyTool,
        /,
        *tools: AnyTool,
    ) -> Self:
        return self.__class__.of(
            *(tool, *tools, *self.tools.values()),
            suggest=self.suggest_call,
            repeated_calls_limit=self.repeated_calls_limit,
        )


@runtime_checkable
class ToolsFetching(Protocol):
    async def __call__(
        self,
        **extra: Any,
    ) -> Sequence[AnyTool]: ...


class ExternalTools(State):
    fetch: ToolsFetching
