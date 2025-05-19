from asyncio import gather
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, Literal, Self, final

from haiway import State, ctx

from draive.lmm.types import (
    LMMException,
    LMMToolError,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    LMMToolResponseHandling,
    LMMToolResponses,
    LMMToolSpecification,
)
from draive.multimodal import MultimodalContent
from draive.tools.state import Tools
from draive.tools.types import Tool

__all__ = ("Toolbox",)


@final
class Toolbox(State):
    @classmethod
    def of(  # noqa: C901, PLR0912
        cls,
        tool_or_toolbox: Self | Tool | Iterable[Tool] | None = None,
        /,
        *tools: Tool,
        suggest: Tool | str | bool | None = None,
        repeated_calls_limit: int | None = None,
    ) -> Self:
        tools_mapping: Mapping[str, Tool]
        suggest_call: Tool | bool
        calls_limit: int
        match tool_or_toolbox:
            case None:
                assert suggest is None or suggest is False  # nosec: B101
                tools_mapping = {}
                suggest_call = False
                calls_limit = 0

            case Toolbox() as toolbox:
                tools_mapping = {
                    **toolbox.tools,
                    **{tool.name: tool for tool in tools},
                }

                match suggest if suggest is not None else toolbox.suggest_call:
                    case bool() as suggestion:
                        suggest_call = suggestion

                    case str() as tool_name:
                        suggest_call = tools_mapping[tool_name]

                    case Tool() as tool:
                        assert tool.name in tools_mapping  # nosec: B101
                        suggest_call = tool

                calls_limit = (
                    repeated_calls_limit
                    if repeated_calls_limit is not None
                    else toolbox.repeated_calls_limit
                )

            case Tool() as tool:
                tools_mapping = {
                    **{tool.name: tool},
                    **{tool.name: tool for tool in tools},
                }

                match suggest:
                    case None:
                        suggest_call = False

                    case bool() as suggestion:
                        suggest_call = suggestion

                    case str() as tool_name:
                        suggest_call = tools_mapping[tool_name]

                    case Tool() as tool:
                        assert tool.name in tools_mapping  # nosec: B101
                        suggest_call = tool

                calls_limit = repeated_calls_limit or 3

            case iterable_tools:
                tools_mapping = {
                    **{tool.name: tool for tool in iterable_tools},
                    **{tool.name: tool for tool in tools},
                }

                match suggest:
                    case None:
                        suggest_call = False

                    case bool() as suggestion:
                        suggest_call = suggestion

                    case str() as tool_name:
                        suggest_call = tools_mapping[tool_name]

                    case Tool() as tool:
                        assert tool.name in tools_mapping  # nosec: B101
                        suggest_call = tool

                calls_limit = repeated_calls_limit or 3

        return cls(
            tools=tools_mapping,
            suggest_call=suggest_call,
            repeated_calls_limit=calls_limit,
        )

    @classmethod
    async def external(
        cls,
        *,
        selection: Collection[str] | None = None,
        repeated_calls_limit: int | None = None,
        suggest: str | bool | None = None,
        other_tools: Iterable[Tool] | None = None,
        **extra: Any,
    ) -> Self:
        external_tools: Sequence[Tool]
        if selected := selection:
            external_tools = [tool for tool in await Tools.fetch(**extra) if tool.name in selected]

        else:
            external_tools = await Tools.fetch(**extra)

        return cls.of(
            *(*external_tools, *(other_tools or ())),
            suggest=suggest,
            repeated_calls_limit=repeated_calls_limit,
        )

    tools: Mapping[str, Tool]
    suggest_call: Tool | bool
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

        elif self.suggest_call.available:  # use suggested tool if able
            return {
                "name": self.suggest_call.name,
                "description": self.suggest_call.description,
                "parameters": self.suggest_call.parameters,
            }

        else:
            return "auto"

    def available_tools(self) -> Sequence[LMMToolSpecification]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self.tools.values()
            if tool.available
        ]

    async def call_tool(
        self,
        name: str,
        /,
        call_id: str,
        arguments: Mapping[str, Any],
    ) -> MultimodalContent:
        if tool := self.tools.get(name):
            return await tool.tool_call(
                call_id,
                **arguments,
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
                handling: LMMToolResponseHandling
                match tool.handling:
                    case "auto":
                        handling = "result"

                    case "direct":
                        handling = "direct_result"

                return LMMToolResponse(
                    identifier=request.identifier,
                    tool=request.tool,
                    content=await tool.tool_call(  # pyright: ignore[reportPrivateUsage]
                        request.identifier,
                        **request.arguments,
                    ),
                    handling=handling,
                )

            except LMMToolError as error:  # use formatted error, blow up on other exception
                ctx.log_error(
                    "Tool (%s) returned an error",
                    request.tool,
                    exception=error,
                )
                handling: LMMToolResponseHandling
                match tool.handling:
                    case "auto":
                        handling = "error"

                    case "direct":
                        handling = "direct_result"

                return LMMToolResponse(
                    identifier=request.identifier,
                    tool=request.tool,
                    content=error.content,
                    handling=handling,
                )

        else:
            # log error and provide fallback result to avoid blowing out the execution
            ctx.log_error("Requested tool (%s) is not defined", request.tool)
            return LMMToolResponse(
                identifier=request.identifier,
                tool=request.tool,
                content=MultimodalContent.of("ERROR"),
                handling="error",
            )

    def with_tools(
        self,
        tool: Tool,
        /,
        *tools: Tool,
    ) -> Self:
        return self.__class__.of(
            *(tool, *tools, *self.tools.values()),
            suggest=self.suggest_call,
            repeated_calls_limit=self.repeated_calls_limit,
        )
