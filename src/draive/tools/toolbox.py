from asyncio import gather
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, Self, final, overload

from haiway import State, ctx

from draive.commons import META_EMPTY, Meta, MetaTags, MetaValues
from draive.lmm.types import (
    LMMException,
    LMMToolError,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    LMMToolResponseHandling,
    LMMToolResponses,
    LMMTools,
    LMMToolSelection,
)
from draive.multimodal import MultimodalContent
from draive.tools.state import Tools
from draive.tools.types import Tool

__all__ = ("Toolbox",)


@final
class Toolbox(State):
    @classmethod
    def of(  # noqa: C901, PLR0912, PLR0915
        cls,
        tool_or_toolbox: Self | Tool | Iterable[Tool] | None = None,
        /,
        *tools: Tool,
        suggest: Tool | str | bool | None = None,
        repeated_calls_limit: int | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        tools_mapping: Mapping[str, Tool]
        suggest_call: Tool | bool
        calls_limit: int
        metadata: Meta
        match tool_or_toolbox:
            case None:
                assert suggest is None or suggest is False  # nosec: B101
                tools_mapping = {}
                suggest_call = False
                calls_limit = 0
                metadata = Meta.of(meta)

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
                metadata = (
                    toolbox.meta.merged_with(Meta.of(meta)) if meta is not None else toolbox.meta
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
                metadata = Meta.of(meta)

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
                metadata = Meta.of(meta)

        return cls(
            tools=tools_mapping,
            suggest_call=suggest_call,
            repeated_calls_limit=calls_limit,
            meta=metadata,
        )

    @classmethod
    async def fetched(
        cls,
        *,
        tools: Collection[str] | None = None,
        tags: MetaTags | None = None,
        repeated_calls_limit: int | None = None,
        suggest: str | bool | None = None,
        meta: Meta | MetaValues | None = None,
        **extra: Any,
    ) -> Self:
        selected_tools: Sequence[Tool] = await Tools.fetch(**extra)
        if tools:
            selected_tools = [tool for tool in selected_tools if tool.name in tools]

        if tags:
            selected_tools = [tool for tool in selected_tools if tool.meta.has_tags(tags)]

        return cls.of(
            *selected_tools,
            suggest=suggest,
            repeated_calls_limit=repeated_calls_limit,
            meta=meta,
        )

    tools: Mapping[str, Tool]
    suggest_call: Tool | bool
    repeated_calls_limit: int
    meta: Meta = META_EMPTY

    def available_tools(
        self,
        *,
        repetition_level: int = 0,
    ) -> LMMTools:
        if repetition_level >= self.repeated_calls_limit:
            # provide no tools if reached the limit
            return LMMTools.of((), selection="none")

        tools_selection: LMMToolSelection
        if repetition_level != 0:
            # require tools only for the first call, use auto otherwise
            tools_selection = "auto"

        elif self.suggest_call is False:
            tools_selection = "auto"

        elif self.suggest_call is True:
            tools_selection = "required"

        elif self.suggest_call.available:  # use suggested tool if able
            tools_selection = {
                "name": self.suggest_call.name,
                "description": self.suggest_call.description,
                "parameters": self.suggest_call.parameters,
            }

        else:
            tools_selection = "auto"

        return LMMTools.of(
            tuple(tool.specification for tool in self.tools.values() if tool.available),
            selection=tools_selection,
        )

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
            meta=self.meta,
        )

    @overload
    def filtered(
        self,
        *,
        tools: Collection[str],
    ) -> Self: ...

    @overload
    def filtered(
        self,
        *,
        tags: MetaTags,
    ) -> Self: ...

    def filtered(
        self,
        *,
        tools: Collection[str] | None = None,
        tags: MetaTags | None = None,
    ) -> Self:
        assert tools is None or tags is None  # nosec: B101
        if tools:
            return self.__class__.of(
                *(tool for tool in self.tools.values() if tool.name in tools),
                suggest=self.suggest_call,
                repeated_calls_limit=self.repeated_calls_limit,
                meta=self.meta,
            )

        elif tags:
            return self.__class__.of(
                *(tool for tool in self.tools.values() if tool.meta.has_tags(tags)),
                suggest=self.suggest_call,
                repeated_calls_limit=self.repeated_calls_limit,
                meta=self.meta,
            )

        else:
            return self
