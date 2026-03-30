from asyncio import Task
from collections.abc import (
    AsyncIterable,
    Collection,
    Iterable,
    Mapping,
    MutableSequence,
    MutableSet,
    Sequence,
)
from typing import ClassVar, Self, final, overload

from haiway import AsyncQueue, BasicValue, Meta, MetaTags, MetaValues, State, ctx
from haiway.context.tasks import ContextTaskGroup

from draive.models import (
    ModelToolDetachedHandling,
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
    ModelToolSpecification,
    ModelToolsSelection,
)
from draive.multimodal import MultimodalContent, MultimodalContentPart
from draive.tools.types import Tool, ToolException, ToolsSuggesting
from draive.utils import ProcessingEvent

__all__ = ("Toolbox",)


@final
class Toolbox(State):
    """Immutable registry of tools with selection and execution helpers."""

    empty: ClassVar[Self]  # defined after the class

    @overload
    @classmethod
    def of(
        cls,
        tool_or_tools: Self | Iterable[Tool] | None = None,
        /,
    ) -> Self: ...

    @overload
    @classmethod
    def of(
        cls,
        tool_or_tools: Iterable[Tool] | Tool | None,
        /,
        *tools: Tool,
        suggesting: ToolsSuggesting | Tool | bool | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @classmethod
    def of(
        cls,
        tool_or_tools: Self | Iterable[Tool] | Tool | None = None,
        /,
        *tools: Tool,
        suggesting: ToolsSuggesting | Tool | bool | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a toolbox from tools, an existing toolbox, or no tools.

        Parameters
        ----------
        tool_or_tools : Self | Iterable[Tool] | Tool | None
            Existing toolbox instance to reuse, a single tool, an iterable of tools,
            or ``None`` to create an empty toolbox.
        *tools : Tool
            Additional tools merged into the resulting toolbox. Tools are indexed by
            name and later duplicates replace previous ones.
        suggesting : ToolsSuggesting | Tool | bool | None
            Selection strategy used to suggest tools to a model.
            ``False``/``None`` disables suggestions, ``True`` enables generic
            suggestion for the first iteration, and a ``Tool`` suggests that specific
            tool for the first iteration when available.
        meta : Meta | MetaValues | None
            Optional metadata attached to the toolbox state.

        Returns
        -------
        Self
            New toolbox instance or the same instance when ``tool_or_tools`` is
            already a toolbox.
        """
        if isinstance(tool_or_tools, Toolbox):
            assert not tools  # nosec: B101
            assert suggesting is None  # nosec: B101
            return tool_or_tools

        tools_mapping: Mapping[str, Tool]
        suggestion: ToolsSuggesting
        metadata: Meta
        if tool_or_tools is None:
            assert suggesting is None or suggesting is False  # nosec: B101
            tools_mapping = {}
            suggestion = _no_suggestion
            metadata = Meta.of(meta)

        elif isinstance(tool_or_tools, Tool):
            tools_mapping = {
                **{tool_or_tools.name: tool_or_tools},
                **{tool.name: tool for tool in tools},
            }
            if suggesting is None or suggesting is False:
                suggestion = _no_suggestion

            elif suggesting is True:
                suggestion = _suggest_any(iterations=1)

            elif isinstance(suggesting, Tool):
                suggestion = _suggest_tool(
                    suggesting,
                    iterations=1,
                )

            else:
                suggestion = suggesting

            metadata = Meta.of(meta)

        else:
            tools_mapping = {
                **{tool.name: tool for tool in tool_or_tools},
                **{tool.name: tool for tool in tools},
            }
            if suggesting is None or suggesting is False:
                suggestion = _no_suggestion

            elif suggesting is True:
                suggestion = _suggest_any(iterations=1)

            elif isinstance(suggesting, Tool):
                suggestion = _suggest_tool(
                    suggesting,
                    iterations=1,
                )

            else:
                suggestion = suggesting

            metadata = Meta.of(meta)

        return cls(
            tools=tools_mapping,
            suggesting=suggestion,
            meta=metadata,
        )

    tools: Mapping[str, Tool]
    suggesting: ToolsSuggesting
    meta: Meta = Meta.empty

    def model_tools(
        self,
        *,
        iteration: int = 0,
    ) -> ModelTools:
        """Build model-facing tool configuration for a specific iteration.

        Parameters
        ----------
        iteration : int, default=0
            Conversation iteration passed to each tool availability check and to the
            suggestion strategy.

        Returns
        -------
        ModelTools
            Tool specifications of currently available tools plus tool selection mode.
            Suggestion ``False`` maps to ``"auto"``, suggestion ``True`` maps to
            ``"required"``, and a suggested specification is forwarded directly.
        """
        specification = tuple(tool.specification for tool in self.tools.values())
        if not specification:
            return ModelTools.none

        tool_suggestion: ModelToolSpecification | bool = self.suggesting(
            iteration=iteration,
            tools=specification,
        )
        if (
            tool_suggestion is not False
            and tool_suggestion is not True
            and tool_suggestion not in specification
        ):
            tool_suggestion = next(
                (
                    available_tool
                    for available_tool in specification
                    if available_tool.name == tool_suggestion.name
                ),
                False,
            )

        tools_selection: ModelToolsSelection
        if tool_suggestion is False:
            tools_selection = "auto"

        elif tool_suggestion is True:
            tools_selection = "required"

        else:  # ModelToolSpecification
            tools_selection = tool_suggestion

        return ModelTools(
            specification=specification,
            selection=tools_selection,
        )

    async def call(
        self,
        tool: str,
        arguments: Mapping[str, BasicValue] | None = None,
    ) -> MultimodalContent:
        """Execute a single tool and collect only its content output.

        Parameters
        ----------
        tool : str
            Name of the tool to execute.
        arguments : Mapping[str, BasicValue] | None, default=None
            Tool arguments passed to the selected tool.

        Returns
        -------
        MultimodalContent
            Aggregated non-event output produced by the tool.

        Raises
        ------
        ToolException
            If no tool with the given name exists in the toolbox.
        Exception
            Re-raises any exception emitted by the underlying tool call after
            logging it.
        """
        async with ctx.scope(f"tool.{tool}"):
            if arguments is None:
                arguments = {}

            selected_tool: Tool | None = self.tools.get(tool)
            if selected_tool is None:
                ctx.log_error(f"Requested unknown tool {tool}")
                raise ToolException(f"Requested unknown tool {tool}")

            accumulator: MutableSequence[MultimodalContentPart] = []
            try:
                async for chunk in selected_tool.call(**arguments):
                    if isinstance(chunk, ProcessingEvent):
                        continue

                    accumulator.append(chunk)

            except Exception as exc:
                ctx.log_error(
                    f"Tool {tool} call failed due to an error: {exc}",
                    exception=exc,
                )
                raise

            return MultimodalContent.of(*accumulator)

    async def handle(  # noqa: C901, PLR0912, PLR0915
        self,
        *requests: ModelToolRequest,
    ) -> AsyncIterable[ModelToolResponse | ProcessingEvent | MultimodalContentPart]:
        """Execute model tool requests and stream responses, events, and output.

        Parameters
        ----------
        *requests : ModelToolRequest
            Tool requests to execute. Each request is dispatched according to the
            matched tool handling mode.

        Yields
        ------
        ModelToolResponse | ProcessingEvent | MultimodalContentPart
            Tool events and output chunks emitted during execution, followed by tool
            responses describing final status and aggregated results.
        """
        if not requests:
            return  # nothing to be done

        # TODO: refine processing for better handling
        async with ContextTaskGroup():  # ensure proper task joins through local task group
            tasks: MutableSet[Task[None]] = set()
            output: AsyncQueue[ModelToolResponse | ProcessingEvent | MultimodalContentPart] = (
                AsyncQueue()
            )
            for request in requests:
                tool: Tool | None = self.tools.get(request.tool)
                if tool is None:
                    async with ctx.scope(f"tool.{request.tool}", request):
                        ctx.log_error(
                            f"Requested unknown tool {request.tool} call [{request.identifier}]"
                        )
                        yield ModelToolResponse.of(
                            request.identifier,
                            tool=request.tool,
                            result=MultimodalContent.of("ERROR: Unknown tool"),
                            handling="response",
                            status="error",
                        )

                elif tool.handling == "response":

                    async def tool_handler(
                        tool: Tool = tool,
                        request: ModelToolRequest = request,
                    ) -> None:
                        async with ctx.scope(f"tool.{request.tool}", request):
                            accumulator: MutableSequence[MultimodalContentPart] = []
                            try:
                                async for chunk in tool.call(**request.arguments):
                                    if isinstance(chunk, ProcessingEvent):
                                        output.enqueue(  # pass events
                                            chunk.updating(
                                                meta=chunk.meta.updating(
                                                    identifier=request.identifier,
                                                    tool=request.tool,
                                                )
                                            )
                                        )

                                    else:
                                        accumulator.append(chunk)  # accumulate output

                                output.enqueue(
                                    ModelToolResponse.of(
                                        request.identifier,
                                        tool=request.tool,
                                        result=MultimodalContent.of(*accumulator),
                                        handling=tool.handling,
                                        status="success",
                                    )
                                )

                            except Exception as exc:
                                ctx.log_error(
                                    f"Tool {request.tool} call [{request.identifier}] failed"
                                    f" due to an error: {exc}",
                                    exception=exc,
                                )
                                output.enqueue(
                                    ModelToolResponse.of(
                                        request.identifier,
                                        tool=request.tool,
                                        result=MultimodalContent.of(*accumulator)
                                        if accumulator
                                        else MultimodalContent.of("ERROR"),  # TODO: custom message?
                                        handling=tool.handling,
                                        status="error",
                                    )
                                )

                    tasks.add(ctx.spawn(tool_handler))

                elif isinstance(tool.handling, ModelToolDetachedHandling):

                    async def tool_handler(
                        tool: Tool = tool,
                        request: ModelToolRequest = request,
                    ) -> None:
                        async with ctx.scope(f"tool.{request.tool}", request):
                            try:
                                async for _ in tool.call(**request.arguments):
                                    pass  # just execute and ensure completion

                            except Exception as exc:
                                ctx.log_error(
                                    f"Tool {request.tool} call [{request.identifier}] failed"
                                    f" due to an error: {exc}",
                                    exception=exc,
                                )

                    ctx.spawn_background(tool_handler)

                    yield ModelToolResponse.of(
                        request.identifier,
                        tool=request.tool,
                        result=MultimodalContent.of(tool.handling.detach_message),
                        handling=tool.handling,
                        status="success",
                    )

                else:  # direct output is handled synchronously to avoid messing up output chunks
                    assert tool.handling == "output"  # nosec: B101
                    # TODO: we could postpone output tools and spawn other types first
                    # for better parallelism
                    async with ctx.scope(f"tool.{request.tool}", request):
                        accumulator: MutableSequence[MultimodalContentPart] = []
                        try:
                            async for chunk in tool.call(**request.arguments):
                                if isinstance(chunk, ProcessingEvent):
                                    yield chunk.updating(
                                        meta=chunk.meta.updating(
                                            identifier=request.identifier,
                                            tool=request.tool,
                                        )
                                    )

                                else:
                                    yield chunk
                                    accumulator.append(chunk)

                            yield ModelToolResponse.of(
                                request.identifier,
                                tool=request.tool,
                                result=MultimodalContent.of(*accumulator),
                                handling=tool.handling,
                                status="success",
                            )

                        except Exception as exc:
                            ctx.log_error(
                                f"Tool {request.tool} call [{request.identifier}] failed"
                                f" due to an error: {exc}",
                                exception=exc,
                            )
                            yield ModelToolResponse.of(
                                request.identifier,
                                tool=request.tool,
                                result=MultimodalContent.of(*accumulator)
                                if accumulator
                                else MultimodalContent.of("ERROR"),  # TODO: custom message?
                                handling=tool.handling,
                                status="error",
                            )

            if not tasks:
                output.finish()
                return

            def task_finish(task: Task[None]) -> None:
                exc: BaseException | None = task.exception()
                if exc is not None:
                    # fail with first exception
                    output.finish(exc)

                elif all(task.done() for task in tasks):
                    # finish when all done
                    output.finish()

            for task in tasks:
                task.add_done_callback(task_finish)

            async for chunk in output:
                yield chunk

            assert all(task.done() for task in tasks)  # nosec: B101

    def with_tools(
        self,
        tool: Tool,
        /,
        *tools: Tool,
    ) -> Self:
        """Return a copy with additional or replaced tools.

        Parameters
        ----------
        tool : Tool
            First tool to add.
        *tools : Tool
            Additional tools to add. Existing tools with the same name are replaced.

        Returns
        -------
        Self
            Toolbox containing previous tools and provided additions.
        """
        return self.__class__(
            tools={
                **self.tools,
                tool.name: tool,
                **{tool.name: tool for tool in tools},
            },
            suggesting=self.suggesting,
            meta=self.meta,
        )

    def with_suggestion(
        self,
        suggesting: ToolsSuggesting | Tool | bool,
        /,
        *,
        iterations: int = 1,
    ) -> Self:
        """Return a copy with updated tool suggestion strategy.

        Parameters
        ----------
        suggesting : ToolsSuggesting | Tool | bool
            New suggestion strategy. ``True`` suggests any available tool for the
            configured number of iterations, ``False`` disables suggestions, and a
            ``Tool`` suggests that specific tool.
        iterations : int, default=1
            Maximum number of iterations where generated strategy can return
            a suggestion when ``suggesting`` is ``True`` or a ``Tool``.

        Returns
        -------
        Self
            Toolbox with unchanged tools and metadata but replaced suggestion logic.
        """
        suggestion: ToolsSuggesting
        if suggesting is True:
            suggestion = _suggest_any(
                iterations=iterations,
            )

        elif suggesting is False:
            suggestion = _no_suggestion

        elif isinstance(suggesting, Tool):
            suggestion = _suggest_tool(
                suggesting,
                iterations=iterations,
            )

        else:
            suggestion = suggesting

        return self.__class__(
            tools=self.tools,
            suggesting=suggestion,
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
        """Return a filtered view of the toolbox.

        Parameters
        ----------
        tools : Collection[str] | None, default=None
            Allowed tool names. When provided, only tools with matching names are
            kept.
        tags : MetaTags | None, default=None
            Required metadata tags. When provided, only tools whose specification
            metadata contains all tags are kept.

        Returns
        -------
        Self
            Filtered toolbox instance. When no filters are provided, returns ``self``.
        """
        assert tools is None or tags is None  # nosec: B101
        if tools is not None:
            return self.__class__(
                tools={name: tool for name, tool in self.tools.items() if tool.name in tools},
                suggesting=self.suggesting,
                meta=self.meta,
            )

        elif tags is not None:
            return self.__class__(
                tools={
                    name: tool
                    for name, tool in self.tools.items()
                    if tool.specification.meta.has_tags(tags)
                },
                suggesting=self.suggesting,
                meta=self.meta,
            )

        else:
            return self


def _suggest_tool(
    tool: Tool,
    /,
    *,
    iterations: int,
) -> ToolsSuggesting:
    suggested_tool: ModelToolSpecification = tool.specification

    def suggest_tool(
        tools: Sequence[ModelToolSpecification],
        iteration: int,
    ) -> ModelToolSpecification | bool:
        if iteration >= iterations:
            return False

        if suggested_tool in tools:
            return suggested_tool

        return next(
            (
                available_tool
                for available_tool in tools
                if available_tool.name == suggested_tool.name
            ),
            False,
        )

    return suggest_tool


def _suggest_any(
    *,
    iterations: int,
) -> ToolsSuggesting:
    def suggest_any(
        tools: Sequence[ModelToolSpecification],
        iteration: int,
    ) -> ModelToolSpecification | bool:
        return iteration < iterations and len(tools) > 0

    return suggest_any


def _no_suggestion(
    tools: Sequence[ModelToolSpecification],
    iteration: int,
) -> ModelToolSpecification | bool:
    return False


Toolbox.empty = Toolbox(
    tools={},
    suggesting=_no_suggestion,
    meta=Meta.empty,
)
