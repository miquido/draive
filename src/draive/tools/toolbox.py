from asyncio import Lock, Task
from collections.abc import (
    AsyncGenerator,
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
    ModelToolHandling,
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
        suggesting: ToolsSuggesting | Tool | int | bool | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @classmethod
    def of(  # noqa: C901, PLR0912
        cls,
        tool_or_tools: Self | Iterable[Tool] | Tool | None = None,
        /,
        *tools: Tool,
        suggesting: ToolsSuggesting | Tool | int | bool | None = None,
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
        suggesting : ToolsSuggesting | Tool | int | bool | None
            Selection strategy used to suggest tools to a model.
            ``False``/``None`` disables suggestions, ``True`` enables generic
            suggestion for the first iteration, integer enables suggestion for given number
            of iterations, and a ``Tool`` suggests that specific
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

            elif isinstance(suggesting, int):
                suggestion = _suggest_any(iterations=suggesting)

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

            elif isinstance(suggesting, int):
                suggestion = _suggest_any(iterations=suggesting)

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
        iteration: int,
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
        if not self.tools:
            return ModelTools.none

        specification: Sequence[ModelToolSpecification] = tuple(
            tool.specification for tool in self.tools.values()
        )
        tool_suggestion: ModelToolSpecification | bool = self.suggesting(
            tools=specification,
            iteration=iteration,
        )
        tools_selection: ModelToolsSelection
        if tool_suggestion is False:
            tools_selection = "auto"

        elif tool_suggestion is True:
            tools_selection = "required"

        else:  # ModelToolSpecification
            assert isinstance(tool_suggestion, ModelToolSpecification)  # nosec: B101
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
        """
        async with ctx.scope(f"tool.{tool}"):
            if arguments is None:
                arguments = {}

            selected_tool: Tool | None = self.tools.get(tool)
            if selected_tool is None:
                raise ToolException(
                    f"Requested unknown tool {tool}",
                    tool=tool,
                )

            accumulator: MutableSequence[MultimodalContentPart] = []
            try:
                async for chunk in selected_tool.call(**arguments):
                    if isinstance(chunk, ProcessingEvent):
                        continue  # skip events

                    accumulator.append(chunk)

            except Exception as exc:
                raise ToolException(
                    f"Tool {tool} call failed due to an error: {exc}",
                    tool=tool,
                ) from exc

            return MultimodalContent.of(*accumulator)

    async def handle(  # noqa: C901
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

        async with ContextTaskGroup():  # ensure proper task joins through local task group
            output_stream: AsyncQueue[
                ModelToolResponse | ProcessingEvent | MultimodalContentPart
            ] = AsyncQueue()
            tasks: MutableSet[Task[None]] = set()
            lock: Lock = Lock()  # synchronize outputs

            def task_finish(task: Task[None]) -> None:
                exc: BaseException | None = task.exception()
                if exc is not None:
                    # fail with first exception
                    output_stream.finish(exc)

                elif all(task.done() for task in tasks):
                    # finish when all done
                    output_stream.finish()

            for request in requests:
                tool: Tool | None = self.tools.get(request.tool)
                handling: ModelToolHandling
                if request.handling is None:
                    if tool is None:
                        handling = "response"

                    else:
                        handling = tool.handling

                else:
                    handling = request.handling

                match handling:
                    case "response":
                        task: Task[None] = ctx.spawn(
                            self._response_execute(
                                tool,
                                request=request,
                                output_stream=output_stream,
                            )
                        )
                        tasks.add(task)
                        task.add_done_callback(task_finish)

                    case "output" | "output_stream":
                        task: Task[None] = ctx.spawn(
                            self._output_stream_execute(
                                tool,
                                request=request,
                                lock=lock,
                                output_stream=output_stream,
                            )
                        )
                        tasks.add(task)
                        task.add_done_callback(task_finish)

            async for chunk in output_stream:
                yield chunk

            assert all(task.done() for task in tasks)  # nosec: B101

    async def _execute(
        self,
        tool: Tool | None,
        *,
        request: ModelToolRequest,
    ) -> AsyncGenerator[ModelToolResponse | ProcessingEvent | MultimodalContentPart]:
        async with ctx.scope(f"tool.{request.tool}", request):
            if tool is None:
                ctx.log_error(f"Requested unknown tool `{request.tool}` [{request.identifier}]")
                yield ModelToolResponse(
                    identifier=request.identifier,
                    tool=request.tool,
                    status="error",
                    content=MultimodalContent.of(
                        f"<error>Requested unknown tool `{request.tool}`</error>"
                    ),
                )
                return  # execution finished

            accumulator: MutableSequence[MultimodalContentPart] = []
            try:
                async for chunk in tool.call(**request.arguments):
                    if isinstance(chunk, ProcessingEvent):
                        ctx.record_info(event=chunk.event)
                        yield chunk.updating(
                            meta=chunk.meta.updating(
                                tool=request.tool,
                                request=request.identifier,
                            )
                        )

                    else:
                        accumulator.append(chunk)
                        yield chunk  # stream content to output

                yield ModelToolResponse(
                    identifier=request.identifier,
                    tool=request.tool,
                    status="success",
                    # TODO: perhaps we should replace final response content for direct outputs?
                    content=MultimodalContent.of(*accumulator),
                )

            except Exception as exc:
                ctx.log_error(
                    f"Tool `{request.tool}` execution [{request.identifier}] failed"
                    f" due to an error: {exc}",
                    exception=exc,
                )
                yield ModelToolResponse(
                    identifier=request.identifier,
                    tool=request.tool,
                    status="error",
                    content=MultimodalContent.of(
                        *accumulator,  # preserve output up to the error occurrence
                        # TODO: should we allow custom error message?
                        "<error>Tool execution failed due to an error</error>",
                    ),
                )

    async def _response_execute(
        self,
        tool: Tool | None,
        *,
        request: ModelToolRequest,
        output_stream: AsyncQueue[ModelToolResponse | ProcessingEvent | MultimodalContentPart],
    ) -> None:
        async for chunk in self._execute(
            tool,
            request=request,
        ):
            if isinstance(chunk, ProcessingEvent | ModelToolResponse):
                output_stream.enqueue(chunk)

    async def _output_execute(
        self,
        tool: Tool | None,
        *,
        request: ModelToolRequest,
        lock: Lock,
        output_stream: AsyncQueue[ModelToolResponse | ProcessingEvent | MultimodalContentPart],
    ) -> None:
        accumulator: MutableSequence[MultimodalContentPart] = []
        async for chunk in self._execute(
            tool,
            request=request,
        ):
            if isinstance(chunk, ProcessingEvent | ModelToolResponse):
                output_stream.enqueue(chunk)

            else:
                accumulator.append(chunk)

        async with lock:  # synchronize outputs so only one streams at the same time
            for chunk in accumulator:
                output_stream.enqueue(chunk)

    async def _output_stream_execute(
        self,
        tool: Tool | None,
        *,
        request: ModelToolRequest,
        lock: Lock,
        output_stream: AsyncQueue[ModelToolResponse | ProcessingEvent | MultimodalContentPart],
    ) -> None:
        async with lock:  # synchronize outputs so only one streams at the same time
            async for chunk in self._execute(
                tool,
                request=request,
            ):
                output_stream.enqueue(chunk)

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
