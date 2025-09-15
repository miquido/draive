from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, ClassVar, Self, final, overload

from haiway import META_EMPTY, Meta, MetaTags, MetaValues, State, ctx

from draive.models.tools.types import Tool, ToolError, ToolsSuggesting
from draive.models.types import (
    ModelToolRequest,
    ModelToolResponse,
    ModelToolsDeclaration,
    ModelToolSpecification,
    ModelToolsSelection,
)
from draive.multimodal import MultimodalContent

__all__ = ("Toolbox",)


@final
class Toolbox(State):
    """A collection of tools with optional suggestion policy.

    Use ``Toolbox.of(...)`` to construct, or start from ``Toolbox.empty`` and build up via
    helper methods. This state is immutable; builders return new instances.

    Attributes
    ----------
    tools : Mapping[str, Tool]
        Registered tools keyed by name.
    suggesting : ToolsSuggesting
        Policy used to suggest a specific tool or selection mode during loops.
    tool_turns_limit : int
        Number of turns to strictly disable suggestions (``selection='none'``).
    meta : Meta
        Additional metadata for the toolbox.
    """

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
        tool_turns_limit: int | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @classmethod
    def of(
        cls,
        tool_or_tools: Self | Iterable[Tool] | Tool | None = None,
        /,
        *tools: Tool,
        suggesting: ToolsSuggesting | Tool | bool | None = None,
        tool_turns_limit: int | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Construct a toolbox from tools and optional suggestion policy.

        Parameters
        ----------
        tool_or_tools : Toolbox | Iterable[Tool] | Tool | None, optional
            Existing toolbox to reuse, an iterable of tools, a single tool, or ``None``.
        tools : Tool
            Additional tools to include.
        suggesting : ToolsSuggesting | Tool | bool | None, optional
            Suggestion policy; ``True`` suggests any tool for given number of turns,
            a ``Tool`` suggests that specific tool, ``False`` disables suggestions,
            or provide a custom ``ToolsSuggesting`` callable.
        tool_turns_limit : int | None, optional
            Number of turns to disable suggestions (strict tools usage). Defaults to 3 for
            non-empty toolbox.
        meta : Meta | Mapping[str, Any] | None, optional
            Additional metadata for the toolbox.

        Returns
        -------
        Toolbox
            Constructed toolbox instance.
        """
        if isinstance(tool_or_tools, Toolbox):
            return tool_or_tools

        tools_mapping: Mapping[str, Tool]
        suggestion: ToolsSuggesting
        turns_limit: int
        metadata: Meta
        if tool_or_tools is None:
            assert suggesting is None or suggesting is False  # nosec: B101
            tools_mapping = {}
            suggestion = _no_suggestion
            turns_limit = 0
            metadata = Meta.of(meta)

        elif isinstance(tool_or_tools, Tool):
            tools_mapping = {
                **{tool_or_tools.name: tool_or_tools},
                **{tool.name: tool for tool in tools},
            }
            if suggesting is None or suggesting is False:
                suggestion = _no_suggestion

            elif suggesting is True:
                suggestion = _suggest_any(
                    turns=1,
                )

            elif isinstance(suggesting, Tool):
                suggestion = _suggest_tool(
                    suggesting,
                    turns=1,
                )

            else:
                suggestion = suggesting

            turns_limit = tool_turns_limit if tool_turns_limit is not None else 3
            metadata = Meta.of(meta)

        else:
            tools_mapping = {
                **{tool.name: tool for tool in tool_or_tools},
                **{tool.name: tool for tool in tools},
            }
            if suggesting is None or suggesting is False:
                suggestion = _no_suggestion

            elif suggesting is True:
                suggestion = _suggest_any(
                    turns=1,
                )

            elif isinstance(suggesting, Tool):
                suggestion = _suggest_tool(
                    suggesting,
                    turns=1,
                )

            else:
                suggestion = suggesting

            turns_limit = tool_turns_limit if tool_turns_limit is not None else 3
            metadata = Meta.of(meta)

        return cls(
            tools=tools_mapping,
            suggesting=suggestion,
            tool_turns_limit=turns_limit,
            meta=metadata,
        )

    tools: Mapping[str, Tool]
    suggesting: ToolsSuggesting
    tool_turns_limit: int
    meta: Meta = META_EMPTY

    def available_tools_declaration(
        self,
        *,
        tools_turn: int = 0,
    ) -> ModelToolsDeclaration:
        """Build a ``ModelToolsDeclaration`` for a given loop turn.

        Filters tools by availability and applies the suggestion policy to derive the
        selection mode.

        Parameters
        ----------
        tools_turn : int, optional
            Current tool turn (0-based).

        Returns
        -------
        ModelToolsDeclaration
            The declaration to pass to model generation.
        """
        available_tools: Sequence[ModelToolSpecification] = tuple(
            tool.specification
            for tool in self.tools.values()
            if tool.available(tools_turn=tools_turn)
        )
        if tools_turn >= self.tool_turns_limit:
            return ModelToolsDeclaration(
                specifications=available_tools,
                selection="none",
            )

        tool_suggestion: ModelToolSpecification | bool = self.suggesting(
            tools_turn=tools_turn,
            tools=available_tools,
        )
        tools_selection: ModelToolsSelection
        if tool_suggestion is False:
            tools_selection = "auto"

        elif tool_suggestion is True:
            tools_selection = "required"

        else:  # ModelToolSpecification
            tools_selection = tool_suggestion.name

        return ModelToolsDeclaration(
            specifications=available_tools,
            selection=tools_selection,
        )

    async def call_tool(
        self,
        name: str,
        /,
        call_id: str,
        arguments: Mapping[str, Any],
    ) -> MultimodalContent:
        """Call a tool by name with provided arguments.

        Parameters
        ----------
        name : str
            Registered tool name.
        call_id : str
            Identifier for the tool invocation.
        arguments : Mapping[str, Any]
            Arguments passed to the underlying tool.

        Returns
        -------
        MultimodalContent
            Content returned by the tool.

        Raises
        ------
        ToolError
            If the requested tool is not defined in this toolbox.
        """
        if tool := self.tools.get(name):
            return await tool.call(
                call_id,
                **arguments,
            )

        else:
            raise ToolError(
                f"Requested tool ({name}) is not defined",
                content=MultimodalContent.empty,
            )

    async def respond(
        self,
        request: ModelToolRequest,
        /,
    ) -> ModelToolResponse:
        """Resolve a ``ModelToolRequest`` to a ``ModelToolResponse``.

        Handles ``detached`` tools by spawning their execution and returning an immediate
        response, and formats ``ToolError`` into error responses. Unknown tools return a
        fallback error response and are logged.

        Parameters
        ----------
        request : ModelToolRequest
            Incoming tool execution request from the model.

        Returns
        -------
        ModelToolResponse
            Response containing success, error or detached acknowledgement payload.
        """
        if tool := self.tools.get(request.tool):
            try:
                if tool.handling == "detached":
                    ctx.spawn(
                        tool.call,
                        request.identifier,
                        **request.arguments,
                    )
                    return ModelToolResponse(
                        identifier=request.identifier,
                        tool=request.tool,
                        content=MultimodalContent.of(
                            f"{request.tool} tool execution has been requested"
                        ),
                        handling="detached",
                        meta=META_EMPTY,
                    )

                return ModelToolResponse(
                    identifier=request.identifier,
                    tool=request.tool,
                    content=await tool.call(
                        request.identifier,
                        **request.arguments,
                    ),
                    handling=tool.handling,
                    meta=META_EMPTY,
                )

            except ToolError as error:  # use formatted error, blow up on other exceptions
                return ModelToolResponse(
                    identifier=request.identifier,
                    tool=request.tool,
                    content=error.content,
                    handling="error",
                    meta=META_EMPTY,
                )

        else:
            # log error and provide fallback result to avoid blowing out the execution
            ctx.log_error(f"Requested tool ({request.tool}) is not defined")
            return ModelToolResponse(
                identifier=request.identifier,
                tool=request.tool,
                content=MultimodalContent.of(f"ERROR: Unavailable tool {request.tool}"),
                handling="error",
                meta=META_EMPTY,
            )

    def with_tools(
        self,
        tool: Tool,
        /,
        *tools: Tool,
    ) -> Self:
        """Return a new toolbox extended with additional tools.

        Parameters
        ----------
        tool : Tool
            Primary tool to add.
        *tools : Tool
            Additional tools to add alongside the primary tool.

        Returns
        -------
        Toolbox
            New toolbox instance containing the original and provided tools.
        """
        return self.__class__(
            tools={
                **self.tools,
                tool.name: tool,
                **{tool.name: tool for tool in tools},
            },
            suggesting=self.suggesting,
            tool_turns_limit=self.tool_turns_limit,
            meta=self.meta,
        )

    def with_suggestion(
        self,
        suggesting: ToolsSuggesting | Tool | bool,
        /,
        *,
        turns: int = 1,
    ) -> Self:
        """Return a new toolbox with an updated suggestion policy.

        Parameters
        ----------
        suggesting : ToolsSuggesting | Tool | bool
            ``True`` to suggest any tool, a ``Tool`` instance to suggest that tool, ``False``
            to disable suggestions, or a custom ``ToolsSuggesting`` implementation.
        turns : int, optional
            Number of initial loop turns for which the suggestion applies. Default is ``1``.

        Returns
        -------
        Toolbox
            New toolbox instance configured with the provided suggestion policy.
        """
        suggestion: ToolsSuggesting
        if suggesting is True:
            suggestion = _suggest_any(
                turns=turns,
            )

        elif suggesting is False:
            suggestion = _no_suggestion

        elif isinstance(suggesting, Tool):
            suggestion = _suggest_tool(
                suggesting,
                turns=turns,
            )

        else:
            suggestion = suggesting

        return self.__class__(
            tools=self.tools,
            suggesting=suggestion,
            tool_turns_limit=self.tool_turns_limit,
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
        """Return a new toolbox filtered by tool names or meta tags.

        Parameters
        ----------
        tools : Collection[str] | None, optional
            Names of tools to keep. Mutually exclusive with ``tags``.
        tags : MetaTags | None, optional
            Metadata tags the tools must contain. Mutually exclusive with ``tools``.

        Returns
        -------
        Toolbox
            Toolbox instance containing only tools matching the filter.

        Notes
        -----
        Exactly one of ``tools`` or ``tags`` must be provided.
        """
        assert tools is None or tags is None  # nosec: B101
        if tools:
            return self.__class__(
                tools={name: tool for name, tool in self.tools.items() if tool.name in tools},
                suggesting=self.suggesting,
                tool_turns_limit=self.tool_turns_limit,
                meta=self.meta,
            )

        elif tags:
            return self.__class__(
                tools={
                    name: tool
                    for name, tool in self.tools.items()
                    if tool.specification.meta.has_tags(tags)
                },
                suggesting=self.suggesting,
                tool_turns_limit=self.tool_turns_limit,
                meta=self.meta,
            )

        else:
            return self


def _suggest_tool(
    tool: Tool,
    /,
    *,
    turns: int,
) -> ToolsSuggesting:
    suggested_tool: ModelToolSpecification = tool.specification

    def suggest_tool(
        tools_turn: int,
        tools: Sequence[ModelToolSpecification],
    ) -> ModelToolSpecification | bool:
        if tools_turn >= turns or suggested_tool not in tools:
            return False

        return suggested_tool

    return suggest_tool


def _suggest_any(
    *,
    turns: int,
) -> ToolsSuggesting:
    def suggest_any(
        tools_turn: int,
        tools: Sequence[ModelToolSpecification],
    ) -> ModelToolSpecification | bool:
        return tools_turn < turns and len(tools) > 0

    return suggest_any


def _no_suggestion(
    tools_turn: int,
    tools: Sequence[ModelToolSpecification],
) -> ModelToolSpecification | bool:
    return False


Toolbox.empty = Toolbox(
    tools={},
    suggesting=_no_suggestion,
    tool_turns_limit=0,
    meta=META_EMPTY,
)
