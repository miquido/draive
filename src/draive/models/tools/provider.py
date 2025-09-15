from collections.abc import Sequence
from typing import Any, final, overload

from haiway import META_EMPTY, Meta, MetaValues, State, statemethod

from draive.models.tools.toolbox import Toolbox
from draive.models.tools.types import Tool, ToolsLoading, ToolsSuggesting

__all__ = ("ToolsProvider",)


async def _no_tools(
    **extra: Any,
) -> Sequence[Tool]:
    return ()


@final
class ToolsProvider(State):
    """Loads tools and materializes ``Toolbox`` instances on demand.

    Providers implement ``loading`` to fetch or construct available tools (possibly remote),
    and the ``toolbox`` statemethod wraps them with optional suggestion policy and metadata.
    """

    @overload
    @classmethod
    async def toolbox(
        cls,
        *tools: Tool,
        suggesting: ToolsSuggesting | bool | None = None,
        tool_turns_limit: int | None = None,
        meta: Meta | MetaValues | None = None,
        **extra: Any,
    ) -> Toolbox: ...

    @overload
    async def toolbox(
        self,
        *tools: Tool,
        suggesting: ToolsSuggesting | bool | None = None,
        tool_turns_limit: int | None = None,
        meta: Meta | MetaValues | None = None,
        **extra: Any,
    ) -> Toolbox: ...

    @statemethod
    async def toolbox(
        self,
        *tools: Tool,
        suggesting: ToolsSuggesting | bool | None = None,
        tool_turns_limit: int | None = None,
        meta: Meta | MetaValues | None = None,
        **extra: Any,
    ) -> Toolbox:
        """Create a ``Toolbox`` from loaded and extra tools.

        Parameters
        ----------
        tools : Tool
            Additional tools to add on top of those returned from ``loading``.
        suggesting : ToolsSuggesting | bool | None, optional
            Suggestion policy; ``True`` suggests any tool for first turns, ``False`` disables,
            or provide custom logic.
        tool_turns_limit : int | None, optional
            Number of turns to disable suggestions (strict tools). Defaults to a sensible
            value in ``Toolbox.of``.
        meta : Meta | MetaValues | None, optional
            Toolbox metadata accepted either as existing ``Meta`` or values convertible via
            ``Meta.of``.
        **extra : Any
            Extra kwargs forwarded to ``loading``.

        Returns
        -------
        Toolbox
            Constructed toolbox instance.
        """
        return Toolbox.of(
            (*await self.loading(**extra), *tools),
            suggesting=suggesting,
            tool_turns_limit=tool_turns_limit,
            meta=meta,
        )

    loading: ToolsLoading = _no_tools
    meta: Meta = META_EMPTY
