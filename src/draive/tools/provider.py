from collections.abc import Sequence
from typing import Any, final, overload

from haiway import Meta, MetaValues, State, statemethod

from draive.tools.toolbox import Toolbox
from draive.tools.types import Tool, ToolsLoading, ToolsSuggesting

__all__ = ("ToolsProvider",)


@final
class ToolsProvider(State):
    """State wrapper exposing async tool loaders as toolbox helpers.

    Instances keep an async tools loader together with provider metadata and expose
    convenience helpers for loading tools or building a toolbox in the active context.
    """

    @overload
    @classmethod
    async def load(
        cls,
        **extra: Any,
    ) -> Sequence[Tool]: ...

    @overload
    async def load(
        self,
        **extra: Any,
    ) -> Sequence[Tool]: ...

    @statemethod
    async def load(
        self,
        **extra: Any,
    ) -> Sequence[Tool]:
        """Load tools for the current context.

        Parameters
        ----------
        **extra : Any
            Loader-specific contextual arguments.

        Returns
        -------
        Sequence[Tool]
            Loaded tools ready to be added to a toolbox.
        """
        return await self._loading(**extra)

    _loading: ToolsLoading
    meta: Meta

    def __init__(
        self,
        loading: ToolsLoading,
        meta: Meta = Meta.empty,
    ) -> None:
        """Initialize the provider.

        Parameters
        ----------
        loading : ToolsLoading
            Async loader returning runtime tools.
        meta : Meta, default=Meta.empty
            Provider metadata stored in state.
        """
        super().__init__(
            _loading=loading,
            meta=meta,
        )

    @overload
    @classmethod
    async def toolbox(
        cls,
        *tools: Tool,
        suggesting: ToolsSuggesting | Tool | bool | None = None,
        meta: Meta | MetaValues | None = None,
        **extra: Any,
    ) -> Toolbox: ...

    @overload
    async def toolbox(
        self,
        *tools: Tool,
        suggesting: ToolsSuggesting | Tool | bool | None = None,
        meta: Meta | MetaValues | None = None,
        **extra: Any,
    ) -> Toolbox: ...

    @statemethod
    async def toolbox(
        self,
        *tools: Tool,
        suggesting: ToolsSuggesting | Tool | bool | None = None,
        meta: Meta | MetaValues | None = None,
        **extra: Any,
    ) -> Toolbox:
        """Load tools and build a toolbox for the current context.

        Parameters
        ----------
        *tools : Tool
            Additional tools appended to the loaded tools.
        suggesting : ToolsSuggesting | Tool | bool | None, default=None
            Suggestion strategy forwarded to :meth:`Toolbox.of`.
        meta : Meta | MetaValues | None, default=None
            Metadata attached to the resulting toolbox.
        **extra : Any
            Loader-specific contextual arguments forwarded to the configured tool
            loader.

        Returns
        -------
        Toolbox
            Toolbox containing loaded and explicitly supplied tools.
        """
        return Toolbox.of(
            (*await self._loading(**extra), *tools),
            suggesting=suggesting,
            meta=self.meta if meta is None else meta,
        )
