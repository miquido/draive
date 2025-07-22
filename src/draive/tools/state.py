from collections.abc import Sequence
from typing import Any

from haiway import META_EMPTY, Meta, State, ctx

from draive.tools.types import Tool, ToolsFetching

__all__ = ("Tools",)


async def _no_tools(
    **extra: Any,
) -> Sequence[Tool]:
    return ()


class Tools(State):
    @classmethod
    async def fetch(
        cls,
        **extra: Any,
    ) -> Sequence[Tool]:
        return await ctx.state(cls).fetching(**extra)

    fetching: ToolsFetching = _no_tools
    meta: Meta = META_EMPTY
