from collections.abc import Callable, Coroutine
from typing import Literal

from haiway import State, async_noop

from draive.parameters import DataModel
from draive.types import MultimodalContent

__all__ = [
    "ToolStatus",
    "ToolContext",
]


class ToolStatus(DataModel):
    identifier: str
    tool: str
    status: Literal[
        "STARTED",
        "PROGRESS",
        "FINISHED",
        "FAILED",
    ]
    content: MultimodalContent | None = None


class ToolContext(State):
    call_id: str
    tool: str
    report_status: Callable[[ToolStatus], Coroutine[None, None, None]] = async_noop

    async def progress(
        self,
        /,
        content: MultimodalContent | None = None,
    ) -> None:
        await self.report_status(
            ToolStatus(
                identifier=self.call_id,
                tool=self.tool,
                status="PROGRESS",
                content=content,
            )
        )
