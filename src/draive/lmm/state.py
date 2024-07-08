from collections.abc import Callable, Coroutine
from typing import Literal

from draive.lmm.invocation import LMMInvocation
from draive.parameters import DataModel, State
from draive.types import ToolCallStatus

__all__: list[str] = [
    "LMM",
    "ToolCallContext",
    "ToolStatusStream",
]


class LMM(State):
    invocation: LMMInvocation


class ToolStatusStream(State):
    send: Callable[[ToolCallStatus], Coroutine[None, None, None]] | None = None


class ToolCallContext(State):
    call_id: str
    tool: str
    send_status: Callable[[ToolCallStatus], Coroutine[None, None, None]]

    async def report(
        self,
        status: Literal[
            "STARTED",
            "RUNNING",
            "FINISHED",
            "FAILED",
        ],
        /,
        content: DataModel | None = None,
    ) -> None:
        call_status: ToolCallStatus
        match content:
            case None:
                call_status = ToolCallStatus(
                    identifier=self.call_id,
                    tool=self.tool,
                    status=status,
                )

            case DataModel() as model:
                call_status = ToolCallStatus(
                    identifier=self.call_id,
                    tool=self.tool,
                    status=status,
                    content=model,
                )

        await self.send_status(call_status)
