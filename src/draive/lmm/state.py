from collections.abc import Callable
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
    send: Callable[[ToolCallStatus], None] | None = None


class ToolCallContext(State):
    call_id: str
    tool: str
    send_status: Callable[[ToolCallStatus], None]

    def report(
        self,
        status: Literal[
            "STARTED",
            "RUNNING",
            "FINISHED",
            "FAILED",
        ],
        /,
        content: DataModel | dict[str, object] | None = None,
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
                    content=model.as_dict(),
                )

            case content:
                call_status = ToolCallStatus(
                    identifier=self.call_id,
                    tool=self.tool,
                    status=status,
                    content=content,
                )

        self.send_status(call_status)
