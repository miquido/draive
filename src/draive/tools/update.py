from typing import Literal

from draive.types.model import Model

__all__ = [
    "ToolCallStatus",
    "ToolCallUpdate",
]


ToolCallStatus = Literal["STARTED", "RUNNING", "FINISHED", "FAILED"]


class ToolCallUpdate(Model):
    call_id: str
    tool: str
    status: ToolCallStatus
    content: Model | None
