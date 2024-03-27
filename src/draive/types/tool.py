from typing import Literal

from draive.types.model import Model

__all__ = [
    "ToolCallStatus",
    "ToolCallProgress",
    "ToolException",
]


ToolCallStatus = Literal["STARTED", "RUNNING", "FINISHED", "FAILED"]


class ToolCallProgress(Model):
    call_id: str
    tool: str
    status: ToolCallStatus
    content: Model | None


class ToolException(Exception):
    pass
