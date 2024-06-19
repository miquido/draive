from typing import Literal

from draive.parameters.model import DataModel

__all__ = [
    "ToolCallStatus",
]


class ToolCallStatus(DataModel):
    identifier: str
    tool: str
    status: Literal[
        "STARTED",
        "RUNNING",
        "FINISHED",
        "FAILED",
    ]
    content: DataModel | None = None
