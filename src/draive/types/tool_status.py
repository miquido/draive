from typing import Literal

from draive.types.model import Model

__all__ = [
    "ToolCallStatus",
]


class ToolCallStatus(Model):
    identifier: str
    tool: str
    status: Literal[
        "STARTED",
        "RUNNING",
        "FINISHED",
        "FAILED",
    ]
    content: dict[str, object] | None = None
