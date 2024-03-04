from draive.scope import ScopeState
from draive.types import Model, StreamingProgressUpdate

__all__ = [
    "ToolCallContext",
]


class ToolCallContext(ScopeState):
    call_id: str | None
    progress: StreamingProgressUpdate[Model] = lambda update: None  # noqa: E731
