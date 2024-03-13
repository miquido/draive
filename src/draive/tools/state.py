from draive.scope import ScopeState, ctx
from draive.types import Model, ProgressUpdate, ToolCallProgress

__all__ = [
    "ToolsProgressContext",
    "ToolCallContext",
]


class ToolsProgressContext(ScopeState):
    progress: ProgressUpdate[ToolCallProgress] = lambda update: None  # noqa: E731


class ToolCallContext(ScopeState):
    call_id: str
    tool: str

    def progress(
        self,
        content: Model,
    ) -> None:
        ctx.state(ToolsProgressContext).progress(
            ToolCallProgress(
                call_id=self.call_id,
                tool=self.tool,
                status="RUNNING",
                content=content,
            )
        )
