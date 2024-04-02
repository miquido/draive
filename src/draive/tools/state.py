from draive.scope import ctx
from draive.tools.update import ToolCallUpdate
from draive.types import Model, State, UpdateSend

__all__ = [
    "ToolsUpdatesContext",
    "ToolCallContext",
]


class ToolsUpdatesContext(State):
    send_update: UpdateSend[ToolCallUpdate] | None = None


class ToolCallContext(State):
    call_id: str
    tool: str

    def send_update(
        self,
        content: Model,
    ) -> None:
        if send_update := ctx.state(ToolsUpdatesContext).send_update:
            send_update(
                ToolCallUpdate(
                    call_id=self.call_id,
                    tool=self.tool,
                    status="RUNNING",
                    content=content,
                )
            )
