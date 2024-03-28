from collections.abc import Callable, Coroutine
from typing import (
    Any,
    ParamSpec,
    Protocol,
    TypeVar,
    final,
    overload,
)
from uuid import uuid4

from draive.scope import ArgumentsTrace, ResultTrace, ctx
from draive.tools.errors import ToolException
from draive.tools.state import ToolCallContext, ToolsUpdatesContext
from draive.tools.update import ToolCallUpdate
from draive.types import (
    Function,
    ParametrizedTool,
    UpdateSend,
)

__all__ = [
    "ToolAvailability",
    "Tool",
    "tool",
]


class ToolAvailability(Protocol):
    def __call__(self) -> bool:
        ...


ToolArgs = ParamSpec(
    name="ToolArgs",
    # bound= - ideally it should be bound to allowed types, not implemented in python yet
)
ToolResult = TypeVar(
    name="ToolResult",
)


@final
class Tool(ParametrizedTool[ToolArgs, Coroutine[None, None, ToolResult]]):
    def __init__(
        self,
        /,
        name: str,
        *,
        function: Function[ToolArgs, Coroutine[None, None, ToolResult]],
        description: str | None = None,
        availability: ToolAvailability | None = None,
    ) -> None:
        super().__init__(
            name=name,
            function=function,
            description=description,
        )
        self._availability: ToolAvailability = availability or (
            lambda: True  # available by default
        )

        def frozen(
            __name: str,
            __value: Any,
        ) -> None:
            raise RuntimeError("Tool can't be modified")

        self.__setattr__ = frozen

    @property
    def available(self) -> bool:
        return self._availability()

    async def __call__(
        self,
        tool_call_id: str | None = None,
        *args: ToolArgs.args,
        **kwargs: ToolArgs.kwargs,
    ) -> ToolResult:
        call_context: ToolCallContext = ToolCallContext(
            call_id=tool_call_id or uuid4().hex,
            tool=self.name,
        )
        send_update: UpdateSend[ToolCallUpdate] = ctx.state(ToolsUpdatesContext).send_update or (
            lambda update: None
        )
        with ctx.nested(
            self.name,
            state=[call_context],
            metrics=[ArgumentsTrace(call_id=call_context.call_id, **kwargs)],
        ):
            try:
                send_update(  # notify on start
                    ToolCallUpdate(
                        call_id=call_context.call_id,
                        tool=call_context.tool,
                        status="STARTED",
                        content=None,
                    )
                )
                if not self.available:
                    raise ToolException("Attempting to use unavailable tool", self.name)

                result: ToolResult = await super().__call__(
                    *args,
                    **kwargs,
                )

                ctx.record(ResultTrace(result))
                send_update(  # notify on finish
                    ToolCallUpdate(
                        call_id=call_context.call_id,
                        tool=call_context.tool,
                        status="FINISHED",
                        content=None,
                    )
                )

                return result

            except Exception as exc:
                send_update(  # notify on fail
                    ToolCallUpdate(
                        call_id=call_context.call_id,
                        tool=call_context.tool,
                        status="FAILED",
                        content=None,
                    )
                )
                raise ToolException(
                    "Tool call %s of %s failed due to an error: %s",
                    call_context.call_id,
                    call_context.tool,
                    exc,
                ) from exc


@overload
def tool(
    function: Function[ToolArgs, Coroutine[None, None, ToolResult]],
    /,
) -> Tool[ToolArgs, ToolResult]:
    ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    availability: ToolAvailability | None = None,
) -> Callable[[Function[ToolArgs, Coroutine[None, None, ToolResult]]], Tool[ToolArgs, ToolResult]]:
    ...


def tool(
    function: Function[ToolArgs, Coroutine[None, None, ToolResult]] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    availability: ToolAvailability | None = None,
) -> (
    Callable[[Function[ToolArgs, Coroutine[None, None, ToolResult]]], Tool[ToolArgs, ToolResult]]
    | Tool[ToolArgs, ToolResult]
):
    """
    Convert a function to a tool. Tool arguments support only limited types.
    """

    def wrap(
        function: Function[ToolArgs, Coroutine[None, None, ToolResult]],
    ) -> Tool[ToolArgs, ToolResult]:
        return Tool(
            name=name or function.__name__,
            description=description,
            function=function,
            availability=availability,
        )

    if function := function:
        return wrap(function=function)
    else:
        return wrap
