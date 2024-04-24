from collections.abc import Callable, Coroutine
from typing import (
    Protocol,
    final,
    overload,
)
from uuid import uuid4

from draive.helpers import freeze
from draive.metrics import ArgumentsTrace, ResultTrace
from draive.parameters import Function, ParametrizedTool
from draive.scope import ctx
from draive.tools.errors import ToolException
from draive.tools.state import ToolCallContext, ToolsUpdatesContext
from draive.tools.update import ToolCallUpdate

__all__ = [
    "tool",
    "Tool",
    "ToolAvailability",
]


class ToolAvailability(Protocol):
    def __call__(self) -> bool: ...


@final
class Tool[**Args, Result](ParametrizedTool[Args, Coroutine[None, None, Result]]):
    def __init__(
        self,
        /,
        name: str,
        *,
        function: Function[Args, Coroutine[None, None, Result]],
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

        freeze(self)

    @property
    def available(self) -> bool:
        return self._availability()

    async def __call__(
        self,
        tool_call_id: str | None = None,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        call_context: ToolCallContext = ToolCallContext(
            call_id=tool_call_id or uuid4().hex,
            tool=self.name,
        )
        send_update: Callable[[ToolCallUpdate], None] = ctx.state(
            ToolsUpdatesContext
        ).send_update or (lambda _: None)
        with ctx.nested(
            self.name,
            state=[call_context],
            metrics=[ArgumentsTrace.of(*args, call_id=call_context.call_id, **kwargs)],
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

                result: Result = await super().__call__(
                    *args,
                    **kwargs,
                )

                ctx.record(ResultTrace.of(result))
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
def tool[**Args, Result](
    function: Function[Args, Coroutine[None, None, Result]],
    /,
) -> Tool[Args, Result]: ...


@overload
def tool[**Args, Result](
    *,
    name: str | None = None,
    description: str | None = None,
    availability: ToolAvailability | None = None,
) -> Callable[[Function[Args, Coroutine[None, None, Result]]], Tool[Args, Result]]: ...


def tool[**Args, Result](
    function: Function[Args, Coroutine[None, None, Result]] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    availability: ToolAvailability | None = None,
) -> (
    Callable[[Function[Args, Coroutine[None, None, Result]]], Tool[Args, Result]]
    | Tool[Args, Result]
):
    """
    Convert a function to a tool. Tool arguments support only limited types.
    """

    def wrap(
        function: Function[Args, Coroutine[None, None, Result]],
    ) -> Tool[Args, Result]:
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
