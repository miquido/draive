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
    def __init__(  # noqa: PLR0913
        self,
        /,
        name: str,
        *,
        function: Function[Args, Coroutine[None, None, Result]],
        description: str | None = None,
        availability: ToolAvailability | None = None,
        require_direct_result: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            function=function,
            description=description,
        )
        self._require_direct_result: bool = require_direct_result
        self._availability: ToolAvailability = availability or (
            lambda: True  # available by default
        )

        freeze(self)

    @property
    def available(self) -> bool:
        return self._availability()

    @property
    def requires_direct_result(self) -> bool:
        return self._require_direct_result

    async def __call__(
        self,
        call_id: str | None = None,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        call_context: ToolCallContext = ToolCallContext(
            call_id=call_id or uuid4().hex,
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
) -> Tool[Args, Result]:
    """
    Convert a function to a tool using default parameters and no description.

    In order to adjust the arguments behavior and specification use an instance of Argument
    as a default value of any given argument with desired configuration
    for each argument individually.

    Parameters
    ----------
    function: Function[Args, Coroutine[None, None, Result]]
        a function to be wrapped as a Tool.
    Returns
    -------
    Tool[Args, Result]
        a Tool representation of the provided function.
    """


@overload
def tool[**Args, Result](
    *,
    name: str | None = None,
    description: str | None = None,
    availability: ToolAvailability | None = None,
    direct_result: bool = False,
) -> Callable[[Function[Args, Coroutine[None, None, Result]]], Tool[Args, Result]]:
    """
    Convert a function to a tool using provided parameters.

    In order to adjust the arguments behavior and specification use an instance of Argument
    as a default value of any given argument with desired configuration
    for each argument individually.

    Parameters
    ----------
    name: str
        name to be used in a tool specification.
        Default is the name of the wrapped function.
    description: int
        description to be used in a tool specification. Allows to present the tool behavior to the
        external system.
        Default is empty.
    availability: ToolAvailability
        function used to verify availability of the tool in given context. It can be used to check
        permissions or occurrence of a specific state to allow its usage.
        Default is always available.
    direct_result: bool
        controls if tool result should break the ongoing processing and be the direct result of it.
        Note that during concurrent execution of multiple tools the call/result order defines
        direct result and exact behavior is not defined.
        Default is False.

    Returns
    -------
    Callable[[Function[Args, Coroutine[None, None, Result]]], Tool[Args, Result]]
        function allowing to convert other function to a Tool using provided configuration.
    """


def tool[**Args, Result](
    function: Function[Args, Coroutine[None, None, Result]] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    availability: ToolAvailability | None = None,
    direct_result: bool = False,
) -> (
    Callable[[Function[Args, Coroutine[None, None, Result]]], Tool[Args, Result]]
    | Tool[Args, Result]
):
    def wrap(
        function: Function[Args, Coroutine[None, None, Result]],
    ) -> Tool[Args, Result]:
        return Tool(
            name=name or function.__name__,
            description=description,
            function=function,
            availability=availability,
            require_direct_result=direct_result,
        )

    if function := function:
        return wrap(function=function)
    else:
        return wrap
