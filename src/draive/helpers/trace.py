from asyncio import iscoroutinefunction
from collections.abc import Callable, Coroutine
from typing import Any, cast

from draive.metrics import ArgumentsTrace, ResultTrace
from draive.scope import ctx
from draive.utils import mimic_function

__all__ = [
    "traced",
]


def traced[**Args, Result](
    function: Callable[Args, Result],
    /,
) -> Callable[Args, Result]:
    if iscoroutinefunction(function):
        return cast(
            Callable[Args, Result],
            _traced_async(function),
        )
    else:
        return _traced_sync(function)


def _traced_sync[**Args, Result](
    function: Callable[Args, Result],
    /,
) -> Callable[Args, Result]:
    label: str = function.__name__

    @mimic_function(function)
    def wrapped(
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        with ctx.nested(
            label,
            metrics=[ArgumentsTrace.of(*args, **kwargs)],
        ):
            result: Result = function(*args, **kwargs)
            ctx.record(ResultTrace.of(result))
            return result

    return wrapped


def _traced_async[**Args, Result](
    function: Callable[Args, Coroutine[Any, Any, Result]],
    /,
) -> Callable[Args, Coroutine[Any, Any, Result]]:
    label: str = function.__name__

    @mimic_function(function)
    async def wrapped(
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        with ctx.nested(
            label,
            metrics=[ArgumentsTrace.of(*args, **kwargs)],
        ):
            result: Result = await function(*args, **kwargs)
            ctx.record(ResultTrace.of(result))
            return result

    return wrapped
