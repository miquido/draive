from asyncio import iscoroutinefunction
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec, TypeVar, cast

from draive.helpers import mimic_function
from draive.metrics import ArgumentsTrace, ResultTrace
from draive.scope import ctx

__all__ = [
    "traced",
]

_Args_T = ParamSpec(
    name="_Args_T",
)
_Result_T = TypeVar(
    name="_Result_T",
)


def traced(
    function: Callable[_Args_T, _Result_T],
    /,
) -> Callable[_Args_T, _Result_T]:
    if iscoroutinefunction(function):
        return cast(
            Callable[_Args_T, _Result_T],
            _traced_async(function),
        )
    else:
        return _traced_sync(function)


def _traced_sync(
    function: Callable[_Args_T, _Result_T],
    /,
) -> Callable[_Args_T, _Result_T]:
    label: str = function.__name__

    def wrapped(
        *args: _Args_T.args,
        **kwargs: _Args_T.kwargs,
    ) -> _Result_T:
        with ctx.nested(
            label,
            metrics=[ArgumentsTrace.of(*args, **kwargs)],
        ):
            result: _Result_T = function(*args, **kwargs)
            ctx.record(ResultTrace.of(result))
            return result

    return mimic_function(function, within=wrapped)


def _traced_async(
    function: Callable[_Args_T, Coroutine[Any, Any, _Result_T]],
    /,
) -> Callable[_Args_T, Coroutine[Any, Any, _Result_T]]:
    label: str = function.__name__

    async def wrapped(
        *args: _Args_T.args,
        **kwargs: _Args_T.kwargs,
    ) -> _Result_T:
        with ctx.nested(
            label,
            metrics=[ArgumentsTrace.of(*args, **kwargs)],
        ):
            result: _Result_T = await function(*args, **kwargs)
            ctx.record(ResultTrace.of(result))
            return result

    return mimic_function(function, within=wrapped)
