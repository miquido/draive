from asyncio import iscoroutinefunction, sleep
from collections.abc import Callable, Coroutine
from functools import wraps
from random import uniform
from typing import Any, ParamSpec, TypeVar, cast, overload

from draive.scope import ctx

__all__ = [
    "auto_retry",
]

_Args_T = ParamSpec(name="_Args_T")
_Result_T = TypeVar(name="_Result_T")


@overload
def auto_retry(
    function: Callable[_Args_T, _Result_T],
    /,
) -> Callable[_Args_T, _Result_T]:
    ...


@overload
def auto_retry(
    *,
    limit: int = 1,
    delay: tuple[float, float] | float | None = None,
) -> Callable[[Callable[_Args_T, _Result_T]], Callable[_Args_T, _Result_T]]:
    ...


def auto_retry(
    function: Callable[_Args_T, _Result_T] | None = None,
    *,
    limit: int = 1,
    delay: tuple[float, float] | float | None = None,
) -> (
    Callable[[Callable[_Args_T, _Result_T]], Callable[_Args_T, _Result_T]]
    | Callable[_Args_T, _Result_T]
):
    """\
    Simple on fail retry function wrapper. \
    Works for both sync and async functions. \
    It is not allowed to be used on class methods. \
    This wrapper is not thread safe.

    Parameters
    ----------
    function: Callable[_Args_T, _Result_T]
        function to wrap in auto retry, either sync or async
    limit: int
        limit of retries, default is 1
    delay: tuple[float, float] | float | None
        retry delay time in seconds, tuple allows to provide time range to use, \
        default is None (no delay)

    Returns
    -------
    Callable[[Callable[_Args_T, _Result_T]], Callable[_Args_T, _Result_T]]
    | Callable[_Args_T, _Result_T]
        provided function wrapped in auto retry
    """

    def _wrap(
        function: Callable[_Args_T, _Result_T],
        /,
    ) -> Callable[_Args_T, _Result_T]:
        if hasattr(function, "__self__"):
            raise RuntimeError("Method auto_retry is not supported yet")

        if iscoroutinefunction(function):
            return cast(
                Callable[_Args_T, _Result_T],
                _wrap_async(
                    function,
                    limit=limit,
                    delay=delay,
                ),
            )
        else:
            return _wrap_sync(
                function,
                limit=limit,
                delay=delay,
            )

    if function := function:
        return _wrap(function)
    else:
        return _wrap


def _wrap_sync(
    function: Callable[_Args_T, _Result_T],
    *,
    limit: int,
    delay: tuple[float, float] | float | None,
) -> Callable[_Args_T, _Result_T]:
    assert limit > 0, "Limit has to be greater than zero"  # nosec: B101
    assert delay is None, "Delay is not supported in sync wrapper"  # nosec: B101

    @wraps(function)
    def wrapped(
        *args: _Args_T.args,
        **kwargs: _Args_T.kwargs,
    ) -> _Result_T:
        attempt: int = 0
        while True:
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                if attempt < limit:
                    attempt += 1
                    ctx.log_error(
                        "Attempting to retry %s which failed due to an error: %s",
                        function.__name__,
                        exc,
                    )

                else:
                    raise exc

    return wrapped


def _wrap_async(
    function: Callable[_Args_T, Coroutine[Any, Any, _Result_T]],
    *,
    limit: int,
    delay: tuple[float, float] | float | None,
) -> Callable[_Args_T, Coroutine[Any, Any, _Result_T]]:
    assert limit > 0, "Limit has to be greater than zero"  # nosec: B101

    @wraps(function)
    async def wrapped(
        *args: _Args_T.args,
        **kwargs: _Args_T.kwargs,
    ) -> _Result_T:
        attempt: int = 0
        while True:
            try:
                return await function(*args, **kwargs)
            except Exception as exc:
                if attempt < limit:
                    attempt += 1
                    ctx.log_error(
                        "Attempting to retry %s which failed due to an error: %s",
                        function.__name__,
                        exc,
                    )

                    match delay:
                        case (float() as lower, float() as upper):
                            await sleep(delay=uniform(lower, upper))  # nosec: B311

                        case float() as strict:
                            await sleep(delay=strict)

                        case _:
                            continue

                else:
                    raise exc

    return wrapped
