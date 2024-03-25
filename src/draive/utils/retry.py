from asyncio import iscoroutinefunction, sleep
from collections.abc import Callable, Coroutine
from functools import wraps
from random import uniform
from typing import Any, ParamSpec, TypeVar, cast, overload

from draive.scope import ctx

__all__ = [
    "auto_retry",
]

_Args = ParamSpec(name="_Args")
_Result = TypeVar(name="_Result")


@overload
def auto_retry(
    function: Callable[_Args, _Result],
    /,
) -> Callable[_Args, _Result]:
    ...


@overload
def auto_retry(
    *,
    limit: int = 1,
    delay: tuple[float, float] | float | None = None,
    fallback: _Result | None = None,
) -> Callable[[Callable[_Args, _Result]], Callable[_Args, _Result]]:
    ...


def auto_retry(
    function: Callable[_Args, _Result] | None = None,
    *,
    limit: int = 1,
    delay: tuple[float, float] | float | None = None,
    fallback: _Result | None = None,
) -> Callable[[Callable[_Args, _Result]], Callable[_Args, _Result]] | Callable[_Args, _Result]:
    """\
    Simple on fail retry function wrapper. \
    Works for both sync and async functions. \
    It is not allowed to be used on class methods. \
    This wrapper is not thread safe.

    Parameters
    ----------
    function: Callable[_Args, _Result]
        function to wrap in auto retry, either sync or async
    limit: int
        limit of retries, default is 1
    delay: tuple[float, float] | float | None
        retry delay time in seconds, tuple allows to provide time range to use, \
        default is None (no delay)
    fallback: _Result | None
        optional fallback result to use after all retry attempts failure

    Returns
    -------
    Callable[[Callable[_Args, _Result]], Callable[_Args, _Result]] | Callable[_Args, _Result]
        provided function wrapped in auto retry
    """

    def _wrap(function: Callable[_Args, _Result], /) -> Callable[_Args, _Result]:
        if iscoroutinefunction(function):
            return cast(
                Callable[_Args, _Result],
                _wrap_async(
                    function,
                    limit=limit,
                    delay=delay,
                    fallback=fallback,
                ),
            )
        else:
            return _wrap_sync(
                function,
                limit=limit,
                delay=delay,
                fallback=fallback,
            )

    if function := function:
        return _wrap(function)
    else:
        return _wrap


def _wrap_sync(
    function: Callable[_Args, _Result],
    *,
    limit: int,
    delay: tuple[float, float] | float | None,
    fallback: _Result | None,
) -> Callable[_Args, _Result]:
    assert limit > 0, "Limit has to be greater than zero"  # nosec: B101
    assert delay is None, "Delay is not supported in sync wrapper"  # nosec: B101

    @wraps(function)
    def wrapped(
        *args: _Args.args,
        **kwargs: _Args.kwargs,
    ) -> _Result:
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

                elif result := fallback:
                    ctx.log_error(
                        "Using fallback value for %s which failed due to an error: %s",
                        function.__name__,
                        exc,
                    )
                    return result

                else:
                    raise exc

    return wrapped


def _wrap_async(
    function: Callable[_Args, Coroutine[Any, Any, _Result]],
    *,
    limit: int,
    delay: tuple[float, float] | float | None,
    fallback: _Result | None,
) -> Callable[_Args, Coroutine[Any, Any, _Result]]:
    assert limit > 0, "Limit has to be greater than zero"  # nosec: B101

    @wraps(function)
    async def wrapped(
        *args: _Args.args,
        **kwargs: _Args.kwargs,
    ) -> _Result:
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

                elif result := fallback:
                    ctx.log_error(
                        "Using fallback value for %s which failed due to an error: %s",
                        function.__name__,
                        exc,
                    )
                    return result

                else:
                    raise exc

    return wrapped
