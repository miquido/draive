from asyncio import CancelledError, iscoroutinefunction, sleep
from collections.abc import Callable, Coroutine
from typing import cast, overload

from draive.scope import ctx
from draive.types import RateLimitError
from draive.utils import mimic_function

__all__ = [
    "auto_retry",
]


@overload
def auto_retry[**Args, Result](
    function: Callable[Args, Result],
    /,
) -> Callable[Args, Result]:
    """\
    Function wrapper retrying the wrapped function again on fail. \
    Works for both sync and async functions. \
    It is not allowed to be used on class methods. \
    This wrapper is not thread safe.

    Parameters
    ----------
    function: Callable[_Args_T, _Result_T]
        function to wrap in auto retry, either sync or async.

    Returns
    -------
    Callable[_Args_T, _Result_T]
        provided function wrapped in auto retry with default configuration.
    """


@overload
def auto_retry[**Args, Result](
    *,
    limit: int = 1,
    delay: Callable[[int], float] | float | None = None,
    catching: set[type[Exception]] | tuple[type[Exception], ...] | type[Exception] = Exception,
) -> Callable[[Callable[Args, Result]], Callable[Args, Result]]:
    """\
    Function wrapper retrying the wrapped function again on fail. \
    Works for both sync and async functions. \
    It is not allowed to be used on class methods. \
    This wrapper is not thread safe.

    Parameters
    ----------
    limit: int
        limit of retries, default is 1
    delay: Callable[[], float] | float | None
        retry delay time in seconds, either concrete value or a function producing it, \
        default is None (no delay)
    catching: set[type[Exception]] | type[Exception] | None
        Exception types that are triggering auto retry. Retry will trigger only when \
        exceptions of matching types (including subclasses) will occur. CancelledError \
        will be always propagated even if specified explicitly.
        Default is Exception - all subclasses of Exception will be handled.

    Returns
    -------
    Callable[[Callable[_Args_T, _Result_T]], Callable[_Args_T, _Result_T]]
        function wrapper for adding auto retry
    """


def auto_retry[**Args, Result](
    function: Callable[Args, Result] | None = None,
    *,
    limit: int = 1,
    delay: Callable[[int], float] | float | None = None,
    catching: set[type[Exception]] | tuple[type[Exception], ...] | type[Exception] = Exception,
) -> Callable[[Callable[Args, Result]], Callable[Args, Result]] | Callable[Args, Result]:
    """\
    Function wrapper retrying the wrapped function again on fail. \
    Works for both sync and async functions. \
    It is not allowed to be used on class methods. \
    This wrapper is not thread safe.

    Parameters
    ----------
    function: Callable[_Args_T, _Result_T]
        function to wrap in auto retry, either sync or async.
    limit: int
        limit of retries, default is 1
    delay: Callable[[], float] | float | None
        retry delay time in seconds, either concrete value or a function producing it, \
        default is None (no delay)
    catching: set[type[Exception]] | type[Exception] | None
        Exception types that are triggering auto retry. Retry will trigger only when \
        exceptions of matching types (including subclasses) will occur. CancelledError \
        will be always propagated even if specified explicitly.
        Default is Exception - all subclasses of Exception will be handled.

    Returns
    -------
    Callable[[Callable[_Args_T, _Result_T]], Callable[_Args_T, _Result_T]] | \
    Callable[_Args_T, _Result_T]
        function wrapper for adding auto retry or a wrapped function
    """

    def _wrap(
        function: Callable[Args, Result],
        /,
    ) -> Callable[Args, Result]:
        if iscoroutinefunction(function):
            return cast(
                Callable[Args, Result],
                _wrap_async(
                    function,
                    limit=limit,
                    delay=delay,
                    catching=catching if isinstance(catching, set | tuple) else {catching},
                ),
            )
        else:
            return _wrap_sync(
                function,
                limit=limit,
                delay=delay,
                catching=catching if isinstance(catching, set | tuple) else {catching},
            )

    if function := function:
        return _wrap(function)
    else:
        return _wrap


def _wrap_sync[**Args, Result](
    function: Callable[Args, Result],
    *,
    limit: int,
    delay: Callable[[int], float] | float | None,
    catching: set[type[Exception]] | tuple[type[Exception], ...],
) -> Callable[Args, Result]:
    assert limit > 0, "Limit has to be greater than zero"  # nosec: B101
    assert delay is None, "Delay is not supported in sync wrapper"  # nosec: B101

    @mimic_function(function)
    def wrapped(
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        attempt: int = 0
        while True:
            try:
                return function(*args, **kwargs)
            except CancelledError as exc:
                raise exc

            except Exception as exc:
                if attempt < limit and any(isinstance(exc, exception) for exception in catching):
                    attempt += 1
                    ctx.log_error(
                        "Attempting to retry %s which failed due to an error: %s",
                        function.__name__,
                        exc,
                    )

                else:
                    raise exc

    return wrapped


def _wrap_async[**Args, Result](
    function: Callable[Args, Coroutine[None, None, Result]],
    *,
    limit: int,
    delay: Callable[[int], float] | float | None,
    catching: set[type[Exception]] | tuple[type[Exception], ...],
) -> Callable[Args, Coroutine[None, None, Result]]:
    assert limit > 0, "Limit has to be greater than zero"  # nosec: B101

    @mimic_function(function)
    async def wrapped(
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        attempt: int = 0
        while True:
            try:
                return await function(*args, **kwargs)
            except CancelledError as exc:
                raise exc

            except RateLimitError as exc:
                if attempt < limit and any(isinstance(exc, exception) for exception in catching):
                    attempt += 1
                    ctx.log_warning(
                        "Attempting to retry %s after %.2fs which failed due to rate limit",
                        function.__name__,
                        exc.retry_after,
                    )
                    await sleep(exc.retry_after)

                else:
                    raise exc

            except Exception as exc:
                if attempt < limit and any(isinstance(exc, exception) for exception in catching):
                    attempt += 1
                    ctx.log_error(
                        "Attempting to retry %s which failed due to an error",
                        function.__name__,
                        exception=exc,
                    )

                    match delay:
                        case None:
                            continue

                        case float(strict):
                            await sleep(delay=strict)

                        case make_delay:  # type: Callable[[], float]
                            await sleep(delay=make_delay(attempt))  # pyright: ignore[reportCallIssue, reportUnknownArgumentType]

                else:
                    raise exc

    return wrapped
