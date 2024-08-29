from asyncio import (
    Lock,
    iscoroutinefunction,
    sleep,
)
from collections import deque
from collections.abc import Callable, Coroutine
from datetime import timedelta
from time import monotonic
from typing import cast, overload

from draive.utils.mimic import mimic_function

__all__ = [
    "throttle",
]


@overload
def throttle[**Args, Result](
    function: Callable[Args, Coroutine[None, None, Result]],
    /,
) -> Callable[Args, Coroutine[None, None, Result]]: ...


@overload
def throttle[**Args, Result](
    *,
    limit: int = 1,
    period: timedelta | float = 1,
) -> Callable[
    [Callable[Args, Coroutine[None, None, Result]]], Callable[Args, Coroutine[None, None, Result]]
]: ...


def throttle[**Args, Result](
    function: Callable[Args, Coroutine[None, None, Result]] | None = None,
    *,
    limit: int = 1,
    period: timedelta | float = 1,
) -> (
    Callable[
        [Callable[Args, Coroutine[None, None, Result]]],
        Callable[Args, Coroutine[None, None, Result]],
    ]
    | Callable[Args, Coroutine[None, None, Result]]
):
    """\
    Simple throttle for function calls with custom limit and period time. \
    Works only for async functions. \
    It is not allowed to be used on class or instance methods. \
    This wrapper is not thread safe.

    Parameters
    ----------
    function: Callable[Args, Coroutine[None, None, Result]]
        function to wrap in throttle
    limit: int
        limit of executions in given period, if no period was specified
        it is number of concurrent executions instead, default is 1
    period: timedelta | float | None
        period time (in seconds by default) during which the limit resets, default is 1 second

    Returns
    -------
    Callable[[Callable[Args, Coroutine[None, None, Result]]], Callable[Args, Coroutine[None, None, Result]]] \
    | Callable[Args, Coroutine[None, None, Result]]
        provided function wrapped in throttle
    """  # noqa: E501

    def _wrap(
        function: Callable[Args, Coroutine[None, None, Result]],
    ) -> Callable[Args, Coroutine[None, None, Result]]:
        assert iscoroutinefunction(function)  # nosec: B101
        return cast(
            Callable[Args, Coroutine[None, None, Result]],
            _AsyncThrottle(
                function,
                limit=limit,
                period=period,
            ),
        )

    if function := function:
        return _wrap(function)

    else:
        return _wrap


class _AsyncThrottle[**Args, Result]:
    def __init__(
        self,
        function: Callable[Args, Coroutine[None, None, Result]],
        /,
        limit: int,
        period: timedelta | float,
    ) -> None:
        self._function: Callable[Args, Coroutine[None, None, Result]] = function
        self._entries: deque[float] = deque()
        self._lock: Lock = Lock()
        self._limit: int = limit
        self._period: float
        match period:
            case timedelta() as delta:
                self._period = delta.total_seconds()

            case period_seconds:
                self._period = period_seconds

        # mimic function attributes if able
        mimic_function(function, within=self)

    async def __call__(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        async with self._lock:
            time_now: float = monotonic()
            while self._entries:  # cleanup old entries
                if self._entries[0] + self._period <= time_now:
                    self._entries.popleft()

                else:
                    break

            if len(self._entries) >= self._limit:
                await sleep(self._entries[0] - time_now)

            self._entries.append(monotonic())

        return await self._function(*args, **kwargs)
