from asyncio import sleep
from collections.abc import Callable, Coroutine
from functools import wraps
from random import uniform
from typing import Any, ParamSpec, TypeVar

from draive.scope import ctx

__all__ = [
    "autoretry",
]

_Args_T = ParamSpec(name="_Args_T")

_Result_T = TypeVar(name="_Result_T")


def autoretry(
    limit: int,
    delay: tuple[float, float] | float | None = None,
    fallback: _Result_T | None = None,
) -> Callable[
    [Callable[_Args_T, Coroutine[Any, Any, _Result_T]]],
    Callable[_Args_T, Coroutine[Any, Any, _Result_T]],
]:
    assert limit > 0, "Retries limit has to be at least one"  # nosec: B101

    def wrapped(
        function: Callable[_Args_T, Coroutine[Any, Any, _Result_T]],
    ) -> Callable[_Args_T, Coroutine[Any, Any, _Result_T]]:
        @wraps(function)
        async def with_autoretry(*args: _Args_T.args, **kwargs: _Args_T.kwargs) -> _Result_T:
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

        return with_autoretry

    return wrapped
