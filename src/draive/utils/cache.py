from collections.abc import Callable
from functools import lru_cache, wraps
from time import time
from typing import ParamSpec, TypeVar, overload

__all__ = [
    "cache",
]

_Args = ParamSpec(
    name="_Args",
)
_Result = TypeVar(
    name="_Result",
)


@overload
def cache(
    function: Callable[_Args, _Result],
    /,
) -> Callable[_Args, _Result]:
    ...


@overload
def cache(
    *,
    expiration_time: int,
) -> Callable[[Callable[_Args, _Result]], Callable[_Args, _Result]]:
    ...


@overload
def cache(
    *,
    limit: int | None,
) -> Callable[[Callable[_Args, _Result]], Callable[_Args, _Result]]:
    ...


@overload
def cache(
    *,
    limit: int | None,
    expiration_time: int,
) -> Callable[[Callable[_Args, _Result]], Callable[_Args, _Result]]:
    ...


def cache(
    function: Callable[_Args, _Result] | None = None,
    *,
    limit: int | None = 1,
    expiration_time: int | None = None,
) -> Callable[[Callable[_Args, _Result]], Callable[_Args, _Result]] | Callable[_Args, _Result]:
    if expiration_time := expiration_time:

        def wrap(function: Callable[_Args, _Result]) -> Callable[_Args, _Result]:
            @wraps(function)
            def expiring(
                *args: _Args.args,
                **kwargs: _Args.kwargs,
            ) -> _Result:
                @lru_cache(maxsize=limit)
                def cached(
                    __ttl: int,
                    *args: _Args.args,
                    **kwargs: _Args.kwargs,
                ) -> _Result:
                    return function(*args, **kwargs)

                return cached(
                    *args,
                    __ttl=round(time() / expiration_time),
                    **kwargs,
                )

            return expiring

    else:

        def wrap(function: Callable[_Args, _Result]) -> Callable[_Args, _Result]:
            @wraps(function)
            def indefinite(
                *args: _Args.args,
                **kwargs: _Args.kwargs,
            ) -> _Result:
                @lru_cache(maxsize=limit)
                def cached(
                    *args: _Args.args,
                    **kwargs: _Args.kwargs,
                ) -> _Result:
                    return function(*args, **kwargs)

                return cached(
                    *args,
                    **kwargs,
                )

            return indefinite

    if function := function:
        return wrap(function=function)
    else:
        return wrap
