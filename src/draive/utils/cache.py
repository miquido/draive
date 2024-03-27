from asyncio import AbstractEventLoop, Task, get_running_loop, iscoroutinefunction, shield
from collections import OrderedDict
from collections.abc import Callable, Coroutine, Hashable
from functools import _make_key, wraps  # pyright: ignore[reportPrivateUsage]
from time import monotonic
from typing import Any, Generic, NamedTuple, ParamSpec, TypeVar, cast, overload

__all__ = [
    "cache",
]

_Args = ParamSpec(
    name="_Args",
)
_Result = TypeVar(
    name="_Result",
)
_Entry = TypeVar(
    name="_Entry",
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
    limit: int = 1,
    expiration: float | None = None,
) -> Callable[[Callable[_Args, _Result]], Callable[_Args, _Result]]:
    ...


def cache(
    function: Callable[_Args, _Result] | None = None,
    *,
    limit: int = 1,
    expiration: float | None = None,
) -> Callable[[Callable[_Args, _Result]], Callable[_Args, _Result]] | Callable[_Args, _Result]:
    """\
    Simple lru function result cache with optional expire time. \
    Works for both sync and async functions. \
    It is not allowed to be used on class methods. \
    This wrapper is not thread safe.

    Parameters
    ----------
    function: Callable[_Args, _Result]
        function to wrap in cache, either sync or async
    limit: int
        limit of cache entries to keep, default is 1
    expiration: float | None
        entries expiration time in seconds, default is None (not expiring)

    Returns
    -------
    Callable[[Callable[_Args, _Result]], Callable[_Args, _Result]] | Callable[_Args, _Result]
        provided function wrapped in cache
    """

    def _wrap(function: Callable[_Args, _Result], /) -> Callable[_Args, _Result]:
        if hasattr(function, "__self__"):
            raise RuntimeError("Method cache is not supported yet")

        if iscoroutinefunction(function):
            return cast(
                Callable[_Args, _Result],
                _wrap_async(
                    function,
                    limit=limit,
                    expiration=expiration,
                ),
            )
        else:
            return _wrap_sync(
                function,
                limit=limit,
                expiration=expiration,
            )

    if function := function:
        return _wrap(function)
    else:
        return _wrap


class _CacheEntry(Generic[_Entry], NamedTuple):
    value: _Entry
    expire: float | None


def _wrap_sync(
    function: Callable[_Args, _Result],
    *,
    limit: int,
    expiration: float | None,
) -> Callable[_Args, _Result]:
    assert limit > 0, "Limit has to be greater than zero"  # nosec: B101
    cached: OrderedDict[Hashable, _CacheEntry[_Result]] = OrderedDict()

    @wraps(function)
    def wrapped(
        *args: _Args.args,
        **kwargs: _Args.kwargs,
    ) -> _Result:
        key: Hashable = _make_key(args=args, kwds=kwargs, typed=True)
        match cached.get(key):
            case None:
                pass

            case entry:
                if (expire := entry[1]) and expire < monotonic():
                    del cached[key]  # continue the same way as if empty
                else:
                    cached.move_to_end(key)
                    return entry[0]

        result: _Result = function(*args, **kwargs)
        cached[key] = _CacheEntry(
            value=result,
            expire=monotonic() + expiration if expiration else None,
        )
        if len(cached) > limit:
            _, entry = cached.popitem(last=False)
        return result

    return wrapped


def _wrap_async(
    function: Callable[_Args, Coroutine[Any, Any, _Result]],
    *,
    limit: int = 1,
    expiration: float | None,
) -> Callable[_Args, Coroutine[Any, Any, _Result]]:
    assert limit > 0, "Limit has to be greater than zero"  # nosec: B101
    cached: OrderedDict[Hashable, _CacheEntry[Task[_Result]]] = OrderedDict()

    @wraps(function)
    async def wrapped(
        *args: _Args.args,
        **kwargs: _Args.kwargs,
    ) -> _Result:
        loop: AbstractEventLoop = get_running_loop()
        key: Hashable = _make_key(args=args, kwds=kwargs, typed=True)
        match cached.get(key):
            case None:
                pass

            case entry:
                if (expire := entry[1]) and expire < monotonic():
                    # if still running let it complete if able
                    del cached[key]  # continue the same way as if empty
                else:
                    cached.move_to_end(key)
                    return await shield(entry[0])

        task: Task[_Result] = loop.create_task(function(*args, **kwargs))
        cached[key] = _CacheEntry(
            value=task,
            expire=monotonic() + expiration if expiration else None,
        )
        if len(cached) > limit:
            # if still running let it complete if able
            cached.popitem(last=False)
        return await shield(task)

    return wrapped
