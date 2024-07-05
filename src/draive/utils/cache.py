from asyncio import AbstractEventLoop, Task, get_running_loop, iscoroutinefunction, shield
from collections import OrderedDict
from collections.abc import Callable, Coroutine, Hashable
from functools import _make_key, partial  # pyright: ignore[reportPrivateUsage]
from time import monotonic
from typing import NamedTuple, cast, overload
from weakref import ref

from draive.utils.mimic import mimic_function

__all__ = [
    "cache",
]


@overload
def cache[**Args, Result](
    function: Callable[Args, Result],
    /,
) -> Callable[Args, Result]: ...


@overload
def cache[**Args, Result](
    *,
    limit: int = 1,
    expiration: float | None = None,
) -> Callable[[Callable[Args, Result]], Callable[Args, Result]]: ...


def cache[**Args, Result](
    function: Callable[Args, Result] | None = None,
    *,
    limit: int = 1,
    expiration: float | None = None,
) -> Callable[[Callable[Args, Result]], Callable[Args, Result]] | Callable[Args, Result]:
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

    def _wrap(function: Callable[Args, Result]) -> Callable[Args, Result]:
        if iscoroutinefunction(function):
            return cast(
                Callable[Args, Result],
                _AsyncCache(
                    function,
                    limit=limit,
                    expiration=expiration,
                ),
            )
        else:
            return cast(
                Callable[Args, Result],
                _SyncCache(
                    function,
                    limit=limit,
                    expiration=expiration,
                ),
            )

    if function := function:
        return _wrap(function)
    else:
        return _wrap


class _CacheEntry[Entry](NamedTuple):
    value: Entry
    expire: float | None


class _SyncCache[**Args, Result]:
    def __init__(
        self,
        function: Callable[Args, Result],
        /,
        limit: int,
        expiration: float | None,
    ) -> None:
        self._function: Callable[Args, Result] = function
        self._cached: OrderedDict[Hashable, _CacheEntry[Result]] = OrderedDict()
        self._limit: int = limit
        if expiration := expiration:

            def next_expire_time() -> float | None:
                return monotonic() + expiration
        else:

            def next_expire_time() -> float | None:
                return None

        self._next_expire_time: Callable[[], float | None] = next_expire_time

        # mimic function attributes if able
        mimic_function(function, within=self)

    def __get__(
        self,
        instance: object,
        owner: type | None = None,
        /,
    ) -> Callable[Args, Result]:
        if owner is None:
            return self

        else:
            return mimic_function(
                self._function,
                within=partial(
                    self.__method_call__,
                    instance,
                ),
            )

    def __call__(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        key: Hashable = _make_key(
            args=args,
            kwds=kwargs,
            typed=True,
        )

        match self._cached.get(key):
            case None:
                pass

            case entry:
                if (expire := entry[1]) and expire < monotonic():
                    # if still running let it complete if able
                    del self._cached[key]  # continue the same way as if empty
                else:
                    self._cached.move_to_end(key)
                    return entry[0]

        result: Result = self._function(*args, **kwargs)
        self._cached[key] = _CacheEntry(
            value=result,
            expire=self._next_expire_time(),
        )
        if len(self._cached) > self._limit:
            # if still running let it complete if able
            self._cached.popitem(last=False)
        return result

    def __method_call__(
        self,
        __method_self: object,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        key: Hashable = _make_key(
            args=(ref(__method_self), *args),
            kwds=kwargs,
            typed=True,
        )

        match self._cached.get(key):
            case None:
                pass

            case entry:
                if (expire := entry[1]) and expire < monotonic():
                    # if still running let it complete if able
                    del self._cached[key]  # continue the same way as if empty
                else:
                    self._cached.move_to_end(key)
                    return entry[0]

        result: Result = self._function(__method_self, *args, **kwargs)  # pyright: ignore[reportUnknownVariableType, reportCallIssue]
        self._cached[key] = _CacheEntry(
            value=result,  # pyright: ignore[reportUnknownArgumentType]
            expire=self._next_expire_time(),
        )
        if len(self._cached) > self._limit:
            # if still running let it complete if able
            self._cached.popitem(last=False)
        return result  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]


class _AsyncCache[**Args, Result]:
    def __init__(
        self,
        function: Callable[Args, Coroutine[None, None, Result]],
        /,
        limit: int,
        expiration: float | None,
    ) -> None:
        self._function: Callable[Args, Coroutine[None, None, Result]] = function
        self._cached: OrderedDict[Hashable, _CacheEntry[Task[Result]]] = OrderedDict()
        self._limit: int = limit
        if expiration := expiration:

            def next_expire_time() -> float | None:
                return monotonic() + expiration
        else:

            def next_expire_time() -> float | None:
                return None

        self._next_expire_time: Callable[[], float | None] = next_expire_time

        # mimic function attributes if able
        mimic_function(function, within=self)

    def __get__(
        self,
        instance: object,
        owner: type | None = None,
        /,
    ) -> Callable[Args, Coroutine[None, None, Result]]:
        if owner is None:
            return self
        else:
            return mimic_function(
                self._function,
                within=partial(
                    self.__method_call__,
                    instance,
                ),
            )

    async def __call__(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        loop: AbstractEventLoop = get_running_loop()
        key: Hashable = _make_key(
            args=args,
            kwds=kwargs,
            typed=True,
        )

        match self._cached.get(key):
            case None:
                pass

            case entry:
                if (expire := entry[1]) and expire < monotonic():
                    # if still running let it complete if able
                    del self._cached[key]  # continue the same way as if empty
                else:
                    self._cached.move_to_end(key)
                    return await shield(entry[0])

        task: Task[Result] = loop.create_task(self._function(*args, **kwargs))  # pyright: ignore[reportCallIssue]
        self._cached[key] = _CacheEntry(
            value=task,
            expire=self._next_expire_time(),
        )
        if len(self._cached) > self._limit:
            # if still running let it complete if able
            self._cached.popitem(last=False)
        return await shield(task)

    async def __method_call__(
        self,
        __method_self: object,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        loop: AbstractEventLoop = get_running_loop()
        key: Hashable = _make_key(
            args=(ref(__method_self), *args),
            kwds=kwargs,
            typed=True,
        )

        match self._cached.get(key):
            case None:
                pass

            case entry:
                if (expire := entry[1]) and expire < monotonic():
                    # if still running let it complete if able
                    del self._cached[key]  # continue the same way as if empty
                else:
                    self._cached.move_to_end(key)
                    return await shield(entry[0])

        task: Task[Result] = loop.create_task(
            self._function(__method_self, *args, **kwargs),  # pyright: ignore[reportCallIssue, reportUnknownArgumentType]
        )
        self._cached[key] = _CacheEntry(
            value=task,
            expire=self._next_expire_time(),
        )
        if len(self._cached) > self._limit:
            # if still running let it complete if able
            self._cached.popitem(last=False)
        return await shield(task)
