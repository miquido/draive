from asyncio import AbstractEventLoop, Task, get_running_loop, iscoroutinefunction, shield
from collections import OrderedDict
from collections.abc import Callable, Coroutine, Hashable
from functools import _make_key, partial  # pyright: ignore[reportPrivateUsage]
from time import monotonic
from typing import Any, Generic, NamedTuple, ParamSpec, TypeVar, cast, overload
from weakref import ref

__all__ = [
    "cache",
]

_Args_T = ParamSpec(
    name="_Args_T",
)
_Result_T = TypeVar(
    name="_Result_T",
)
_Entry_T = TypeVar(
    name="_Entry_T",
)


@overload
def cache(
    function: Callable[_Args_T, _Result_T],
    /,
) -> Callable[_Args_T, _Result_T]: ...


@overload
def cache(
    *,
    limit: int = 1,
    expiration: float | None = None,
) -> Callable[[Callable[_Args_T, _Result_T]], Callable[_Args_T, _Result_T]]: ...


def cache(
    function: Callable[_Args_T, _Result_T] | None = None,
    *,
    limit: int = 1,
    expiration: float | None = None,
) -> (
    Callable[[Callable[_Args_T, _Result_T]], Callable[_Args_T, _Result_T]]
    | Callable[_Args_T, _Result_T]
):
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

    def _wrap(function: Callable[_Args_T, _Result_T]) -> Callable[_Args_T, _Result_T]:
        if iscoroutinefunction(function):
            return cast(
                Callable[_Args_T, _Result_T],
                _AsyncCache(
                    function,
                    limit=limit,
                    expiration=expiration,
                ),
            )
        else:
            return cast(
                Callable[_Args_T, _Result_T],
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


class _CacheEntry(Generic[_Entry_T], NamedTuple):
    value: _Entry_T
    expire: float | None


class _SyncCache(Generic[_Args_T, _Result_T]):
    def __init__(
        self,
        function: Callable[_Args_T, _Result_T],
        /,
        limit: int,
        expiration: float | None,
    ) -> None:
        self._function: Callable[_Args_T, _Result_T] = function
        self._cached: OrderedDict[Hashable, _CacheEntry[_Result_T]] = OrderedDict()
        self._limit: int = limit
        if expiration := expiration:

            def next_expire_time() -> float | None:
                return monotonic() + expiration
        else:

            def next_expire_time() -> float | None:
                return None

        self._next_expire_time: Callable[[], float | None] = next_expire_time

        # mimic function attributes if able
        _mimic(function, within=self)

    def __get__(
        self,
        instance: object,
        owner: type | None = None,
        /,
    ) -> Callable[_Args_T, _Result_T]:
        if owner is None:
            return self
        else:
            return _mimic(
                self._function,
                within=partial(
                    self.__method_call__,
                    instance,
                ),
            )

    def __call__(
        self,
        *args: _Args_T.args,
        **kwargs: _Args_T.kwargs,
    ) -> _Result_T:
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

        result: _Result_T = self._function(*args, **kwargs)
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
        *args: _Args_T.args,
        **kwargs: _Args_T.kwargs,
    ) -> _Result_T:
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

        result: _Result_T = self._function(__method_self, *args, **kwargs)  # pyright: ignore[reportUnknownVariableType, reportCallIssue]
        self._cached[key] = _CacheEntry(
            value=result,  # pyright: ignore[reportUnknownArgumentType]
            expire=self._next_expire_time(),
        )
        if len(self._cached) > self._limit:
            # if still running let it complete if able
            self._cached.popitem(last=False)
        return result  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]


class _AsyncCache(Generic[_Args_T, _Result_T]):
    def __init__(
        self,
        function: Callable[_Args_T, Coroutine[Any, Any, _Result_T]],
        /,
        limit: int,
        expiration: float | None,
    ) -> None:
        self._function: Callable[_Args_T, Coroutine[Any, Any, _Result_T]] = function
        self._cached: OrderedDict[Hashable, _CacheEntry[Task[_Result_T]]] = OrderedDict()
        self._limit: int = limit
        if expiration := expiration:

            def next_expire_time() -> float | None:
                return monotonic() + expiration
        else:

            def next_expire_time() -> float | None:
                return None

        self._next_expire_time: Callable[[], float | None] = next_expire_time

        # mimic function attributes if able
        _mimic(function, within=self)

    def __get__(
        self,
        instance: object,
        owner: type | None = None,
        /,
    ) -> Callable[_Args_T, Coroutine[Any, Any, _Result_T]]:
        if owner is None:
            return self
        else:
            return _mimic(
                self._function,
                within=partial(
                    self.__method_call__,
                    instance,
                ),
            )

    async def __call__(
        self,
        *args: _Args_T.args,
        **kwargs: _Args_T.kwargs,
    ) -> _Result_T:
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

        task: Task[_Result_T] = loop.create_task(self._function(*args, **kwargs))  # pyright: ignore[reportCallIssue]
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
        *args: _Args_T.args,
        **kwargs: _Args_T.kwargs,
    ) -> _Result_T:
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

        task: Task[_Result_T] = loop.create_task(
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


def _mimic(
    function: Callable[_Args_T, _Result_T],
    *,
    within: Callable[..., Any],
) -> Callable[_Args_T, _Result_T]:
    # mimic function attributes if able
    for attribute in [
        "__module__",
        "__name__",
        "__qualname__",
        "__doc__",
        "__annotations__",
    ]:
        try:
            setattr(
                within,
                attribute,
                getattr(
                    function,
                    attribute,
                ),
            )

        except AttributeError:
            pass
    try:
        within.__dict__.update(function.__dict__)
    except AttributeError:
        pass

    return cast(
        Callable[_Args_T, _Result_T],
        within,
    )
