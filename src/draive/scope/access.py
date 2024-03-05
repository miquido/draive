from asyncio import Task, TaskGroup
from collections.abc import Callable, Coroutine, Iterable
from contextvars import Context, ContextVar, Token, copy_context
from functools import wraps
from logging import Logger
from types import TracebackType
from typing import Any, ParamSpec, TypeVar, final

from draive.scope.dependencies import (
    ScopeDependencies,
    _ScopeDependencies_Var,  # pyright: ignore[reportPrivateUsage]
    _ScopeDependency_T,  # pyright: ignore[reportPrivateUsage]
)
from draive.scope.errors import MissingScopeContext
from draive.scope.metrics import (
    ScopeMetric,
    ScopeMetrics,  # pyright: ignore[reportPrivateUsage]
    _ScopeMetric_T,  # pyright: ignore[reportPrivateUsage]
    _ScopeMetrics_Var,  # pyright: ignore[reportPrivateUsage]
)
from draive.scope.state import (
    ScopeState,
    ScopeStates,  # pyright: ignore[reportPrivateUsage]
    _ScopeState_T,  # pyright: ignore[reportPrivateUsage]
    _ScopeState_Var,  # pyright: ignore[reportPrivateUsage]
)

__all__ = [
    "ctx",
]


class _RootContext:
    def __init__(
        self,
        task_group: TaskGroup,
        metrics: ScopeMetrics,
        state: ScopeStates,
        dependencies: ScopeDependencies,
    ) -> None:
        self._task_group: TaskGroup = task_group
        self._metrics: ScopeMetrics = metrics
        self._state: ScopeStates = state
        self._dependencies: ScopeDependencies = dependencies
        self._token: Token[TaskGroup] | None = None

    async def __aenter__(self) -> None:
        assert self._token is None, "Reentrance is not allowed"  # nosec: B101
        self._token = _TaskGroup_Var.set(self._task_group)
        # start the task group first
        await self._task_group.__aenter__()
        # then begin metrics capture
        await self._metrics.__aenter__()
        # prepare state next
        self._state.__enter__()
        # finally initialize dependencies
        self._dependencies.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        assert self._token is not None, "Can't exit scope without entering"  # nosec: B101
        _TaskGroup_Var.reset(self._token)
        # finish task group - wait for completion
        await self._task_group.__aexit__(
            et=exc_type,
            exc=exc_val,
            tb=exc_tb,
        )
        # then end metrics capture
        await self._metrics.__aexit__(
            exc_type=exc_type,
            exc_val=exc_val,
            exc_tb=exc_tb,
        )
        # cleanup dependencies next
        self._dependencies.__exit__(
            exc_type=exc_type,
            exc_val=exc_val,
            exc_tb=exc_tb,
        )
        # finally reset state
        self._state.__exit__(
            exc_type=exc_type,
            exc_val=exc_val,
            exc_tb=exc_tb,
        )


_Args_T = ParamSpec("_Args_T")
_Result_T = TypeVar("_Result_T")


@final
class ctx:
    @staticmethod
    def new(
        label: str | None = None,
        *,
        logger: Logger | None = None,
        state: Iterable[ScopeState] | None = None,
        dependencies: ScopeDependencies | None = None,
        log_summary: bool = __debug__,
    ) -> _RootContext:
        return _RootContext(
            task_group=TaskGroup(),
            metrics=ScopeMetrics(
                label=label,
                logger=logger,
                parent=None,
                metrics=None,
                log_summary=log_summary,
            ),
            state=ScopeStates(*state or []),
            dependencies=dependencies or ScopeDependencies(),
        )

    @staticmethod
    def with_current(
        function: Callable[_Args_T, Coroutine[Any, Any, _Result_T]],
        /,
    ) -> Callable[_Args_T, Coroutine[Any, Any, _Result_T]]:
        # capture current context
        current_metrics: ScopeMetrics = ctx.current_metrics()
        current_metrics._enter_task()  # pyright: ignore[reportPrivateUsage]
        current: Context = copy_context()

        @wraps(function)
        async def wrapped(*args: _Args_T.args, **kwargs: _Args_T.kwargs) -> _Result_T:
            try:
                return await current.run(function, *args, **kwargs)
            finally:
                await current_metrics._exit_task()  # pyright: ignore[reportPrivateUsage]

        return wrapped

    @staticmethod
    def current_task_group() -> TaskGroup:
        try:
            return _TaskGroup_Var.get()
        except LookupError as exc:
            raise MissingScopeContext("TaskGroup requested but not defined!") from exc

    @staticmethod
    def current_metrics() -> ScopeMetrics:
        try:
            return _ScopeMetrics_Var.get()
        except LookupError as exc:
            raise MissingScopeContext("ScopeMetrics requested but not defined!") from exc

    @staticmethod
    def current_dependencies() -> ScopeDependencies:
        try:
            return _ScopeDependencies_Var.get()
        except LookupError:
            return ScopeDependencies()  # use empty dependencies by default

    @staticmethod
    def current_state() -> ScopeStates:
        try:
            return _ScopeState_Var.get()
        except LookupError:
            return ScopeStates()  # use empty state by default

    @staticmethod
    def spawn_task(
        coro: Coroutine[Any, Any, None],
        /,
    ) -> Task[None]:
        return ctx.current_task_group().create_task(coro)

    @staticmethod
    def nested(
        label: str,
        /,
        *metrics: ScopeMetric,
    ) -> ScopeMetrics:
        return ctx.current_metrics().nested(
            label=label,
            metrics=metrics,
        )

    @staticmethod
    def updated(
        *state: ScopeState,
    ) -> ScopeStates:
        return ctx.current_state().updated(state)

    @staticmethod
    def id() -> str:
        return ctx.current_metrics().trace_id

    @staticmethod
    def state(
        _type: type[_ScopeState_T],
        /,
    ) -> _ScopeState_T:
        return ctx.current_state().state(_type)

    @staticmethod
    def dependency(
        _type: type[_ScopeDependency_T],
        /,
    ) -> _ScopeDependency_T:
        return ctx.current_dependencies().dependency(_type)

    @staticmethod
    async def read(
        _type: type[_ScopeMetric_T],
        /,
    ) -> _ScopeMetric_T | None:
        return await ctx.current_metrics().read(_type)

    @staticmethod
    async def record(
        *metrics: ScopeMetric,
    ) -> None:
        await ctx.current_metrics().record(*metrics)

    @staticmethod
    def log_error(
        message: str,
        /,
        *args: Any,
    ) -> None:
        ctx.current_metrics().log_error(
            message,
            *args,
        )

    @staticmethod
    def log_warning(
        message: str,
        /,
        *args: Any,
    ) -> None:
        ctx.current_metrics().log_warning(
            message,
            *args,
        )

    @staticmethod
    def log_info(
        message: str,
        /,
        *args: Any,
    ) -> None:
        ctx.current_metrics().log_info(
            message,
            *args,
        )

    @staticmethod
    def log_debug(
        message: str,
        /,
        *args: Any,
    ) -> None:
        ctx.current_metrics().log_debug(
            message,
            *args,
        )


_TaskGroup_Var = ContextVar[TaskGroup]("_TaskGroup_Var")
