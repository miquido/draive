from asyncio import Task, TaskGroup
from collections.abc import Callable, Coroutine, Iterable
from contextvars import Context, ContextVar, Token, copy_context
from logging import Logger
from types import TracebackType
from typing import Any, Literal, ParamSpec, TypeVar, final

from draive.helpers import getenv_bool
from draive.scope.dependencies import DependenciesScope, ScopeDependency
from draive.scope.errors import MissingScopeContext
from draive.scope.metrics import MetricsScope, ScopeMetric
from draive.scope.state import StateScope
from draive.types.parameters import ParametrizedState

__all__ = [
    "ctx",
]

_TaskGroup_Var = ContextVar[TaskGroup]("_TaskGroup_Var")
_ScopeMetric_T = TypeVar(
    "_ScopeMetric_T",
    bound=ScopeMetric,
)
_MetricsScope_Var = ContextVar[MetricsScope]("_MetricsScope_Var")
_ScopeState_T = TypeVar(
    "_ScopeState_T",
    bound=ParametrizedState,
)
_StateScope_Var = ContextVar[StateScope]("_ScopeState_Var")
_ScopeDependency_T = TypeVar(
    "_ScopeDependency_T",
    bound=ScopeDependency,
)
_DependenciesScope_Var = ContextVar[DependenciesScope]("_DependenciesScope_Var")
_Args_T = ParamSpec("_Args_T")
_Result_T = TypeVar("_Result_T")


class _RootContext:
    def __init__(
        self,
        task_group: TaskGroup,
        metrics: MetricsScope,
        state: StateScope,
        dependencies: DependenciesScope,
    ) -> None:
        self._task_group: TaskGroup = task_group
        self._task_group_token: Token[TaskGroup] | None = None
        self._metrics: MetricsScope = metrics
        self._metrics_token: Token[MetricsScope] | None = None
        self._state: StateScope = state
        self._state_token: Token[StateScope] | None = None
        self._dependencies: DependenciesScope = dependencies
        self._dependencies_token: Token[DependenciesScope] | None = None

    async def __aenter__(self) -> None:
        # start the task group first
        assert self._task_group_token is None, "Reentrance is not allowed"  # nosec: B101
        self._task_group_token = _TaskGroup_Var.set(self._task_group)
        await self._task_group.__aenter__()
        # then begin metrics capture
        assert self._metrics_token is None, "Reentrance is not allowed"  # nosec: B101
        self._metrics.enter_task()
        self._metrics_token = _MetricsScope_Var.set(self._metrics)
        # prepare state scope next
        assert self._state_token is None, "Reentrance is not allowed"  # nosec: B101
        self._state_token = _StateScope_Var.set(self._state)
        # finally initialize dependencies
        assert self._dependencies_token is None, "Reentrance is not allowed"  # nosec: B101
        self._dependencies_token = _DependenciesScope_Var.set(self._dependencies)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        assert self._task_group_token is not None, "Can't exit scope without entering"  # nosec: B101
        _TaskGroup_Var.reset(self._task_group_token)
        # finish task group - wait for completion
        try:
            await self._task_group.__aexit__(
                et=exc_type,
                exc=exc_val,
                tb=exc_tb,
            )
        except BaseException as exc:
            # then end metrics capture with either context or task group error
            assert self._metrics_token is not None, "Can't exit scope without entering"  # nosec: B101
            _MetricsScope_Var.reset(self._metrics_token)
            self._metrics.exit_task(exception=exc_val or exc)
        else:
            # or end metrics capture with context error
            assert self._metrics_token is not None, "Can't exit scope without entering"  # nosec: B101
            _MetricsScope_Var.reset(self._metrics_token)
            self._metrics.exit_task(exception=exc_val)
        finally:
            # cleanup dependencies next
            assert self._dependencies_token is not None, "Can't exit scope without entering"  # nosec: B101
            _DependenciesScope_Var.reset(self._dependencies_token)
            # finally reset state
            assert self._state_token is not None, "Can't exit scope without entering"  # nosec: B101
            _StateScope_Var.reset(self._state_token)


class _PartialContext:
    def __init__(
        self,
        metrics: MetricsScope | None = None,
        state: StateScope | None = None,
    ) -> None:
        self._metrics: MetricsScope | None = metrics
        self._metrics_token: Token[MetricsScope] | None = None
        self._state: StateScope | None = state
        self._state_token: Token[StateScope] | None = None

    def __enter__(self) -> None:
        if metrics := self._metrics:
            assert self._metrics_token is None, "Reentrance is not allowed"  # nosec: B101
            metrics.enter_task()
            self._metrics_token = _MetricsScope_Var.set(metrics)
        if state := self._state:
            assert self._state_token is None, "Reentrance is not allowed"  # nosec: B101
            self._state_token = _StateScope_Var.set(state)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if token := self._metrics_token:
            _MetricsScope_Var.reset(self._metrics_token)
            if metrics := self._metrics:
                metrics.exit_task(exception=exc_val)
            else:
                raise RuntimeError("Can't exit metrics scope without valid reference to it")
        if token := self._state_token:
            _StateScope_Var.reset(token)


@final
class ctx:
    @staticmethod
    def new(  # noqa: PLR0913
        label: str | None = None,
        *,
        dependencies: DependenciesScope
        | Iterable[type[ScopeDependency] | ScopeDependency]
        | None = None,
        state: StateScope | Iterable[ParametrizedState] | None = None,
        metrics: Iterable[ScopeMetric] | None = None,
        logger: Logger | None = None,
        log_summary: Literal["full", "trimmed", "none"] | None = None,
    ) -> _RootContext:
        root_dependencies: DependenciesScope
        if dependencies is None:
            root_dependencies = DependenciesScope()
        elif isinstance(dependencies, DependenciesScope):
            root_dependencies = dependencies
        else:
            root_dependencies = DependenciesScope(*dependencies)

        root_state: StateScope
        if state is None:
            root_state = StateScope()
        elif isinstance(state, StateScope):
            root_state = state
        else:
            root_state = StateScope(*state)

        return _RootContext(
            task_group=TaskGroup(),
            metrics=MetricsScope(
                label=label,
                logger=logger,
                parent=None,
                metrics=metrics,
                log_summary=log_summary
                or ("trimmed" if getenv_bool("DEBUG_LOGGING", __debug__) else "none"),
            ),
            state=root_state,
            dependencies=root_dependencies,
        )

    @staticmethod
    def _current_task_group() -> TaskGroup:
        try:
            return _TaskGroup_Var.get()
        except LookupError as exc:
            raise MissingScopeContext("TaskGroup requested but not defined!") from exc

    @staticmethod
    def _current_metrics() -> MetricsScope:
        try:
            return _MetricsScope_Var.get()
        except LookupError as exc:
            raise MissingScopeContext("MetricsScope requested but not defined!") from exc

    @staticmethod
    def _current_dependencies() -> DependenciesScope:
        try:
            return _DependenciesScope_Var.get()
        except LookupError as exc:
            raise MissingScopeContext("DependenciesScope requested but not defined!") from exc

    @staticmethod
    def _current_state() -> StateScope:
        try:
            return _StateScope_Var.get()
        except LookupError as exc:
            raise MissingScopeContext("StateScope requested but not defined!") from exc

    @staticmethod
    def spawn_task(
        function: Callable[_Args_T, Coroutine[Any, Any, _Result_T]],
        /,
        *args: _Args_T.args,
        **kwargs: _Args_T.kwargs,
    ) -> Task[_Result_T]:
        nested_context: _PartialContext = ctx.nested(function.__name__)
        current_context: Context = copy_context()

        async def wrapped(*args: _Args_T.args, **kwargs: _Args_T.kwargs) -> _Result_T:
            with nested_context:
                return await function(*args, **kwargs)

        return ctx._current_task_group().create_task(current_context.run(wrapped, *args, **kwargs))

    @staticmethod
    def nested(
        label: str,
        /,
        state: StateScope | Iterable[ParametrizedState] | None = None,
        metrics: Iterable[ScopeMetric] | None = None,
    ) -> _PartialContext:
        nested_state: StateScope | None
        if isinstance(state, StateScope):
            nested_state = state
        else:
            nested_state = ctx._current_state().updated(state)

        return _PartialContext(
            metrics=ctx._current_metrics().nested(
                label=label,
                metrics=metrics,
            ),
            state=nested_state,
        )

    @staticmethod
    def updated(
        *state: ParametrizedState,
    ) -> _PartialContext:
        return _PartialContext(state=ctx._current_state().updated(state))

    @staticmethod
    def id() -> str:
        return ctx._current_metrics().trace_id

    @staticmethod
    def state(
        _type: type[_ScopeState_T],
        /,
    ) -> _ScopeState_T:
        return ctx._current_state().state(_type)

    @staticmethod
    def dependency(
        _type: type[_ScopeDependency_T],
        /,
    ) -> _ScopeDependency_T:
        return ctx._current_dependencies().dependency(_type)

    @staticmethod
    def read(
        _type: type[_ScopeMetric_T],
        /,
    ) -> _ScopeMetric_T | None:
        return ctx._current_metrics().read(_type)

    @staticmethod
    def record(
        *metrics: ScopeMetric,
    ) -> None:
        ctx._current_metrics().record(*metrics)

    @staticmethod
    def log_error(
        message: str,
        /,
        *args: Any,
    ) -> None:
        ctx._current_metrics().log_error(
            message,
            *args,
        )

    @staticmethod
    def log_warning(
        message: str,
        /,
        *args: Any,
    ) -> None:
        ctx._current_metrics().log_warning(
            message,
            *args,
        )

    @staticmethod
    def log_info(
        message: str,
        /,
        *args: Any,
    ) -> None:
        ctx._current_metrics().log_info(
            message,
            *args,
        )

    @staticmethod
    def log_debug(
        message: str,
        /,
        *args: Any,
    ) -> None:
        ctx._current_metrics().log_debug(
            message,
            *args,
        )
