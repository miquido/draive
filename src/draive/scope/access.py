from asyncio import Task, TaskGroup, shield
from collections.abc import Callable, Coroutine, Iterable
from contextvars import Context, ContextVar, Token, copy_context
from logging import Logger
from types import TracebackType
from typing import Any, ParamSpec, TypeVar, final

from draive.helpers import getenv_bool
from draive.metrics import (
    ExceptionTrace,
    Metric,
    MetricsTrace,
    MetricsTraceReporter,
    metrics_trimmed_log_report,
)
from draive.scope.dependencies import ScopeDependencies, ScopeDependency
from draive.scope.errors import MissingScopeContext
from draive.scope.state import ScopeState
from draive.types.parameters import ParametrizedState

__all__ = [
    "ctx",
]

_TaskGroup_Var = ContextVar[TaskGroup]("_TaskGroup_Var")
_ScopeMetric_T = TypeVar(
    "_ScopeMetric_T",
    bound=Metric,
)
_MetricsScope_Var = ContextVar[MetricsTrace]("_MetricsScope_Var")
_ScopeState_T = TypeVar(
    "_ScopeState_T",
    bound=ParametrizedState,
)
_StateScope_Var = ContextVar[ScopeState]("_ScopeState_Var")
_ScopeDependency_T = TypeVar(
    "_ScopeDependency_T",
    bound=ScopeDependency,
)
_DependenciesScope_Var = ContextVar[ScopeDependencies]("_DependenciesScope_Var")
_Args_T = ParamSpec("_Args_T")
_Result_T = TypeVar("_Result_T")


class _RootContext:
    def __init__(  # noqa: PLR0913
        self,
        task_group: TaskGroup,
        dependencies: ScopeDependencies,
        state: ScopeState,
        metrics: MetricsTrace,
        trace_reporting: MetricsTraceReporter | None,
    ) -> None:
        self._task_group: TaskGroup = task_group
        self._task_group_token: Token[TaskGroup] | None = None
        self._dependencies: ScopeDependencies = dependencies
        self._dependencies_token: Token[ScopeDependencies] | None = None
        self._state: ScopeState = state
        self._state_token: Token[ScopeState] | None = None
        self._metrics: MetricsTrace = metrics
        self._metrics_token: Token[MetricsTrace] | None = None
        self._report_trace: MetricsTraceReporter | None = trace_reporting

    async def __aenter__(self) -> None:
        # start the task group first
        assert self._task_group_token is None, "Reentrance is not allowed"  # nosec: B101
        self._task_group_token = _TaskGroup_Var.set(self._task_group)
        await self._task_group.__aenter__()
        # prepare state scope
        assert self._state_token is None, "Reentrance is not allowed"  # nosec: B101
        self._state_token = _StateScope_Var.set(self._state)
        # then initialize dependencies
        assert self._dependencies_token is None, "Reentrance is not allowed"  # nosec: B101
        self._dependencies_token = _DependenciesScope_Var.set(self._dependencies)
        # finally begin metrics capture
        assert self._metrics_token is None, "Reentrance is not allowed"  # nosec: B101
        self._metrics.enter()
        self._metrics_token = _MetricsScope_Var.set(self._metrics)

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
            # record task group exceptions
            self._metrics.record(ExceptionTrace.of(exc_val or exc))

        else:
            # or context exception
            if exception := exc_val:
                self._metrics.record(ExceptionTrace.of(exception))

        finally:
            # then end metrics capture
            assert self._metrics_token is not None, "Can't exit scope without entering"  # nosec: B101
            _MetricsScope_Var.reset(self._metrics_token)
            self._metrics.exit()
            assert (  # nosec: B101
                self._metrics.is_finished
            ), "Unbalanced metrics trace enter/exit calls, possibly an unstructured task running"
            # report metrics trace
            if report_trace := self._report_trace:
                # it still have access to the dependencies and state
                try:  # catch all exceptions - we don't wan't to blow up on metrics
                    await shield(
                        report_trace(
                            trace_id=self._metrics.trace_id,
                            logger=self._metrics._logger,  # pyright: ignore[reportPrivateUsage]
                            report=self._metrics.report(),
                        )
                    )

                except Exception as exc:
                    self._metrics.log_error(
                        "Failed to finish metrics trace report",
                        exception=exc,
                    )
                    return None

            # cleanup dependencies next
            assert self._dependencies_token is not None, "Can't exit scope without entering"  # nosec: B101
            _DependenciesScope_Var.reset(self._dependencies_token)
            # finally reset state
            assert self._state_token is not None, "Can't exit scope without entering"  # nosec: B101
            _StateScope_Var.reset(self._state_token)


class _PartialContext:
    def __init__(
        self,
        metrics: MetricsTrace | None = None,
        state: ScopeState | None = None,
    ) -> None:
        self._metrics: MetricsTrace | None = metrics
        self._metrics_token: Token[MetricsTrace] | None = None
        self._state: ScopeState | None = state
        self._state_token: Token[ScopeState] | None = None

    def __enter__(self) -> None:
        if metrics := self._metrics:
            assert self._metrics_token is None, "Reentrance is not allowed"  # nosec: B101
            metrics.enter()
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
        if (token := self._metrics_token) and (metrics := self._metrics):
            _MetricsScope_Var.reset(self._metrics_token)
            if exception := exc_val:
                metrics.record(ExceptionTrace.of(exception))

            metrics.exit()

        if token := self._state_token:
            _StateScope_Var.reset(token)


@final
class ctx:
    @staticmethod
    def new(  # noqa: PLR0913
        label: str | None = None,
        *,
        dependencies: ScopeDependencies
        | Iterable[type[ScopeDependency] | ScopeDependency]
        | None = None,
        state: ScopeState | Iterable[ParametrizedState] | None = None,
        metrics: Iterable[Metric] | None = None,
        logger: Logger | None = None,
        trace_reporting: MetricsTraceReporter | None = None,
    ) -> _RootContext:
        root_dependencies: ScopeDependencies
        if dependencies is None:
            root_dependencies = ScopeDependencies()
        elif isinstance(dependencies, ScopeDependencies):
            root_dependencies = dependencies
        else:
            root_dependencies = ScopeDependencies(*dependencies)

        root_state: ScopeState
        if state is None:
            root_state = ScopeState()
        elif isinstance(state, ScopeState):
            root_state = state
        else:
            root_state = ScopeState(*state)

        return _RootContext(
            task_group=TaskGroup(),
            dependencies=root_dependencies,
            state=root_state,
            metrics=MetricsTrace(
                label=label,
                logger=logger,
                parent=None,
                metrics=metrics,
            ),
            trace_reporting=trace_reporting
            or (metrics_trimmed_log_report if getenv_bool("DEBUG_LOGGING", __debug__) else None),
        )

    @staticmethod
    def _current_task_group() -> TaskGroup:
        try:
            return _TaskGroup_Var.get()
        except LookupError as exc:
            raise MissingScopeContext("TaskGroup requested but not defined!") from exc

    @staticmethod
    def _current_metrics() -> MetricsTrace:
        try:
            return _MetricsScope_Var.get()
        except LookupError as exc:
            raise MissingScopeContext("MetricsScope requested but not defined!") from exc

    @staticmethod
    def _current_dependencies() -> ScopeDependencies:
        try:
            return _DependenciesScope_Var.get()
        except LookupError as exc:
            raise MissingScopeContext("DependenciesScope requested but not defined!") from exc

    @staticmethod
    def _current_state() -> ScopeState:
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
        state: ScopeState | Iterable[ParametrizedState] | None = None,
        metrics: Iterable[Metric] | None = None,
    ) -> _PartialContext:
        nested_state: ScopeState | None
        if isinstance(state, ScopeState):
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
        state: type[_ScopeState_T],
        /,
    ) -> _ScopeState_T:
        return ctx._current_state().state(state)

    @staticmethod
    def dependency(
        dependency: type[_ScopeDependency_T],
        /,
    ) -> _ScopeDependency_T:
        return ctx._current_dependencies().dependency(dependency)

    @staticmethod
    def read(
        metric: type[_ScopeMetric_T],
        /,
    ) -> _ScopeMetric_T | None:
        return ctx._current_metrics().read(metric)

    @staticmethod
    def record(
        *metrics: Metric,
    ) -> None:
        ctx._current_metrics().record(*metrics)

    @staticmethod
    def log_error(
        message: str,
        /,
        *args: Any,
        exception: BaseException | None = None,
    ) -> None:
        ctx._current_metrics().log_error(
            message,
            *args,
            exception=exception,
        )

    @staticmethod
    def log_warning(
        message: str,
        /,
        *args: Any,
        exception: Exception | None = None,
    ) -> None:
        ctx._current_metrics().log_warning(
            message,
            *args,
            exception=exception,
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
        exception: Exception | None = None,
    ) -> None:
        ctx._current_metrics().log_debug(
            message,
            *args,
            exception=exception,
        )
