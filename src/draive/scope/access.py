from asyncio import Task, TaskGroup, current_task, shield
from collections.abc import (
    AsyncGenerator,
    Callable,
    Coroutine,
    Generator,
    Iterable,
    Iterator,
)
from concurrent.futures import Executor
from contextvars import ContextVar, Token
from logging import Logger, getLogger
from types import TracebackType
from typing import Any, final

from draive.metrics import (
    ExceptionTrace,
    Metric,
    MetricsTrace,
    MetricsTraceReporter,
    metrics_log_reporter,
)
from draive.parameters import ParametrizedData
from draive.scope.dependencies import ScopeDependencies, ScopeDependency
from draive.scope.errors import MissingScopeContext
from draive.scope.state import ScopeState
from draive.utils import AsyncStream, getenv_bool, mimic_function, run_async

__all__ = [
    "ctx",
]

_TaskGroup_Var = ContextVar[TaskGroup]("_TaskGroup_Var")
_MetricsScope_Var = ContextVar[MetricsTrace]("_MetricsScope_Var")
_StateScope_Var = ContextVar[ScopeState]("_ScopeState_Var")
_DependenciesScope_Var = ContextVar[ScopeDependencies]("_DependenciesScope_Var")


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
            if (exception := exc_val) and exc_type is not GeneratorExit:
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
                try:  # catch all exceptions - we don't want to blow up on metrics
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
        task_group: TaskGroup | None = None,
        metrics: MetricsTrace | None = None,
        state: ScopeState | None = None,
    ) -> None:
        self._task_group: TaskGroup | None = task_group
        self._task_group_token: Token[TaskGroup] | None = None
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
            if (exception := exc_val) and exc_type is not GeneratorExit:
                metrics.record(ExceptionTrace.of(exception))

            metrics.exit()

        if token := self._state_token:
            _StateScope_Var.reset(token)

    async def __aenter__(self) -> None:
        if task_group := self._task_group:
            assert self._task_group_token is None, "Reentrance is not allowed"  # nosec: B101
            self._task_group_token = _TaskGroup_Var.set(task_group)
            await task_group.__aenter__()

        self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            if task_group := self._task_group:
                assert self._task_group_token is not None, "Can't exit scope without entering"  # nosec: B101
                _TaskGroup_Var.reset(self._task_group_token)
                await task_group.__aexit__(
                    et=exc_type,
                    exc=exc_val,
                    tb=exc_tb,
                )

        except BaseException as exc:
            self.__exit__(
                exc_type=type(exc),
                exc_val=exc,
                exc_tb=exc.__traceback__,
            )

        else:
            self.__exit__(
                exc_type=exc_type,
                exc_val=exc_val,
                exc_tb=exc_tb,
            )


@final
class ctx:
    @staticmethod
    def new(  # noqa: PLR0913
        label: str | None = None,
        /,
        *,
        dependencies: ScopeDependencies
        | Iterable[type[ScopeDependency] | ScopeDependency]
        | None = None,
        state: ScopeState | Iterable[ParametrizedData] | None = None,
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

        root_logger: Logger = logger or getLogger(name=label)

        trace_reporter: MetricsTraceReporter | None
        if trace_reporting:
            trace_reporter = trace_reporting
        elif getenv_bool("DEBUG_LOGGING", __debug__):
            trace_reporter = metrics_log_reporter(
                list_items_limit=-4,
                item_character_limit=64,
            )
        else:
            trace_reporter = None

        return _RootContext(
            task_group=TaskGroup(),
            dependencies=root_dependencies,
            state=root_state,
            metrics=MetricsTrace(
                label=label,
                logger=root_logger,
                parent=None,
                metrics=metrics,
            ),
            trace_reporting=trace_reporter,
        )

    @staticmethod
    def wrap[**Args, Result](  # noqa: PLR0913
        label: str | None = None,
        /,
        *,
        dependencies: ScopeDependencies
        | Iterable[type[ScopeDependency] | ScopeDependency]
        | None = None,
        state: ScopeState | Iterable[ParametrizedData] | None = None,
        metrics: Iterable[Metric] | None = None,
        logger: Logger | None = None,
        trace_reporting: MetricsTraceReporter | None = None,
    ) -> Callable[
        [Callable[Args, Coroutine[None, None, Result]]],
        Callable[Args, Coroutine[None, None, Result]],
    ]:
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

        trace_reporter: MetricsTraceReporter | None
        if trace_reporting:
            trace_reporter = trace_reporting
        elif getenv_bool("DEBUG_LOGGING", __debug__):
            trace_reporter = metrics_log_reporter(
                list_items_limit=-4,
                item_character_limit=64,
            )
        else:
            trace_reporter = None

        def wrapper(
            function: Callable[Args, Coroutine[None, None, Result]],
            /,
        ) -> Callable[Args, Coroutine[None, None, Result]]:
            @mimic_function(function)
            async def wrapped(*args: Args.args, **kwargs: Args.kwargs) -> Result:
                async with ctx.new(
                    label,
                    dependencies=root_dependencies,
                    state=root_state,
                    metrics=metrics,
                    logger=logger,
                    trace_reporting=trace_reporter,
                ):
                    return await function(*args, **kwargs)

            return wrapped

        return wrapper

    @staticmethod
    def update[**Args, Result](
        *state: ParametrizedData,
    ) -> Callable[
        [Callable[Args, Coroutine[None, None, Result]]],
        Callable[Args, Coroutine[None, None, Result]],
    ]:
        def wrapper(
            function: Callable[Args, Coroutine[None, None, Result]],
            /,
        ) -> Callable[Args, Coroutine[None, None, Result]]:
            @mimic_function(function)
            async def wrapped(*args: Args.args, **kwargs: Args.kwargs) -> Result:
                with ctx.updated(*state):
                    return await function(*args, **kwargs)

            return wrapped

        return wrapper

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
    def spawn_task[**Args, Result](
        function: Callable[Args, Coroutine[None, None, Result]],
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Task[Result]:
        nested_context: _PartialContext = ctx.nested(function.__name__)

        async def wrapped(*args: Args.args, **kwargs: Args.kwargs) -> Result:
            with nested_context:
                return await function(*args, **kwargs)

        return ctx._current_task_group().create_task(wrapped(*args, **kwargs))

    @staticmethod
    def spawn_subtask[**Args, Result](
        function: Callable[Args, Coroutine[None, None, Result]],
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Task[Result]:
        return ctx._current_task_group().create_task(function(*args, **kwargs))

    @staticmethod
    def cancel() -> None:
        if task := current_task():
            task.cancel()

        else:
            raise RuntimeError("Attempting to cancel context out of asyncio task")

    @staticmethod
    def nested(
        label: str,
        /,
        *,
        state: ScopeState | Iterable[ParametrizedData] | None = None,
        metrics: Iterable[Metric] | None = None,
    ) -> _PartialContext:
        nested_state: ScopeState | None
        if isinstance(state, ScopeState):
            nested_state = state
        else:
            nested_state = ctx._current_state().updated(state)

        return _PartialContext(
            task_group=TaskGroup(),
            metrics=ctx._current_metrics().nested(
                label=label,
                metrics=metrics,
            ),
            state=nested_state,
        )

    @staticmethod
    def stream[Element](
        generator: AsyncGenerator[Element, None],
    ) -> AsyncStream[Element]:
        # TODO: find better solution for streaming without spawning tasks if able
        stream: AsyncStream[Element] = AsyncStream()
        current_metrics: MetricsTrace = ctx._current_metrics()
        current_metrics.enter()  # ensure valid metrics scope closing

        async def iterate() -> None:
            try:
                async for element in generator:
                    await stream.send(element)

            except BaseException as exc:
                stream.finish(exception=exc)

            else:
                stream.finish()

            finally:
                current_metrics.exit()

        ctx.spawn_subtask(iterate)
        return stream

    @staticmethod
    def stream_sync[Element](
        generator: Generator[Element, None] | Iterator[Element],
        /,
        executor: Executor | None = None,
    ) -> AsyncStream[Element]:
        # TODO: find better solution for streaming without spawning tasks if able
        stream: AsyncStream[Element] = AsyncStream()
        current_metrics: MetricsTrace = ctx._current_metrics()
        current_metrics.enter()  # ensure valid metrics scope closing

        iterator: Iterator[Element] = iter(generator)

        @run_async(executor=executor)
        def next_element() -> Element:
            try:
                return next(iterator)

            except StopIteration as exc:
                raise StopAsyncIteration() from exc

        async def iterate() -> None:
            try:
                while True:
                    await stream.send(await next_element())

            except BaseException as exc:
                stream.finish(exception=exc)

            finally:
                current_metrics.exit()

        ctx.spawn_subtask(iterate)
        return stream

    @staticmethod
    def updated(
        *state: ParametrizedData,
    ) -> _PartialContext:
        return _PartialContext(state=ctx._current_state().updated(state))

    @staticmethod
    def id() -> str:
        return ctx._current_metrics().trace_id

    @staticmethod
    def state[State_T: ParametrizedData](
        state: type[State_T],
        /,
    ) -> State_T:
        return ctx._current_state().state(state)

    @staticmethod
    def dependency[Dependency_T: ScopeDependency](
        dependency: type[Dependency_T],
        /,
    ) -> Dependency_T:
        return ctx._current_dependencies().dependency(dependency)

    @staticmethod
    def read[Metric_T: Metric](
        metric: type[Metric_T],
        /,
    ) -> Metric_T | None:
        return ctx._current_metrics().read(metric)

    @staticmethod
    def record(
        *metrics: Metric,
    ) -> None:
        try:
            ctx._current_metrics().record(*metrics)

        # ignoring metrics record when using out of metrics context
        # using default logger as fallback as we already know that we are missing metrics
        except MissingScopeContext as exc:
            logger: Logger = getLogger()
            logger.error("Attempting to record metrics outside of metrics context")
            logger.error(
                exc,
                exc_info=True,
            )

    @staticmethod
    def log_error(
        message: str,
        /,
        *args: Any,
        exception: BaseException | None = None,
    ) -> None:
        try:
            ctx._current_metrics().log_error(
                message,
                *args,
                exception=exception,
            )

        # using default logger as fallback when using out of metrics context
        except MissingScopeContext:
            logger: Logger = getLogger()
            logger.error(
                message,
                *args,
            )
            if exception := exception:
                logger.error(
                    exception,
                    exc_info=True,
                )

    @staticmethod
    def log_warning(
        message: str,
        /,
        *args: Any,
        exception: Exception | None = None,
    ) -> None:
        try:
            ctx._current_metrics().log_warning(
                message,
                *args,
                exception=exception,
            )

        # using default logger as fallback when using out of metrics context
        except MissingScopeContext:
            logger: Logger = getLogger()
            logger.warning(
                message,
                *args,
            )
            if exception := exception:
                logger.error(
                    exception,
                    exc_info=True,
                )

    @staticmethod
    def log_info(
        message: str,
        /,
        *args: Any,
    ) -> None:
        try:
            ctx._current_metrics().log_info(
                message,
                *args,
            )

        # using default logger as fallback when using out of metrics context
        except MissingScopeContext:
            getLogger().info(
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
        try:
            ctx._current_metrics().log_debug(
                message,
                *args,
                exception=exception,
            )

        # using default logger as fallback when using out of metrics context
        except MissingScopeContext:
            logger: Logger = getLogger()
            logger.debug(
                message,
                *args,
            )
            if exception := exception:
                logger.error(
                    exception,
                    exc_info=True,
                )
