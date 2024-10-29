from asyncio import (
    AbstractEventLoop,
    Task,
    TaskGroup,
    current_task,
    get_running_loop,
    shield,
)
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    Callable,
    Coroutine,
    Generator,
    Iterable,
)
from concurrent.futures import Executor
from contextvars import ContextVar, Token
from logging import Logger, getLogger
from types import TracebackType
from typing import Any, Literal, final

from haiway import MISSING, Missing, asynchronous, getenv_bool, mimic_function
from typing_extensions import deprecated

from draive.metrics import (
    ExceptionTrace,
    Metric,
    MetricsTrace,  # pyright: ignore[reportDeprecated]
    MetricsTraceReporter,
    metrics_log_reporter,
)
from draive.parameters import ParametrizedData
from draive.scope.dependencies import (
    ScopeDependencies,  # pyright: ignore[reportDeprecated]
    ScopeDependency,  # pyright: ignore[reportDeprecated]
)
from draive.scope.errors import MissingScopeContext
from draive.scope.state import ScopeState
from draive.utils import AsyncStream

__all__ = [
    "ctx",
]

_TaskGroup_Var = ContextVar[TaskGroup]("_TaskGroup_Var")
_MetricsScope_Var = ContextVar[MetricsTrace]("_MetricsScope_Var")  # pyright: ignore[reportDeprecated]
_StateScope_Var = ContextVar[ScopeState]("_ScopeState_Var")
_DependenciesScope_Var = ContextVar[ScopeDependencies]("_DependenciesScope_Var")  # pyright: ignore[reportDeprecated]


class _RootContext:
    def __init__(  # noqa: PLR0913
        self,
        task_group: TaskGroup,
        dependencies: ScopeDependencies,  # pyright: ignore[reportDeprecated]
        state: ScopeState,
        metrics: MetricsTrace,  # pyright: ignore[reportDeprecated]
        trace_reporting: MetricsTraceReporter | None,
        completion: Callable[[MetricsTrace], Coroutine[None, None, None]] | None,  # pyright: ignore[reportDeprecated]
    ) -> None:
        self._task_group: TaskGroup = task_group
        self._task_group_token: Token[TaskGroup] | None = None
        self._dependencies: ScopeDependencies = dependencies  # pyright: ignore[reportDeprecated]
        self._dependencies_token: Token[ScopeDependencies] | None = None  # pyright: ignore[reportDeprecated]
        self._state: ScopeState = state
        self._state_token: Token[ScopeState] | None = None
        self._metrics: MetricsTrace = metrics  # pyright: ignore[reportDeprecated]
        self._metrics_token: Token[MetricsTrace] | None = None  # pyright: ignore[reportDeprecated]
        self._report_trace: MetricsTraceReporter | None = trace_reporting
        self._completion: Callable[[MetricsTrace], Coroutine[None, None, None]] | None = completion  # pyright: ignore[reportDeprecated]

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

            if completion := self._completion:
                await completion(self._metrics)

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
        metrics: MetricsTrace | None = None,  # pyright: ignore[reportDeprecated]
        state: ScopeState | None = None,
        completion: Callable[[MetricsTrace], Coroutine[None, None, None]] | None = None,  # pyright: ignore[reportDeprecated]
    ) -> None:
        self._task_group: TaskGroup | None = task_group
        self._task_group_token: Token[TaskGroup] | None = None
        self._metrics: MetricsTrace | None = metrics  # pyright: ignore[reportDeprecated]
        self._metrics_token: Token[MetricsTrace] | None = None  # pyright: ignore[reportDeprecated]
        self._state: ScopeState | None = state
        self._state_token: Token[ScopeState] | None = None
        self._completion: Callable[[MetricsTrace], Coroutine[None, None, None]] | None = completion  # pyright: ignore[reportDeprecated]

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

        finally:
            if completion := self._completion:
                await completion(self._metrics or _MetricsScope_Var.get())


@final
class ctx:
    @deprecated("`new` will be replaced by `scope`")
    @staticmethod
    def new(  # noqa: PLR0913
        label: str | None = None,
        /,
        *,
        dependencies: ScopeDependencies  # pyright: ignore[reportDeprecated]
        | Iterable[type[ScopeDependency] | ScopeDependency]  # pyright: ignore[reportDeprecated]
        | None = None,
        state: ScopeState | Iterable[ParametrizedData] | None = None,
        metrics: Iterable[Metric] | None = None,
        logger: Logger | None = None,
        trace_reporting: MetricsTraceReporter | None = None,
    ) -> _RootContext:
        root_dependencies: ScopeDependencies  # pyright: ignore[reportDeprecated]
        if dependencies is None:
            root_dependencies = ScopeDependencies()  # pyright: ignore[reportDeprecated]
        elif isinstance(dependencies, ScopeDependencies):  # pyright: ignore[reportDeprecated]
            root_dependencies = dependencies
        else:
            root_dependencies = ScopeDependencies(*dependencies)  # pyright: ignore[reportDeprecated]

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
            metrics=MetricsTrace(  # pyright: ignore[reportDeprecated]
                label=label,
                logger=root_logger,
                parent=None,
                metrics=metrics,
            ),
            trace_reporting=trace_reporter,
            completion=None,
        )

    @deprecated("`wrap` will be removed")
    @staticmethod
    def wrap[**Args, Result](  # noqa: PLR0913
        label: str | None = None,
        /,
        *,
        dependencies: ScopeDependencies  # pyright: ignore[reportDeprecated]
        | Iterable[type[ScopeDependency] | ScopeDependency]  # pyright: ignore[reportDeprecated]
        | None = None,
        state: ScopeState | Iterable[ParametrizedData] | None = None,
        metrics: Iterable[Metric] | None = None,
        logger: Logger | None = None,
        trace_reporting: MetricsTraceReporter | None = None,
    ) -> Callable[
        [Callable[Args, Coroutine[None, None, Result]]],
        Callable[Args, Coroutine[None, None, Result]],
    ]:
        root_dependencies: ScopeDependencies  # pyright: ignore[reportDeprecated]
        if dependencies is None:
            root_dependencies = ScopeDependencies()  # pyright: ignore[reportDeprecated]
        elif isinstance(dependencies, ScopeDependencies):  # pyright: ignore[reportDeprecated]
            root_dependencies = dependencies
        else:
            root_dependencies = ScopeDependencies(*dependencies)  # pyright: ignore[reportDeprecated]

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
                async with ctx.new(  # pyright: ignore[reportDeprecated]
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
    def _current_metrics() -> MetricsTrace:  # pyright: ignore[reportDeprecated]
        try:
            return _MetricsScope_Var.get()

        except LookupError as exc:
            raise MissingScopeContext("MetricsScope requested but not defined!") from exc

    @staticmethod
    def _current_dependencies() -> ScopeDependencies:  # pyright: ignore[reportDeprecated]
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
        nested_context: _PartialContext = ctx.nested(function.__name__)  # pyright: ignore[reportDeprecated]

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

    @deprecated("`nested` will be replaced by `scope`")
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
    def scope(
        label: str,
        /,
        *state: ParametrizedData,
        logger: Logger | None = None,
        trace_id: str | None = None,
        completion: Callable[[MetricsTrace], Coroutine[None, None, None]] | None = None,  # pyright: ignore[reportDeprecated]
    ) -> _RootContext | _PartialContext:
        try:
            return _PartialContext(
                task_group=TaskGroup(),
                metrics=ctx._current_metrics().nested(
                    label=label,
                ),
                state=ctx._current_state().updated(state),
                completion=completion,
            )

        except MissingScopeContext:
            root_dependencies = ScopeDependencies()  # pyright: ignore[reportDeprecated]
            root_state = ScopeState(*state)

            root_logger: Logger = logger or getLogger(name=label)

            trace_reporter: MetricsTraceReporter | None
            if getenv_bool("DEBUG_LOGGING", __debug__):
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
                metrics=MetricsTrace(  # pyright: ignore[reportDeprecated]
                    label=label,
                    logger=root_logger,
                    parent=None,
                    metrics=None,
                    trace_id=trace_id,
                ),
                trace_reporting=trace_reporter,
                completion=completion,
            )

    @staticmethod
    def stream[Element](
        source: AsyncGenerator[Element, None],
        /,
    ) -> AsyncIterable[Element]:
        metrics: MetricsTrace = ctx._current_metrics()  # pyright: ignore[reportDeprecated]
        metrics.enter()

        stream: AsyncStream[Element] = AsyncStream()

        async def consumer() -> None:
            try:
                async for element in source:
                    await stream.send(element)

            except BaseException as exc:
                stream.finish(exception=exc)

            else:
                stream.finish()

        task: Task[None] = ctx.spawn_subtask(consumer)

        def on_finish(task: Task[None]) -> None:
            metrics.exit()

        task.add_done_callback(on_finish)

        return stream

    @deprecated("`stream_sync` will be removed")
    @staticmethod
    def stream_sync[Element](
        source: Generator[Element, None],
        /,
        executor: Executor | Literal["default"] | Missing = "default",
    ) -> AsyncIterable[Element]:
        metrics: MetricsTrace = ctx._current_metrics()  # pyright: ignore[reportDeprecated]
        metrics.enter()

        loop: AbstractEventLoop = get_running_loop()

        @asynchronous(loop=loop, executor=executor)  # pyright: ignore[reportArgumentType]
        def source_next() -> Element:
            try:
                return next(source)

            except GeneratorExit as exc:
                raise StopAsyncIteration() from exc

            except StopIteration as exc:
                raise StopAsyncIteration() from exc

        stream: AsyncStream[Element] = AsyncStream()

        async def consumer() -> None:
            try:
                while True:
                    try:
                        await stream.send(await source_next())

                    except StopAsyncIteration:
                        break

            except BaseException as exc:
                stream.finish(exception=exc)

            else:
                stream.finish()

        task: Task[None] = ctx.spawn_subtask(consumer)

        def on_finish(task: Task[None]) -> None:
            metrics.exit()

        task.add_done_callback(on_finish)

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
        default: State_T | Missing = MISSING,
    ) -> State_T:
        return ctx._current_state().state(
            state,
            default=default,
        )

    @deprecated("dependencies will be removed in favor of context state propagation")
    @staticmethod
    def dependency[Dependency_T: ScopeDependency](
        dependency: type[Dependency_T],
        /,
    ) -> Dependency_T:
        return ctx._current_dependencies().dependency(dependency)

    @deprecated("read will be removed")
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
