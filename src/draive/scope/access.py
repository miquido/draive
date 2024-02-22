from collections.abc import Iterable
from logging import Logger
from types import TracebackType
from typing import Any, final

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


class _ctx:
    def __init__(
        self,
        metrics: ScopeMetrics,
        state: ScopeStates,
        dependencies: ScopeDependencies,
    ):
        self._metrics: ScopeMetrics = metrics
        self._state: ScopeStates = state
        self._dependencies: ScopeDependencies = dependencies

    async def __aenter__(self) -> None:
        await self._metrics.__aenter__()
        self._state.__enter__()
        self._dependencies.__enter__()

    async def __aexit__(
        self,
        exc_type: BaseException | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._state.__exit__(
            exc_type=exc_type,
            exc_val=exc_val,
            exc_tb=exc_tb,
        )
        self._dependencies.__exit__(
            exc_type=exc_type,
            exc_val=exc_val,
            exc_tb=exc_tb,
        )
        await self._metrics.__aexit__(
            exc_type=exc_type,
            exc_val=exc_val,
            exc_tb=exc_tb,
        )


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
    ) -> _ctx:
        return _ctx(
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
    def _current_metrics() -> ScopeMetrics:
        try:
            return _ScopeMetrics_Var.get()
        except LookupError as exc:
            raise MissingScopeContext("ScopeMetrics requested but not defined!") from exc

    @staticmethod
    def _current_dependencies() -> ScopeDependencies:
        try:
            return _ScopeDependencies_Var.get()
        except LookupError as exc:
            raise MissingScopeContext("ScopeDependencies requested but not defined!") from exc

    @staticmethod
    def _current_state() -> ScopeStates:
        try:
            return _ScopeState_Var.get()
        except LookupError as exc:
            raise MissingScopeContext("ScopeState requested but not defined!") from exc

    @staticmethod
    def nested(
        label: str,
        /,
        *metrics: ScopeMetric,
    ) -> ScopeMetrics:
        return ctx._current_metrics().nested(label=label, metrics=metrics)

    @staticmethod
    def updated(
        *state: ScopeState,
    ) -> ScopeStates:
        return ctx._current_state().updated(state)

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
    async def read(
        _type: type[_ScopeMetric_T],
        /,
    ) -> _ScopeMetric_T | None:
        return await ctx._current_metrics().read(_type)

    @staticmethod
    async def record(
        *metrics: ScopeMetric,
    ) -> None:
        await ctx._current_metrics().record(*metrics)

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
