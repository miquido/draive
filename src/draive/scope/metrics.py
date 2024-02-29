from asyncio import Lock
from collections.abc import Iterable
from contextvars import ContextVar, Token
from logging import Logger, getLogger
from time import time
from types import TracebackType
from typing import Any, Protocol, Self, TypeVar, cast, final, runtime_checkable
from uuid import uuid4

__all__ = [
    "ScopeMetric",
    "ArgumentsTrace",
    "ResultTrace",
    "TokenUsage",
    "ScopeMetrics",
]


@runtime_checkable
class ScopeMetric(Protocol):
    def metric_summary(self) -> str | None:
        ...


@runtime_checkable
class CombinableScopeMetric(Protocol):
    def metric_summary(self) -> str | None:
        ...

    def combine_metric(
        self,
        other: Self,
        /,
    ) -> None:
        ...


# TokenUsage is a bit of a special case, we might want to allow
# similar behavior for other metrics in future by allowing
# summarization method for adding up results from nested contexts.
class TokenUsage(CombinableScopeMetric):
    def __init__(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        self._input_tokens: int = input_tokens or 0
        self._output_tokens: int = output_tokens or 0

    def combine_metric(
        self,
        other: Self,
        /,
    ) -> None:
        self._input_tokens += other._input_tokens
        self._output_tokens += other._output_tokens

    def metric_summary(self) -> str | None:
        return (
            f"token usage: {self._input_tokens + self._output_tokens}"
            f" (in:{self._input_tokens} out:{self._output_tokens})"
        )


class ArgumentsTrace(ScopeMetric):
    if __debug__:

        def __init__(
            self,
            **kwargs: Any,
        ) -> None:
            self._kwargs: dict[str, Any] = kwargs

        def metric_summary(self) -> str | None:
            if self._kwargs:
                arguments_description: str = "\n".join(
                    f"|   - {key}: {value}".replace("\n", "\n|   ")
                    for key, value in self._kwargs.items()
                )
                return f"arguments:\n{arguments_description}"

            else:
                return "arguments: None"

    else:  # in non debug builds redact the values

        def __init__(
            self,
            **kwargs: Any,
        ) -> None:
            pass

        def metric_summary(self) -> str | None:
            return None


class ResultTrace(ScopeMetric):
    if __debug__:

        def __init__(
            self,
            __result: Any,
        ) -> None:
            self._result: Any = __result

        def metric_summary(self) -> str | None:
            return f"result: {self._result}".replace("\n", "\n|  ")

    else:  # in non debug builds redact the values

        def __init__(
            self,
            __result: Any,
        ) -> None:
            pass

        def metric_summary(self) -> str | None:
            return None


_ScopeMetric_T = TypeVar(
    "_ScopeMetric_T",
    bound=ScopeMetric,
)


@final  # unstructured background tasks spawning may result in corrupted data
class ScopeMetrics:
    def __init__(  # noqa: PLR0913
        self,
        *,
        label: str | None,
        logger: Logger | None,
        parent: Self | None,
        metrics: Iterable[ScopeMetric] | None,
        log_summary: bool,
    ) -> None:
        self._start: float = time()
        self._end: float | None = None
        self._exception: BaseException | None = None
        self._pending_tasks: int = 0
        self._lock: Lock = Lock()
        self._trace_id: str = parent._trace_id if parent else uuid4().hex
        self._label: str = label or "metrics"
        self._parent: Self | None = parent
        self._logger: Logger = logger or (parent._logger if parent else getLogger(name=self._label))
        self._metrics: dict[type[ScopeMetric], ScopeMetric] = {
            type(metric): metric for metric in metrics or []
        }
        self._nested_traces: list[ScopeMetrics] = []
        self._log_summary: bool = log_summary and parent is None  # only root can do summary
        self._token: Token[ScopeMetrics] | None = None
        self.log_info("%s started", self)

    # - STATE -

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def is_root(self) -> bool:
        return self._parent is None

    @property
    def is_finished(self) -> bool:
        return self._end is not None

    def nested(
        self,
        label: str,
        *,
        metrics: Iterable[ScopeMetric] | None = None,
    ) -> Self:
        if self.is_finished:
            raise ValueError("Attempting to use already finished metrics")
        self._enter_task()
        child: Self = self.__class__(
            label=label,
            logger=self._logger,
            parent=self,
            metrics=metrics,
            log_summary=False,
        )
        self._nested_traces.append(child)
        return child

    # - METRICS -

    async def record(
        self,
        *metrics: ScopeMetric,
    ) -> None:
        try:  # catch exceptions - we don't wan't to blow up on metrics
            if self.is_finished:
                raise ValueError("Attempting to use already finished metrics")

            async with self._lock:
                for metric in metrics:
                    metric_type: type[ScopeMetric] = type(metric)
                    if metric_type in self._metrics:
                        if isinstance(metric, CombinableScopeMetric):
                            self._metrics[metric_type].combine_metric(  # pyright: ignore[reportUnknownMemberType, reportGeneralTypeIssues, reportAttributeAccessIssue]
                                metric
                            )
                        else:
                            raise NotImplementedError(f"{type(self)} can't be combined!")

                    else:
                        self._metrics[metric_type] = metric

        except Exception as exc:
            self.log_error("Failed to record metrics caused by %s", exc)

    async def read(
        self,
        _type: type[_ScopeMetric_T],
        /,
    ) -> _ScopeMetric_T | None:
        try:  # catch all exceptions - we don't wan't to blow up on metrics
            async with self._lock:
                return cast(_ScopeMetric_T, self._metrics.get(_type))

        except Exception as exc:
            self.log_error("Failed to read metrics caused by %s", exc)
            return None

    # - LOGS -

    def log_error(
        self,
        message: str,
        /,
        *args: Any,
    ) -> None:
        self._logger.error(
            f"[%s] {message}",
            self,
            *args,
        )

    def log_warning(
        self,
        message: str,
        /,
        *args: Any,
    ) -> None:
        self._logger.warning(
            f"[%s] {message}",
            self,
            *args,
        )

    def log_info(
        self,
        message: str,
        /,
        *args: Any,
    ) -> None:
        self._logger.info(
            f"[%s] {message}",
            self,
            *args,
        )

    def log_debug(
        self,
        message: str,
        /,
        *args: Any,
    ) -> None:
        self._logger.debug(
            f"[%s] {message}",
            self,
            *args,
        )

    # - INTERNAL -

    def _enter_task(self) -> None:
        if self.is_finished:
            raise ValueError("Attempting to use already finished metrics")
        self._pending_tasks += 1

    async def _exit_task(self) -> None:
        assert self._pending_tasks > 0, "Unbalanced metrics task exit"  # nosec: B101
        self._pending_tasks -= 1

        if self._pending_tasks > 0:
            return  # can't finish yet

        async with self._lock:
            self._end = time()

            if self._log_summary:
                self.log_debug(
                    "%s",
                    await self._summary(),
                )

            duration: float = self._end - (self._start or self._end)

            if exception := self._exception:
                self.log_error(
                    "%s finished after %.2fs with exception: %s\n%s",
                    self,
                    duration,
                    type(exception).__name__,
                    exception,
                )
            else:
                self.log_info(
                    "%s finished after %.2fs",
                    self,
                    duration,
                )

        # exit out of lock to avoid deadlock on summary
        if parent := self._parent:
            await parent._exit_task()

    def __str__(self) -> str:
        return f"{self._trace_id}|{self._label}"

    async def __aenter__(self) -> None:
        assert not self.is_finished, "Attempting to use already finished metrics"  # nosec: B101
        assert self._token is None, "Reentrance is not allowed"  # nosec: B101
        self._token = _ScopeMetrics_Var.set(self)
        self._enter_task()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        assert self._token is not None, "Can't exit scope without entering"  # nosec: B101
        assert not self.is_finished, "Attempting to use already finished metrics"  # nosec: B101

        try:
            self._exception = exc_val
            await self._exit_task()

        finally:
            _ScopeMetrics_Var.reset(self._token)

    # - PRIVATE -

    # Warning: this method is not using lock
    async def _summary(self) -> str:
        summary: str
        if self.is_root:
            summary = f"[{self._trace_id}] Summary:\n• {self._label}:"
        else:
            summary = f"• {self._label}:"

        if start := self._start:
            duration: float = (self._end or time()) - start
            summary += f"\n- duration: {duration:.2f}s"

        if exception := self._exception:
            summary += f"\n- exception: {type(exception).__name__}s"

        if self.is_root:
            summary += f"\n- total {self._total_tokens_usage().metric_summary()}"

        for metric in self._metrics.values():
            if metric_summary := metric.metric_summary():
                summary += f"\n- {metric_summary}"

        for child in self._nested_traces:
            child_summary: str = (await child._summary()).replace("\n", "\n|   ")
            summary += f"\n{child_summary}"

        return summary

    # TODO: combine all combinable metrics this way
    # Warning: this method is not using lock
    def _total_tokens_usage(self) -> TokenUsage:
        total_usage: TokenUsage = TokenUsage()
        match self._metrics.get(TokenUsage):
            case TokenUsage() as usage:
                total_usage.combine_metric(usage)
            case _:
                pass

        for child in self._nested_traces:
            total_usage.combine_metric(child._total_tokens_usage())

        return total_usage


_ScopeMetrics_Var = ContextVar[ScopeMetrics]("_ScopeMetrics_Var")
