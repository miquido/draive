from collections.abc import Iterable
from contextvars import Token
from logging import Logger, getLogger
from time import monotonic
from typing import Any, Literal, Protocol, Self, TypeVar, cast, final, runtime_checkable
from uuid import uuid4

__all__ = [
    "ScopeMetric",
    "ArgumentsTrace",
    "ResultTrace",
    "TokenUsage",
    "MetricsScope",
]


@runtime_checkable
class ScopeMetric(Protocol):
    def metric_summary(
        self,
        trimmed: bool,
    ) -> str | None:
        ...


@runtime_checkable
class CombinableScopeMetric(Protocol):
    def metric_summary(
        self,
        trimmed: bool,
    ) -> str | None:
        ...

    def combined_metric(
        self,
        other: Self,
        /,
    ) -> Self:
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

    def combined_metric(
        self,
        other: Self,
        /,
    ) -> Self:
        return self.__class__(
            input_tokens=self._input_tokens + other._input_tokens,
            output_tokens=self._output_tokens + other._output_tokens,
        )

    def metric_summary(
        self,
        trimmed: bool,
    ) -> str | None:
        if trimmed:
            return f"token usage: {self._input_tokens + self._output_tokens}"

        else:
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

        def metric_summary(
            self,
            trimmed: bool,
        ) -> str | None:
            if self._kwargs:
                arguments_description: str = ""
                for key, value in self._kwargs.items():
                    value_str: str = str(value)
                    if trimmed and len(value_str) > MetricsScope.TRIMMING_CHARACTER_LIMIT:
                        value_str = (
                            f"{value_str[:MetricsScope.TRIMMING_CHARACTER_LIMIT]}...".replace(
                                "\n", " "
                            )
                        )

                    else:
                        value_str = value_str.replace("\n", "\n|   ")

                    arguments_description += f"\n|   - {key}: {value_str}"

                return f"arguments:{arguments_description}"

            else:
                return "arguments: None"

    else:  # in non debug builds redact the values

        def __init__(
            self,
            **kwargs: Any,
        ) -> None:
            pass

        def metric_summary(
            self,
            trimmed: bool,
        ) -> str | None:
            return None


class ResultTrace(ScopeMetric):
    if __debug__:

        def __init__(
            self,
            __result: Any,
        ) -> None:
            self._result: Any = __result

        def metric_summary(
            self,
            trimmed: bool,
        ) -> str | None:
            result_str: str = str(self._result)
            if trimmed and len(result_str) > MetricsScope.TRIMMING_CHARACTER_LIMIT:
                result_str = (f"{result_str[:MetricsScope.TRIMMING_CHARACTER_LIMIT]}...").replace(
                    "\n", " "
                )
            else:
                result_str = result_str.replace("\n", "\n|  ")

            return f"result: {result_str}"

    else:  # in non debug builds redact the values

        def __init__(
            self,
            __result: Any,
        ) -> None:
            pass

        def metric_summary(
            self,
            trimmed: bool,
        ) -> str | None:
            return None


_ScopeMetric_T = TypeVar(
    "_ScopeMetric_T",
    bound=ScopeMetric,
)


@final  # unstructured background tasks spawning may result in corrupted data
class MetricsScope:
    TRIMMING_CHARACTER_LIMIT: int = 64

    def __init__(  # noqa: PLR0913
        self,
        *,
        label: str | None,
        logger: Logger | None,
        parent: Self | None,
        metrics: Iterable[ScopeMetric] | None,
        log_summary: Literal["full", "trimmed", "none"],
    ) -> None:
        self._start: float = monotonic()
        self._end: float | None = None
        self._exception: BaseExceptionGroup[Any] | BaseException | None = None
        self._pending_tasks: int = 0
        self._trace_id: str = parent._trace_id if parent else uuid4().hex
        self._label: str = label or "metrics"
        self._parent: Self | None = parent
        self._logger: Logger = logger or (parent._logger if parent else getLogger(name=self._label))
        self._metrics: dict[type[ScopeMetric], ScopeMetric] = {
            type(metric): metric for metric in metrics or []
        }
        self._nested_traces: list[MetricsScope] = []
        self._log_summary: Literal["full", "trimmed", "none"] = log_summary
        self._token: Token[MetricsScope] | None = None
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
        self.enter_task()
        child: Self = self.__class__(
            label=label,
            logger=self._logger,
            parent=self,
            metrics=metrics,
            log_summary="none",
        )
        self._nested_traces.append(child)
        return child

    # - METRICS -

    def record(
        self,
        *metrics: ScopeMetric,
    ) -> None:
        try:  # catch exceptions - we don't wan't to blow up on metrics
            if self.is_finished:
                raise ValueError("Attempting to use already finished metrics")

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

    def read(
        self,
        _type: type[_ScopeMetric_T],
        /,
    ) -> _ScopeMetric_T | None:
        try:  # catch all exceptions - we don't wan't to blow up on metrics
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
        exception: BaseException | None = None,
    ) -> None:
        self._logger.error(
            f"[%s] {message}",
            self,
            *args,
        )
        if exception := exception:
            self._logger.error(
                exception,
                exc_info=True,
            )

    def log_warning(
        self,
        message: str,
        /,
        *args: Any,
        exception: Exception | None = None,
    ) -> None:
        self._logger.warning(
            f"[%s] {message}",
            self,
            *args,
        )
        if exception := exception:
            self._logger.warning(
                exception,
                exc_info=True,
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
        exception: Exception | None = None,
    ) -> None:
        self._logger.debug(
            f"[%s] {message}",
            self,
            *args,
        )
        if exception := exception:
            self._logger.debug(
                exception,
                exc_info=True,
            )

    # - INTERNAL -

    def enter_task(self) -> None:
        assert not self.is_finished, "Attempting to use already finished metrics"  # nosec: B101
        self._pending_tasks += 1

    def exit_task(
        self,
        exception: BaseException | None,
    ) -> None:
        assert not self.is_finished, "Attempting to use already finished metrics"  # nosec: B101
        assert self._pending_tasks > 0, "Unbalanced metrics task exit"  # nosec: B101
        self._pending_tasks -= 1
        if exception := exception:
            # the code below does not keep proper exception semantics of BaseException/Exception
            # however we are using it only for logging purposes at the moment
            # because of that merging exceptions in groups is simplified under BaseExceptionGroup
            if self._exception is None:
                self._exception = exception
            elif isinstance(self._exception, BaseExceptionGroup):
                self._exception = BaseExceptionGroup(
                    "Multiple errors",
                    (*self._exception.exceptions, exception),  # pyright: ignore[reportUnknownMemberType]
                )
            else:
                self._exception = BaseExceptionGroup(
                    "Multiple errors", (self._exception, exception)
                )

        if self._pending_tasks > 0:
            return  # can't finish yet

        self._end = monotonic()

        match self._log_summary:
            case "none":
                pass
            case "trimmed":
                self.log_debug(
                    "%s",
                    self._summary(trimmed=True),
                )
            case "full":
                self.log_debug(
                    "%s",
                    self._summary(trimmed=False),
                )

        duration: float = self._end - (self._start or self._end)

        if exception := self._exception:
            self.log_error(
                "%s finished after %.2fs with exception",
                self,
                duration,
                exception=exception,
            )
        else:
            self.log_info(
                "%s finished after %.2fs",
                self,
                duration,
            )

        if parent := self._parent:
            parent.exit_task(exception=None)

    def __str__(self) -> str:
        return f"{self._trace_id}|{self._label}"

    # - PRIVATE -

    # Warning: this method is not using lock
    def _summary(
        self,
        trimmed: bool,
    ) -> str:
        summary: str
        if self.is_root:
            summary = f"[{self._trace_id}] Summary:\n• {self._label}:"
        else:
            summary = f"• {self._label}:"

        if start := self._start:
            duration: float = (self._end or monotonic()) - start
            summary += f"\n- duration: {duration:.2f}s"

        if exception := self._exception:
            summary += f"\n- exception: {type(exception).__name__}"

        if self.is_root:
            for combined_metric in self._combined_metrics().values():
                summary += f"\n- total {combined_metric.metric_summary(trimmed=trimmed)}"

        for metric in self._metrics.values():
            if metric_summary := metric.metric_summary(trimmed=trimmed):
                summary += f"\n- {metric_summary}"

        for child in self._nested_traces:
            child_summary: str = child._summary(trimmed=trimmed).replace("\n", "\n|   ")
            summary += f"\n{child_summary}"

        return summary

    def _combined_metrics(
        self,
    ) -> dict[type[CombinableScopeMetric], CombinableScopeMetric]:
        metrics: dict[type[CombinableScopeMetric], CombinableScopeMetric] = {
            metric_type: metric
            for metric_type, metric in self._metrics.items()
            if isinstance(metric_type, CombinableScopeMetric)
            and isinstance(metric, CombinableScopeMetric)
        }

        for child in self._nested_traces:
            child_metrics = child._combined_metrics()
            for metric_type, child_metric in child_metrics.items():
                if metric_type in metrics:
                    metrics[metric_type] = metrics[metric_type].combined_metric(child_metric)
                else:
                    metrics[metric_type] = child_metric

        return metrics
