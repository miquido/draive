from collections.abc import Iterable
from logging import Logger, getLogger
from time import monotonic
from typing import Any, Self, cast, final
from uuid import uuid4

from draive.metrics.metric import Metric
from draive.metrics.reporter import MetricsTraceReport

__all__ = [
    "MetricsTrace",
]


@final  # unstructured background tasks spawning may result in corrupted data
class MetricsTrace:
    def __init__(
        self,
        *,
        label: str | None,
        logger: Logger | None,
        parent: Self | None,
        metrics: Iterable[Metric] | None,
    ) -> None:
        self._start: float = monotonic()
        self._end: float | None = None
        self._runners: int = 0
        self._trace_id: str = parent._trace_id if parent else uuid4().hex
        self._label: str = label or "metrics"
        self._parent: Self | None = parent
        self._logger: Logger = logger or (parent._logger if parent else getLogger(name=self._label))
        self._metrics: dict[type[Metric], Metric] = {
            type(metric): metric for metric in metrics or []
        }
        self._nested_traces: list[MetricsTrace] = []
        self.log_info("started...")

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
        metrics: Iterable[Metric] | None = None,
    ) -> Self:
        if self.is_finished:
            raise ValueError("Attempting to use already finished metrics trace")
        self.enter()
        nested: Self = self.__class__(
            label=label,
            logger=self._logger,
            parent=self,
            metrics=metrics,
        )
        self._nested_traces.append(nested)
        return nested

    # - METRICS -

    def record(
        self,
        *metrics: Metric,
    ) -> None:
        if self.is_finished:  # ignore record when already finished
            return self.log_error("Attempting to use already finished metrics trace, ignoring...")

        for metric in metrics:
            metric_type: type[Metric] = type(metric)
            try:  # catch exceptions - we don't wan't to blow up on metrics
                if current := self._metrics.get(metric_type):
                    if hasattr(current, "__add__"):
                        self._metrics[metric_type] = current + metric  # pyright: ignore[reportOperatorIssue]
                    else:
                        raise NotImplementedError(f"{metric_type.__qualname__} can't be combined!")

                else:
                    self._metrics[metric_type] = metric

            except Exception as exc:
                self.log_error(
                    "Failed to record %s metric",
                    metric_type.__qualname__,
                    exception=exc,
                )

    def read[Metric_T: Metric](
        self,
        metric: type[Metric_T],
        /,
    ) -> Metric_T | None:
        try:  # catch all exceptions - we don't wan't to blow up on metrics
            return cast(Metric_T, self._metrics.get(metric))

        except Exception as exc:
            self.log_error(
                "Failed to read %s metric",
                metric.__qualname__,
                exception=exc,
            )
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

    def enter(self) -> None:
        assert not self.is_finished, "Attempting to use already finished metrics trace"  # nosec: B101
        self._runners += 1

    def exit(self) -> None:
        assert not self.is_finished, "Attempting to use already finished metrics trace"  # nosec: B101
        assert self._runners > 0, "Unbalanced metrics trace exit call"  # nosec: B101
        self._runners -= 1

        if self._runners > 0:
            return  # can't finish yet

        self._end = monotonic()

        self.log_info(
            "...finished after %.2fs",
            self._end - self._start,
        )

        if parent := self._parent:
            parent.exit()

    def report(self) -> MetricsTraceReport:
        return MetricsTraceReport(
            label=self._label,
            duration=(self._end or monotonic()) - self._start,
            metrics={key.__qualname__: value for key, value in self._metrics.items()},
            nested=[trace.report() for trace in self._nested_traces],
            combined=False,
        )

    def __str__(self) -> str:
        return f"{self._trace_id}|{self._label}"
