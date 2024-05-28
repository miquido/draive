from logging import Logger
from typing import Protocol, Self

from draive.metrics.function import ExceptionTrace
from draive.metrics.metric import Metric
from draive.parameters import State

__all__ = [
    "MetricsTraceReport",
    "MetricsTraceReporter",
]


class MetricsTraceReport(State):
    label: str
    duration: float
    metrics: dict[str, Metric]
    nested: list["MetricsTraceReport"]
    combined: bool

    def with_combined_metrics(self) -> Self:
        if self.combined:
            return self  # avoid combining multiple times

        combined_metrics: dict[str, Metric] = self.metrics.copy()

        nested_reports: list[MetricsTraceReport] = []
        for report in self.nested:
            combined_report: MetricsTraceReport = report.with_combined_metrics()
            for metric_type, metric in combined_report.metrics.items():
                if hasattr(metric, "__add__"):
                    if metric_type == ExceptionTrace:
                        continue  # skip ExceptionTrace combining
                    elif current := combined_metrics.get(metric_type):
                        combined_metrics[metric_type] = current + metric  # pyright: ignore[reportOperatorIssue]

                    else:
                        combined_metrics[metric_type] = metric
                else:
                    continue  # skip metric that can't combine

            nested_reports.append(combined_report)

        return self.__class__(
            label=self.label,
            duration=self.duration,
            metrics=combined_metrics,
            nested=nested_reports,
            combined=True,
        )


class MetricsTraceReporter(Protocol):
    async def __call__(
        self,
        trace_id: str,
        logger: Logger,
        report: MetricsTraceReport,
    ) -> None: ...
