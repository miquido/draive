from logging import Logger
from typing import Self

import pytest
import pytest_asyncio
from draive import DataModel, MetricsTraceReport, TokenUsage, ctx


class ExpMetric(DataModel):
    value: float

    def __add__(
        self,
        other: Self,
        /,
    ) -> Self:
        return self.__class__(value=self.value * other.value)


@pytest_asyncio.fixture
async def metrics_report() -> MetricsTraceReport:
    captured_report: MetricsTraceReport | None = None

    async def capture_report(
        trace_id: str,
        logger: Logger,
        report: MetricsTraceReport,
    ) -> None:
        nonlocal captured_report
        captured_report = report

    async with ctx.new(trace_reporting=capture_report):
        ctx.record(ExpMetric(value=1))

        with ctx.nested("child"):
            ctx.record(TokenUsage.for_model("test", input_tokens=44, output_tokens=55))

            with ctx.nested("grandchild_1"):
                ctx.record(TokenUsage.for_model("test", input_tokens=444, output_tokens=555))
                ctx.record(ExpMetric(value=5))

            with ctx.nested("grandchild_2"):
                ctx.record(TokenUsage.for_model("test", input_tokens=222, output_tokens=333))
                ctx.record(ExpMetric(value=7))

    if captured_report:
        return captured_report
    else:
        raise AssertionError("Missing metrics trace report")


@pytest.mark.asyncio
async def test_combinable_metrics(metrics_report: MetricsTraceReport) -> None:
    combined_metrics = metrics_report.with_combined_metrics().metrics
    assert len(combined_metrics) == 2
    assert "TokenUsage" in combined_metrics
    assert "ExpMetric" in combined_metrics

    token_usage = combined_metrics["TokenUsage"]
    exp_metric = combined_metrics["ExpMetric"]

    assert exp_metric.value == 35
    assert token_usage.usage["test"].input_tokens == 710
    assert token_usage.usage["test"].output_tokens == 943
