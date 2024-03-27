from typing import Self

import pytest
import pytest_asyncio
from draive import ctx
from draive.scope.metrics import CombinableScopeMetric, MetricsScope, TokenUsage


class ExpMetric(CombinableScopeMetric):
    def __init__(self, value: float = 1) -> None:
        self._value = value

    def combined_metric(
        self,
        other: Self,
        /,
    ) -> Self:
        return self.__class__(value=self._value * other._value)

    def metric_summary(
        self,
        trimmed: bool,
    ) -> str | None:
        return f"exp value: {self._value}"


@pytest_asyncio.fixture
async def metrics_scope() -> MetricsScope:
    async with ctx.new():
        ctx.record(ExpMetric(value=1))

        with ctx.nested("child"):
            ctx.record(TokenUsage(input_tokens=44, output_tokens=55))

            with ctx.nested("grandchild_1"):
                ctx.record(TokenUsage(input_tokens=444, output_tokens=555))
                ctx.record(ExpMetric(value=5))

            with ctx.nested("grandchild_2"):
                ctx.record(TokenUsage(input_tokens=222, output_tokens=333))
                ctx.record(ExpMetric(value=7))

        return ctx._current_metrics()


@pytest.mark.asyncio
async def test_combinable_metrics(metrics_scope: MetricsScope) -> None:
    combined_metrics = metrics_scope._combined_metrics()
    assert len(combined_metrics) == 2
    assert TokenUsage in combined_metrics
    assert ExpMetric in combined_metrics

    token_usage = combined_metrics[TokenUsage]
    exp_metric = combined_metrics[ExpMetric]

    assert exp_metric._value == 35
    assert token_usage._input_tokens == 710
    assert token_usage._output_tokens == 943
