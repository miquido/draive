from typing import Self

import pytest
import pytest_asyncio
from draive import ctx
from draive.scope.metrics import CombinableScopeMetric, ScopeMetrics, TokenUsage


class ExpMetric(CombinableScopeMetric):
    def __init__(self, value: float = 1) -> None:
        self._value = value

    def combine_metric(
        self,
        other: Self,
        /,
    ) -> None:
        self._value *= other._value

    def metric_summary(self) -> str | None:
        return (
            f"exp value: {self._value}"
        )

@pytest_asyncio.fixture
async def scope_metrics() -> ScopeMetrics:
    async with ctx.new():
        await ctx.record(TokenUsage(input_tokens=4, output_tokens=5))
        await ctx.record(ExpMetric(value=1))

        async with ctx.nested("child"):
            await ctx.record(TokenUsage(input_tokens=44, output_tokens=55))
            await ctx.record(ExpMetric(value=2))

            async with ctx.nested("grandchild_1"):
                await ctx.record(TokenUsage(input_tokens=444, output_tokens=555))
                await ctx.record(ExpMetric(value=5))

            async with ctx.nested("grandchild_2"):
                await ctx.record(TokenUsage(input_tokens=222, output_tokens=333))
                await ctx.record(ExpMetric(value=7))

        return ctx.current_metrics()

@pytest.mark.asyncio
async def test_combinable_metrics(scope_metrics: ScopeMetrics) -> None:
    combined_metrics = list(scope_metrics._combine_metrics())
    assert len(combined_metrics) == 2
    assert isinstance(combined_metrics[0], TokenUsage)
    assert isinstance(combined_metrics[1], ExpMetric)

    token_usage = combined_metrics[0]
    exp_metric = combined_metrics[1]

    assert exp_metric._value == 70
    assert token_usage._input_tokens == 714
    assert token_usage._output_tokens == 948
