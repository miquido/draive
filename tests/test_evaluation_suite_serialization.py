from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime

import pytest

from draive import ctx
from draive.evaluation import EvaluatorResult, EvaluatorScenarioResult, evaluator_suite
from draive.parameters import DataModel


class _DatetimeCase(DataModel):
    date: datetime


@pytest.mark.asyncio
async def test_evaluator_suite_serializes_datetime(tmp_path) -> None:
    storage_path = tmp_path / "suite.json"
    storage_path.write_text("[]")

    fixed_date = datetime(2025, 9, 25, 11, 42, 9, 804558)

    async def definition(
        parameters: _DatetimeCase,
    ) -> Sequence[EvaluatorScenarioResult | EvaluatorResult]:
        return []

    suite = evaluator_suite(
        _DatetimeCase,
        storage=storage_path,
    )(definition)

    async with ctx.scope("test"):
        await suite.add_case(_DatetimeCase(date=fixed_date))

    persisted = json.loads(storage_path.read_text())

    assert len(persisted) == 1
    assert persisted[0]["parameters"]["date"] == fixed_date.isoformat()
