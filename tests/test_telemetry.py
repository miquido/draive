from collections.abc import Mapping
from typing import Any

import pytest
from haiway import ctx

from draive.evaluation import evaluator, evaluator_suite
from draive.guardrails import GuardrailsQualityVerification
from draive.guardrails.quality.types import GuardrailsQualityException
from draive.models import ModelToolRequest
from draive.multimodal import MultimodalContent
from draive.qdrant import QdrantException
from draive.qdrant.utils import qdrant_operation
from draive.tools import Toolbox, tool


class _Recorder:
    def __init__(self) -> None:
        self.records: list[Mapping[str, Any]] = []

    def __call__(self, **kwargs: Any) -> None:
        self.records.append(kwargs)

    def metric(self, name: str) -> list[Mapping[str, Any]]:
        return [record for record in self.records if record.get("metric") == name]


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(ctx, "record_info", recorder)
    return recorder


@pytest.mark.asyncio
async def test_toolbox_records_tool_invocations(recorder: _Recorder) -> None:
    @tool
    async def working(value: str) -> str:
        return value

    @tool
    async def failing(value: str) -> str:
        raise RuntimeError(value)

    toolbox = Toolbox.of(working, failing)
    async with ctx.scope("test"):
        for name in ("working", "failing", "missing"):
            _ = [
                chunk
                async for chunk in toolbox.response_stream(
                    ModelToolRequest.of(name, tool=name, arguments={"value": "x"})
                )
            ]

    statuses = {
        record["attributes"]["tool.name"]: record["attributes"]["tool.status"]
        for record in recorder.metric("tool.invocations")
    }
    assert statuses == {"working": "success", "failing": "error", "missing": "unknown"}


@evaluator(name="low", threshold="perfect")
async def _low_evaluator(value: MultimodalContent) -> float:
    return 0.1


@evaluator(name="broken", threshold="perfect")
async def _broken_evaluator(value: MultimodalContent) -> float:
    raise RuntimeError("judge unavailable")


@pytest.mark.asyncio
async def test_guardrails_record_verdicts(recorder: _Recorder) -> None:
    async with ctx.scope("test"):
        with pytest.raises(GuardrailsQualityException):
            await GuardrailsQualityVerification.of(_low_evaluator.prepared()).verify("text")

        with pytest.raises(Exception):  # noqa: B017
            await GuardrailsQualityVerification.of(_broken_evaluator.prepared()).verify("text")

    outcomes = [
        (record["attributes"]["guardrails.kind"], record["attributes"]["guardrails.outcome"])
        for record in recorder.metric("guardrails.verdicts")
    ]
    assert outcomes == [("quality", "blocked"), ("quality", "failed")]


@pytest.mark.asyncio
async def test_evaluator_metric_marks_errored_runs(recorder: _Recorder) -> None:
    async with ctx.scope("test"):
        await _broken_evaluator("text")
        await _low_evaluator("text")

    errored = {
        record["metric"]: record["attributes"]["errored"]
        for record in recorder.records
        if str(record.get("metric", "")).startswith("evaluator.")
    }
    assert errored == {
        "evaluator.broken.performance": True,
        "evaluator.low.performance": False,
    }


@pytest.mark.asyncio
async def test_suite_records_performance(recorder: _Recorder) -> None:
    from haiway import State

    class Case(State):
        text: str

    async def definition(parameters: Case) -> list[Any]:
        return [await _low_evaluator(parameters.text)]

    suite = evaluator_suite(Case, name="telemetry")(definition)
    async with ctx.scope("test"):
        await suite([Case(text="a"), Case(text="b")])

    (record,) = recorder.metric("evaluator.suite.telemetry.performance")
    assert record["attributes"] == {"passed": False, "cases": 2, "cases.passed": 0}


def test_qdrant_operation_records_duration_and_outcome(recorder: _Recorder) -> None:
    with qdrant_operation("search"):
        pass

    with pytest.raises(QdrantException):
        with qdrant_operation("search"):
            raise RuntimeError("boom")

    outcomes = [
        record["attributes"]["qdrant.outcome"]
        for record in recorder.metric("qdrant.operation.duration")
    ]
    assert outcomes == ["success", "error"]


@pytest.mark.asyncio
async def test_surreal_query_records_duration_rows_and_outcome(
    recorder: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surrealdb.request_message.methods import RequestMethod

    from draive.surreal import SurrealClient, SurrealException
    from draive.surreal import connection as surreal_connection

    class _FakeConnection:
        def __init__(self, url: str) -> None:
            self.url = url

        async def connect(self, url: str | None = None) -> None:
            self.url = url or self.url

        async def _send(self, message: Any, process: str, bypass: bool = False) -> Any:
            _ = process, bypass
            if message.method is RequestMethod.QUERY:
                if "FAIL" in message.kwargs["query"]:
                    raise RuntimeError("boom")

                return {"result": [{"status": "OK", "result": [{"value": 1}, {"value": 2}]}]}

            return {"result": None}

        async def close(self) -> None:
            return None

    monkeypatch.setattr(surreal_connection, "AsyncWsSurrealConnection", _FakeConnection)

    async with SurrealClient(url="ws://example.test", namespace="ns", database="db") as client:
        async with client.prepare_session() as session:
            await session.execute("SELECT * FROM item;")
            with pytest.raises(SurrealException):
                await session.execute("FAIL;")

    attributes = [record["attributes"] for record in recorder.metric("surreal.query.duration")]
    assert attributes == [
        {"surreal.outcome": "success", "surreal.rows": 2},
        {"surreal.outcome": "error", "surreal.rows": None},
    ]
