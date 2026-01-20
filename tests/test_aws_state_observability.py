import asyncio
import json
from uuid import UUID, uuid4

import pytest

from draive import ObservabilityLevel, ctx
from draive.aws.observability import ScopeStore
from draive.aws.state import AWSCloudwatch


class _FakeIdentifier:
    def __init__(
        self,
        name: str,
        scope_id: UUID,
        parent_id: UUID,
        *,
        is_root: bool = False,
    ) -> None:
        self.name = name
        self.scope_id = scope_id
        self.parent_id = parent_id
        self.is_root = is_root
        self.unique_name = f"{name}:{scope_id}"


@pytest.mark.asyncio
async def test_scope_emits_span_start_end_events(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks: list[asyncio.Task[None]] = []
    events: list[dict[str, object]] = []

    def spawn_background(coro, *args, **kwargs) -> None:
        tasks.append(asyncio.create_task(coro(*args, **kwargs)))

    async def event_putting(
        *,
        event_bus: str,
        event_source: str,
        detail_type: str,
        detail: str,
    ) -> None:
        events.append(
            {
                "event_bus": event_bus,
                "event_source": event_source,
                "detail_type": detail_type,
                "detail": json.loads(detail),
            }
        )

    async def log_putting(*, log_group: str, log_stream: str, message: str) -> None:
        _ = (log_group, log_stream, message)

    async def metric_putting(
        *,
        namespace: str,
        metric: str,
        value: float | int,
        unit: str | None,
        attributes: dict[str, object],
    ) -> None:
        _ = (namespace, metric, value, unit, attributes)

    monkeypatch.setattr(ctx, "spawn_background", spawn_background)
    monkeypatch.setattr(AWSCloudwatch, "event_putting", event_putting, raising=False)
    monkeypatch.setattr(AWSCloudwatch, "log_putting", log_putting, raising=False)
    monkeypatch.setattr(AWSCloudwatch, "metric_putting", metric_putting, raising=False)

    trace_id = uuid4()
    identifier = _FakeIdentifier("work", scope_id=uuid4(), parent_id=trace_id)
    scope = ScopeStore(
        identifier,
        trace_id=trace_id,
        log_group="group",
        log_stream="stream",
        event_bus="default",
        event_source="draive",
        metrics_namespace="metrics",
    )

    scope.record_scope_start()
    scope.record_scope_end(None)
    await asyncio.gather(*tasks)

    assert len(events) == 2
    detail_types = {event["detail_type"] for event in events}
    assert detail_types == {"event"}

    for event in events:
        detail = event["detail"]
        assert detail["trace_id"] == trace_id.hex
        assert detail["span_id"] == identifier.scope_id.hex[-16:]
        assert "time_unix_nano" in detail


@pytest.mark.asyncio
async def test_record_attributes_emit_event_and_metrics_are_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks: list[asyncio.Task[None]] = []
    recorded_metrics: list[dict[str, object]] = []
    recorded_events: list[dict[str, object]] = []

    def spawn_background(coro, *args, **kwargs) -> None:
        tasks.append(asyncio.create_task(coro(*args, **kwargs)))

    async def event_putting(
        *,
        event_bus: str,
        event_source: str,
        detail_type: str,
        detail: str,
    ) -> None:
        recorded_events.append(
            {
                "event_bus": event_bus,
                "event_source": event_source,
                "detail_type": detail_type,
                "detail": json.loads(detail),
            }
        )

    async def log_putting(*, log_group: str, log_stream: str, message: str) -> None:
        _ = (log_group, log_stream, message)

    async def metric_putting(
        *,
        namespace: str,
        metric: str,
        value: float | int,
        unit: str | None,
        attributes: dict[str, object],
    ) -> None:
        recorded_metrics.append(
            {
                "namespace": namespace,
                "metric": metric,
                "value": value,
                "unit": unit,
                "attributes": attributes,
            }
        )

    monkeypatch.setattr(ctx, "spawn_background", spawn_background)
    monkeypatch.setattr(AWSCloudwatch, "event_putting", event_putting, raising=False)
    monkeypatch.setattr(AWSCloudwatch, "log_putting", log_putting, raising=False)
    monkeypatch.setattr(AWSCloudwatch, "metric_putting", metric_putting, raising=False)

    trace_id = uuid4()
    identifier = _FakeIdentifier("work", scope_id=uuid4(), parent_id=trace_id)
    scope = ScopeStore(
        identifier,
        trace_id=trace_id,
        log_group="group",
        log_stream="stream",
        metrics_namespace="metrics",
        event_bus="default",
        event_source="draive",
    )

    scope.record_attributes({"user": "alice"}, level=ObservabilityLevel.INFO)
    scope.record_metric(
        "latency",
        value=1,
        unit=None,
        kind="gauge",
        attributes={"service": "draive"},
    )
    await asyncio.gather(*tasks)

    assert recorded_metrics
    assert recorded_metrics[0]["attributes"] == {
        "service": "draive",
        "otel.metric.kind": "gauge",
    }

    assert recorded_events
    attribute_events = [
        event for event in recorded_events if event["detail"]["name"] == "attributes"
    ]
    assert attribute_events
    attributes = attribute_events[0]["detail"]["attributes"]
    assert attributes["scope.name"] == "work"
    assert attributes["scope.id"] == identifier.scope_id.hex
    assert attributes["user"] == "alice"
