import asyncio
import json
import traceback
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from datetime import UTC, datetime
from time import monotonic, time_ns
from typing import Any, Self, cast
from uuid import UUID, uuid4

from boto3 import Session  # pyright: ignore[reportMissingTypeStubs]
from botocore.exceptions import ClientError  # pyright: ignore[reportMissingModuleSource]
from haiway import (
    MISSING,
    ContextIdentifier,
    Observability,
    ObservabilityAttribute,
    ObservabilityLevel,
    ctx,
)
from haiway.context import ObservabilityMetricKind

from draive.aws.cloudwatch import format_metric_dimensions, translate_cloudwatch_error
from draive.aws.types import (
    AWSCloudwatchEventPutting,
    AWSCloudwatchLogPutting,
    AWSCloudwatchMetricPutting,
)

__all__ = ("CloudwatchObservability",)


def CloudwatchObservability(  # noqa: C901, PLR0915
    *,
    region_name: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    profile_name: str | None = None,
    log_level: ObservabilityLevel,
    log_group: str,
    log_stream: str,
    event_bus: str,
    event_source: str,
    metrics_namespace: str,
) -> Observability:
    trace_id: UUID = uuid4()
    root_scope: ContextIdentifier | None = None
    scopes: MutableMapping[UUID, ScopeStore] = {}

    session_kwargs: MutableMapping[str, object] = {}
    if access_key_id:
        session_kwargs["aws_access_key_id"] = access_key_id

    if secret_access_key:
        session_kwargs["aws_secret_access_key"] = secret_access_key

    if region_name:
        session_kwargs["region_name"] = region_name

    if profile_name:
        session_kwargs["profile_name"] = profile_name

    session: Session = Session(**session_kwargs)
    cloudwatch_client: Any = session.client(  # pyright: ignore[reportUnknownMemberType,  reportUnknownVariableType]
        "cloudwatch",
    )
    cloudwatch_logs_client: Any = session.client(  # pyright: ignore[reportUnknownMemberType,  reportUnknownVariableType]
        "logs",
    )
    eventbridge_client: Any = session.client(  # pyright: ignore[reportUnknownMemberType,  reportUnknownVariableType]
        "events",
    )

    async def log_putting(
        *,
        log_group: str,
        log_stream: str,
        message: str,
    ) -> None:
        try:
            await asyncio.to_thread(
                cloudwatch_logs_client.put_log_events,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                logGroupName=log_group,
                logStreamName=log_stream,
                logEvents=[
                    {
                        "timestamp": time_ns() // 1_000_000,
                        "message": message,
                    }
                ],
            )

        except ClientError as exc:
            raise translate_cloudwatch_error(
                error=exc,
                service="logs",
                operation="put_log",
                resource=f"{log_group}/{log_stream}",
            ) from exc

    async def metric_putting(
        *,
        namespace: str,
        metric: str,
        value: float | int,
        unit: str | None,
        attributes: Mapping[str, ObservabilityAttribute],
    ) -> None:
        try:
            dimensions = format_metric_dimensions(attributes)
            metric_data: dict[str, object] = {
                "MetricName": metric,
                "Value": value,
            }

            if dimensions:
                metric_data["Dimensions"] = dimensions

            if unit:
                metric_data["Unit"] = unit

            await asyncio.to_thread(
                cloudwatch_client.put_metric_data,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                Namespace=namespace,
                MetricData=[metric_data],
            )

        except ClientError as exc:
            raise translate_cloudwatch_error(
                error=exc,
                service="cloudwatch",
                operation="put_metric_data",
                resource=namespace,
            ) from exc

    async def event_putting(
        *,
        event_bus: str,
        event_source: str,
        detail_type: str,
        detail: str,
    ) -> None:
        try:
            await asyncio.to_thread(
                eventbridge_client.put_events,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                Entries=[
                    {
                        "EventBusName": event_bus,
                        "Source": event_source,
                        "DetailType": detail_type,
                        "Detail": detail,
                        "Time": datetime.now(tz=UTC),
                    }
                ],
            )

        except ClientError as exc:
            raise translate_cloudwatch_error(
                error=exc,
                service="events",
                operation="put_events",
                resource=event_bus,
            ) from exc

    def trace_identifying(
        scope: ContextIdentifier,
        /,
    ) -> UUID:
        assert root_scope is not None  # nosec: B101
        assert scope.scope_id in scopes  # nosec: B101

        return trace_id

    def log_recording(
        scope: ContextIdentifier,
        /,
        level: ObservabilityLevel,
        message: str,
        *args: Any,
        exception: BaseException | None,
    ) -> None:
        assert root_scope is not None  # nosec: B101
        assert scope.scope_id in scopes  # nosec: B101

        if level < log_level:
            return

        formatted_message: str = message
        if args:
            try:
                formatted_message = message % args

            except Exception:
                formatted_message = message

        scopes[scope.scope_id].record_log(
            formatted_message,
            level=level,
            exception=exception,
        )

    def event_recording(
        scope: ContextIdentifier,
        /,
        level: ObservabilityLevel,
        *,
        event: str,
        attributes: Mapping[str, ObservabilityAttribute],
    ) -> None:
        assert root_scope is not None  # nosec: B101
        assert scope.scope_id in scopes  # nosec: B101

        if level < log_level:
            return

        scopes[scope.scope_id].record_event(
            event,
            level=level,
            attributes=attributes,
        )

    def metric_recording(
        scope: ContextIdentifier,
        /,
        level: ObservabilityLevel,
        *,
        metric: str,
        value: float | int,
        unit: str | None,
        kind: ObservabilityMetricKind,
        attributes: Mapping[str, ObservabilityAttribute],
    ) -> None:
        assert root_scope is not None  # nosec: B101
        assert scope.scope_id in scopes  # nosec: B101

        if level < log_level:
            return

        scopes[scope.scope_id].record_metric(
            metric,
            value=value,
            unit=unit,
            kind=kind,
            attributes=attributes,
        )

    def attributes_recording(
        scope: ContextIdentifier,
        /,
        level: ObservabilityLevel,
        attributes: Mapping[str, ObservabilityAttribute],
    ) -> None:
        assert root_scope is not None  # nosec: B101
        assert scope.scope_id in scopes  # nosec: B101

        if level < log_level:
            return

        if not attributes:
            return

        scopes[scope.scope_id].record_attributes(
            attributes,
            level=level,
        )

    def scope_entering(
        scope: ContextIdentifier,
        /,
    ) -> str:
        nonlocal root_scope
        assert scope.scope_id not in scopes  # nosec: B101

        scope_store: ScopeStore
        if root_scope is None:
            assert scope.is_root  # nosec: B101
            scope_store = ScopeStore(
                scope,
                trace_id=trace_id,
                log_group=log_group,
                log_stream=log_stream,
                event_bus=event_bus,
                event_source=event_source,
                metrics_namespace=metrics_namespace,
                log_putting=log_putting,
                metric_putting=metric_putting,
                event_putting=event_putting,
            )
            root_scope = scope

        else:
            assert scope.parent_id in scopes  # nosec: B101
            scope_store = scopes[scope.parent_id].child(scope)

        scopes[scope.scope_id] = scope_store

        scope_store.record_scope_start()

        return str(scope_store.identifier.scope_id)

    def scope_exiting(
        scope: ContextIdentifier,
        /,
        *,
        exception: BaseException | None,
    ) -> None:
        nonlocal root_scope
        nonlocal trace_id
        assert root_scope is not None  # nosec: B101
        assert scope.scope_id in scopes  # nosec: B101

        scope_store: ScopeStore = scopes[scope.scope_id]
        scope_store.exit()
        scope_store.record_scope_end(exception)

        if not scope_store.try_complete():
            return  # not completed yet or already completed

        # try complete parent scopes
        if scope != root_scope:
            parent_id: UUID = scope.parent_id
            while scopes[parent_id].try_complete():
                if scopes[parent_id].identifier == root_scope:
                    break

                parent_id = scopes[parent_id].identifier.parent_id

        # check for root completion
        if scopes[root_scope.scope_id].completed:
            trace_id = uuid4()
            root_scope = None
            scopes.clear()

    return Observability(
        trace_identifying=trace_identifying,
        log_recording=log_recording,
        metric_recording=metric_recording,
        event_recording=event_recording,
        attributes_recording=attributes_recording,
        scope_entering=scope_entering,
        scope_exiting=scope_exiting,
    )


class ScopeStore:
    __slots__ = (
        "_completion_time",
        "_end_time",
        "_start_time",
        "event_bus",
        "event_putting",
        "event_source",
        "identifier",
        "log_group",
        "log_putting",
        "log_stream",
        "metric_putting",
        "metrics_namespace",
        "nested",
        "trace_id",
    )

    def __init__(
        self,
        identifier: ContextIdentifier,
        /,
        *,
        trace_id: UUID,
        log_group: str,
        log_stream: str,
        event_bus: str,
        event_source: str,
        metrics_namespace: str,
        log_putting: AWSCloudwatchLogPutting,
        metric_putting: AWSCloudwatchMetricPutting,
        event_putting: AWSCloudwatchEventPutting,
    ) -> None:
        self.identifier: ContextIdentifier = identifier
        self.trace_id: UUID = trace_id
        self.log_group: str = log_group
        self.log_stream: str = log_stream
        self.event_bus: str = event_bus
        self.event_source: str = event_source
        self.metrics_namespace: str = metrics_namespace
        self.log_putting: AWSCloudwatchLogPutting = log_putting
        self.metric_putting: AWSCloudwatchMetricPutting = metric_putting
        self.event_putting: AWSCloudwatchEventPutting = event_putting
        self.nested: MutableSequence[ScopeStore] = []
        self._start_time: float = monotonic()
        self._end_time: float | None = None
        self._completion_time: float | None = None

    def child(
        self,
        identifier: ContextIdentifier,
        /,
    ) -> Self:
        child: Self = self.__class__(
            identifier,
            trace_id=self.trace_id,
            log_group=self.log_group,
            log_stream=self.log_stream,
            event_bus=self.event_bus,
            event_source=self.event_source,
            metrics_namespace=self.metrics_namespace,
            log_putting=self.log_putting,
            metric_putting=self.metric_putting,
            event_putting=self.event_putting,
        )
        self.nested.append(child)
        return child

    @property
    def exited(self) -> bool:
        return self._end_time is not None

    def exit(self) -> None:
        assert self._end_time is None  # nosec: B101
        self._end_time = monotonic()

    @property
    def completed(self) -> bool:
        return self._completion_time is not None and all(nested.completed for nested in self.nested)

    def try_complete(self) -> bool:
        if self._end_time is None:
            return False  # not elegible for completion yet

        if self._completion_time is not None:
            return False  # already completed

        if not all(nested.completed for nested in self.nested):
            return False  # nested not completed

        self._completion_time = monotonic()
        return True  # successfully completed

    def record_log(
        self,
        message: str,
        /,
        *,
        level: ObservabilityLevel,
        exception: BaseException | None,
    ) -> None:
        attributes: MutableMapping[str, Any] = {
            "scope.name": self.identifier.name,
            "scope.id": self.identifier.scope_id.hex,
        }
        parent_span_id: str | None = _parent_span_id(self.identifier)
        if parent_span_id is not None:
            attributes["span.parent_id"] = parent_span_id

        if exception is not None:
            attributes["exception.type"] = exception.__class__.__name__
            attributes["exception.message"] = str(exception)

        ctx.spawn_background(
            self.log_putting,
            log_stream=self.log_stream,
            log_group=self.log_group,
            message=_json_dumps(
                {
                    "time_unix_nano": time_ns(),
                    "severity_text": level.name,
                    "severity_number": level.value,
                    "body": message,
                    "trace_id": self.trace_id.hex,
                    "span_id": _span_id(self.identifier),
                    "attributes": _sanitized_attributes(attributes),
                }
            ),
        )

    def record_exception(
        self,
        exception: BaseException,
        /,
    ) -> None:
        ctx.spawn_background(
            self.event_putting,
            event_bus=self.event_bus,
            event_source=self.event_source,
            detail_type="exception",
            detail=_json_dumps(
                {
                    "time_unix_nano": time_ns(),
                    "trace_id": self.trace_id.hex,
                    "span_id": _span_id(self.identifier),
                    "name": "exception",
                    "attributes": _sanitized_attributes(
                        {
                            "scope.name": self.identifier.name,
                            "scope.id": self.identifier.scope_id.hex,
                            "exception.type": exception.__class__.__name__,
                            "exception.message": str(exception),
                            "exception.stacktrace": "".join(
                                traceback.format_exception(
                                    exception.__class__,
                                    exception,
                                    exception.__traceback__,
                                )
                            ),
                        }
                    ),
                }
            ),
        )

    def record_event(
        self,
        event: str,
        /,
        *,
        level: ObservabilityLevel,
        attributes: Mapping[str, ObservabilityAttribute],
    ) -> None:
        ctx.spawn_background(
            self.event_putting,
            event_bus=self.event_bus,
            event_source=self.event_source,
            detail_type="event",
            detail=_json_dumps(
                {
                    "time_unix_nano": time_ns(),
                    "trace_id": self.trace_id.hex,
                    "span_id": _span_id(self.identifier),
                    "severity_text": level.name,
                    "severity_number": level.value,
                    "name": event,
                    "attributes": _sanitized_attributes(
                        {
                            "scope.name": self.identifier.name,
                            "scope.id": self.identifier.scope_id.hex,
                            **attributes,
                        }
                    ),
                }
            ),
        )

    def record_metric(
        self,
        name: str,
        /,
        *,
        value: float | int,
        unit: str | None,
        kind: ObservabilityMetricKind,
        attributes: Mapping[str, ObservabilityAttribute],
    ) -> None:
        metric_attributes: dict[str, ObservabilityAttribute] = dict(
            _sanitized_attributes(attributes)
        )
        metric_attributes.setdefault("otel.metric.kind", kind)
        ctx.spawn_background(
            self.metric_putting,
            namespace=self.metrics_namespace,
            metric=name,
            value=value,
            unit=unit,
            attributes=metric_attributes,
        )

    def record_attributes(
        self,
        attributes: Mapping[str, ObservabilityAttribute],
        /,
        *,
        level: ObservabilityLevel,
    ) -> None:
        if not attributes:
            return

        self.record_event(
            "attributes",
            level=level,
            attributes=attributes,
        )

    def record_scope_start(self) -> None:
        self.record_event(
            "span.start",
            level=ObservabilityLevel.INFO,
            attributes={"span.name": self.identifier.name},
        )

    def record_scope_end(
        self,
        exception: BaseException | None,
        /,
    ) -> None:
        if exception is not None:
            self.record_exception(exception)

        self.record_event(
            "span.end",
            level=ObservabilityLevel.INFO,
            attributes={
                "span.name": self.identifier.name,
                "status.code": "ERROR" if exception is not None else "OK",
            },
        )


def _span_id(
    identifier: ContextIdentifier,
    /,
) -> str:
    return identifier.scope_id.hex[-16:]


def _parent_span_id(
    identifier: ContextIdentifier,
    /,
) -> str | None:
    if identifier.is_root:
        return None

    return identifier.parent_id.hex[-16:]


def _sanitized_attributes(
    attributes: Mapping[str, Any],
    /,
) -> Mapping[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None or value is MISSING:
            continue  # skip missing/empty

        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            elements: list[Any] = []
            for item in cast(Sequence[Any], value):
                if item is None or item is MISSING:
                    continue  # skip missing/empty

                assert isinstance(item, str | float | int | bool)  # nosec: B101
                elements.append(item)

            if not elements:
                continue  # skip missing/empty

            sanitized[key] = tuple(elements)

        elif isinstance(value, Mapping):
            for name, item in cast(Mapping[str, Any], value).items():
                if item is None or item is MISSING:
                    continue  # skip missing/empty

                assert isinstance(item, str | float | int | bool)  # nosec: B101
                sanitized[f"{key}.{name}"] = item

        else:
            assert isinstance(value, str | float | int | bool)  # nosec: B101
            sanitized[key] = value

    return sanitized


def _json_dumps(
    payload: Mapping[str, Any],
    /,
) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )
