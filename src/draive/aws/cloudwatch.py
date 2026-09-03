from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from time import time_ns
from typing import Any, Final

from botocore.exceptions import ClientError  # pyright: ignore[reportMissingModuleSource]
from haiway import MISSING, ObservabilityAttribute, asynchronous

from draive.aws.api import AWSAPI
from draive.aws.types import (
    AWSAccessDenied,
    AWSError,
    AWSResourceNotFound,
)

__all__ = ("AWSCloudwatchMixin",)


# CloudWatch accepts only its predefined units, anything else fails the whole request
_STANDARD_METRIC_UNITS: Final[Sequence[str]] = (
    "Seconds",
    "Microseconds",
    "Milliseconds",
    "Bytes",
    "Kilobytes",
    "Megabytes",
    "Gigabytes",
    "Terabytes",
    "Bits",
    "Kilobits",
    "Megabits",
    "Gigabits",
    "Terabits",
    "Percent",
    "Count",
    "Bytes/Second",
    "Kilobytes/Second",
    "Megabytes/Second",
    "Gigabytes/Second",
    "Terabytes/Second",
    "Bits/Second",
    "Kilobits/Second",
    "Megabits/Second",
    "Gigabits/Second",
    "Terabits/Second",
    "Count/Second",
    "None",
)
_METRIC_UNITS: Final[Mapping[str, str]] = {
    **{unit.lower(): unit for unit in _STANDARD_METRIC_UNITS},
    # unit names used within draive without a matching CloudWatch unit
    "%": "Percent",
    "tokens": "None",  # there is no unit suitable for tokens
}
_DIMENSIONS_LIMIT: Final[int] = 30
_DIMENSION_NAME_LIMIT: Final[int] = 255
_DIMENSION_VALUE_LIMIT: Final[int] = 1024


class AWSCloudwatchMixin(AWSAPI):
    def __init__(
        self,
        region_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        profile_name: str | None = None,
    ) -> None:
        super().__init__(
            region_name=region_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            profile_name=profile_name,
        )

    async def put_log(
        self,
        *,
        log_group: str,
        log_stream: str,
        message: str,
    ) -> None:
        await self._put_log(
            log_group=log_group,
            log_stream=log_stream,
            message=message,
        )

    @asynchronous
    def _put_log(
        self,
        *,
        log_group: str,
        log_stream: str,
        message: str,
    ) -> None:
        try:
            self._cloudwatch_logs_client.put_log_events(
                logGroupName=log_group,
                logStreamName=log_stream,
                logEvents=[
                    {
                        "timestamp": time_ns() // 1_000_000,  # epoch ms (UTC)
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

    async def put_event(
        self,
        *,
        event_bus: str,
        event_source: str,
        detail_type: str,
        detail: str,
    ) -> None:
        await self._put_event(
            event_bus=event_bus,
            event_source=event_source,
            detail_type=detail_type,
            detail=detail,
        )

    @asynchronous
    def _put_event(
        self,
        *,
        event_bus: str,
        event_source: str,
        detail_type: str,
        detail: str,
    ) -> None:
        try:
            self._eventbridge_client.put_events(
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

    async def put_metric(
        self,
        *,
        namespace: str,
        metric: str,
        value: float | int,
        unit: str | None,
        attributes: Mapping[str, ObservabilityAttribute],
    ) -> None:
        await self._put_metric(
            namespace=namespace,
            metric=metric,
            value=value,
            unit=unit,
            attributes=attributes,
        )

    @asynchronous
    def _put_metric(
        self,
        *,
        namespace: str,
        metric: str,
        value: float | int,
        unit: str | None,
        attributes: Mapping[str, ObservabilityAttribute],
    ) -> None:
        try:
            dimensions = format_metric_dimensions(attributes)
            metric_data: dict[str, Any] = {
                "MetricName": metric,
                "Value": value,
            }

            if dimensions:
                metric_data["Dimensions"] = dimensions

            if metric_unit := format_metric_unit(unit):
                metric_data["Unit"] = metric_unit

            self._cloudwatch_client.put_metric_data(
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


def format_metric_unit(
    unit: str | None,
) -> str | None:
    """Convert a metric unit to its CloudWatch counterpart when available."""

    if not unit:
        return None

    # unmapped units have to be skipped instead of failing the whole request
    return _METRIC_UNITS.get(unit.strip().lower())


def format_metric_dimensions(
    attributes: Mapping[str, ObservabilityAttribute],
) -> list[dict[str, str]]:
    dimensions: list[dict[str, str]] = []
    for key, value in _sanitize_attributes(attributes).items():
        if len(dimensions) >= _DIMENSIONS_LIMIT:
            break  # only a limited number of dimensions is allowed

        if not key:
            continue  # empty dimension names are not allowed

        rendered: str | None = _render_dimension_value(value)
        if not rendered:
            continue  # empty dimension values are not allowed

        dimensions.append(
            {
                "Name": _truncate_dimension(key, limit=_DIMENSION_NAME_LIMIT),
                "Value": _truncate_dimension(rendered, limit=_DIMENSION_VALUE_LIMIT),
            }
        )

    return dimensions


def _render_dimension_value(
    value: ObservabilityAttribute,
) -> str | None:
    if value is None or value is MISSING:
        return None

    if isinstance(value, str):
        return value

    elif isinstance(value, Sequence):
        items: list[str] = []
        for item in value:
            items.append(str(item))

        if not items:
            return None

        return ",".join(items)

    return str(value)


def _sanitize_attributes(
    attributes: Mapping[str, ObservabilityAttribute],
) -> Mapping[str, ObservabilityAttribute]:
    sanitized: dict[str, ObservabilityAttribute] = {}
    for key, value in attributes.items():
        if value is None or value is MISSING:
            continue

        if isinstance(value, str):
            sanitized[key] = value

        elif isinstance(value, Sequence):
            if not value:
                continue

            sanitized[key] = value

        else:
            sanitized[key] = value

    return sanitized


def _truncate_dimension(
    value: str,
    *,
    limit: int,
) -> str:
    if len(value) <= limit:
        return value

    return value[:limit]


def translate_cloudwatch_error(
    *,
    error: ClientError,
    service: str,
    operation: str,
    resource: str,
) -> Exception:
    error_info: Mapping[str, Any] = getattr(error, "response", {}).get("Error", {})
    response_metadata: Mapping[str, Any] = getattr(error, "response", {}).get(
        "ResponseMetadata", {}
    )

    code: str = str(error_info.get("Code") or "").strip()
    message: str = str(error_info.get("Message") or str(error)).strip()
    status_code: int | None = response_metadata.get("HTTPStatusCode")

    normalized_code: str = code.lower()
    uri: str = f"{service}://{resource}"
    contextual_message: str = (
        f"{message} [provider=aws_{service} operation={operation} resource={resource}]"
    )

    if normalized_code in {
        "accessdenied",
        "accessdeniedexception",
        "invalidaccesskeyid",
        "signaturedoesnotmatch",
    } or status_code in {401, 403}:
        return AWSAccessDenied(
            uri=f"{service}://{resource}",
            code=code or None,
            message=contextual_message,
        )

    if (
        status_code == 404 or normalized_code in {"resourcenotfoundexception", "notfound"}  # noqa: PLR2004
    ):
        return AWSResourceNotFound(
            uri=uri,
            code=code or None,
            message=contextual_message,
        )

    return AWSError(
        uri=uri,
        code=code or None,
        message=contextual_message,
    )
