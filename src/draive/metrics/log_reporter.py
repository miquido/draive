from logging import Logger
from typing import Any, cast

from draive.metrics.reporter import MetricsTraceReport, MetricsTraceReporter
from draive.parameters import ParametrizedData
from draive.utils import is_missing

__all__ = [
    "metrics_log_reporter",
]


def metrics_log_reporter(
    list_items_limit: int | None = None,
    item_character_limit: int | None = None,
) -> MetricsTraceReporter:
    async def reporter(
        trace_id: str,
        logger: Logger,
        report: MetricsTraceReport,
    ) -> None:
        report_log: str | None = _report(
            report.with_combined_metrics(),
            list_items_limit=list_items_limit,
            item_character_limit=item_character_limit,
        )
        logger.info(f"[{trace_id}] Metrics report:\n{report_log or 'N/A'}")

    return reporter


def _report(
    report: MetricsTraceReport,
    list_items_limit: int | None,
    item_character_limit: int | None,
) -> str:
    report_log: str = f"@{report.label}({report.duration:.2f}s):"
    for metric_name, metric in report.metrics.items():
        metric_log: str = ""
        for key, value in vars(metric).items():
            if value_log := _value_report(
                value,
                list_items_limit=list_items_limit,
                item_character_limit=item_character_limit,
            ):
                metric_log += f"\n|  + {key}: {value_log}"

            else:
                continue  # skip missing values

        if not metric_log:
            continue  # skip empty logs

        report_log += f"\n• {metric_name}:{metric_log}"

    for nested in report.nested:
        nested_log: str = _report(
            nested,
            list_items_limit=list_items_limit,
            item_character_limit=item_character_limit,
        ).replace("\n", "\n|  ")

        report_log += f"\n{nested_log}"

    return report_log.strip()


def _state_report(
    value: ParametrizedData,
    /,
    list_items_limit: int | None,
    item_character_limit: int | None,
) -> str | None:
    state_log: str = ""
    for key, element in vars(value).items():
        element_log: str | None = _value_report(
            element,
            list_items_limit=list_items_limit,
            item_character_limit=item_character_limit,
        )
        if element_log:
            state_log += f"\n|  + {key}: {element_log}"
        else:
            continue  # skip empty logs

    if state_log:
        return state_log.replace("\n", "\n|  ")
    else:
        return None  # skip empty logs


def _dict_report(
    value: dict[Any, Any],
    /,
    list_items_limit: int | None,
    item_character_limit: int | None,
) -> str | None:
    dict_log: str = ""
    for key, element in value.items():
        element_log: str | None = _value_report(
            element,
            list_items_limit=list_items_limit,
            item_character_limit=item_character_limit,
        )
        if element_log:
            dict_log += f"\n|  + {key}: {element_log}"
        else:
            continue  # skip empty logs

    if dict_log:
        return dict_log.replace("\n", "\n|  ")
    else:
        return None  # skip empty logs


def _list_report(
    value: list[Any],
    /,
    list_items_limit: int | None,
    item_character_limit: int | None,
) -> str | None:
    list_log: str = ""
    enumerared: list[tuple[int, Any]] = list(enumerate(value))
    if list_items_limit:
        if list_items_limit > 0:
            enumerared = enumerared[:list_items_limit]
        else:
            enumerared = enumerared[list_items_limit:]

    for idx, element in enumerared:
        element_log: str | None = _value_report(
            element,
            list_items_limit=list_items_limit,
            item_character_limit=item_character_limit,
        )
        if element_log:
            list_log += f"\n|  [{idx}] {element_log}"
        else:
            continue  # skip empty logs

    if list_log:
        return list_log.replace("\n", "\n|  ")
    else:
        return None  # skip empty logs


def _raw_value_report(
    value: Any,
    /,
    list_items_limit: int | None,
    item_character_limit: int | None,
) -> str | None:
    if is_missing(value):
        return None  # skip missing

    # workaround for pydantic models
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _dict_report(
            value.model_dump(),
            list_items_limit=list_items_limit,
            item_character_limit=item_character_limit,
        )

    else:
        value_log = str(value)
        if not value_log:
            return None  # skip empty logs

        if (item_character_limit := item_character_limit) and len(value_log) > item_character_limit:
            return value_log.replace("\n", " ")[:item_character_limit] + "..."

        else:
            return value_log.replace("\n", "\n|  ")


def _value_report(
    value: Any,
    /,
    list_items_limit: int | None,
    item_character_limit: int | None,
) -> str | None:
    # try unpack dicts
    if isinstance(value, dict):
        return _dict_report(
            cast(dict[Any, Any], value),
            list_items_limit=list_items_limit,
            item_character_limit=item_character_limit,
        )

    # try unpack lists
    elif isinstance(value, list):
        return _list_report(
            cast(list[Any], value),
            list_items_limit=list_items_limit,
            item_character_limit=item_character_limit,
        )

    # try unpack state
    elif isinstance(value, ParametrizedData):
        return _state_report(
            value,
            list_items_limit=list_items_limit,
            item_character_limit=item_character_limit,
        )

    else:
        return _raw_value_report(
            value,
            list_items_limit=list_items_limit,
            item_character_limit=item_character_limit,
        )
