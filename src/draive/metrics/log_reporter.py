from logging import Logger
from typing import Any, cast

from draive.metrics.reporter import MetricsTraceReport
from draive.types import ParametrizedState, is_missing

__all__ = [
    "metrics_log_report",
    "metrics_trimmed_log_report",
]


async def metrics_log_report(
    trace_id: str,
    logger: Logger,
    report: MetricsTraceReport,
) -> None:
    logger.info(
        f"[{trace_id}] Metrics report:"
        f"\n{_report(report.with_combined_metrics(), character_limit=None)}"
    )


async def metrics_trimmed_log_report(
    trace_id: str,
    logger: Logger,
    report: MetricsTraceReport,
) -> None:
    logger.info(
        f"[{trace_id}] Metrics report:"
        f"\n{_report(report.with_combined_metrics(), character_limit=64)}"
    )


def _report(
    report: MetricsTraceReport,
    character_limit: int | None,
) -> str:
    report_log: str = f"@{report.label}({report.duration:.2f}s):"
    for metric_type, metric in report.metrics.items():
        metric_log: str = ""
        for key, value in vars(metric).items():
            if value_log := _value_report(value, character_limit=character_limit):
                metric_log += f"\n|  + {key}: {value_log}"

            else:
                continue  # skip missing values

        if not metric_log:
            continue  # skip empty logs

        report_log += f"\n• {metric_type.__name__}:{metric_log}"

    for nested in report.nested:
        nested_log: str = _report(
            nested,
            character_limit=character_limit,
        ).replace("\n", "\n|  ")

        report_log += f"\n{nested_log}"

    return report_log.strip()


def _state_report(
    value: ParametrizedState,
    /,
    character_limit: int | None,
) -> str | None:
    state_log: str = ""
    for key, element in vars(value).items():
        state_log += f"\n|  + {key}: {_value_report(element, character_limit=character_limit)}"

    return state_log.replace("\n", "\n|  ")


def _dict_report(
    value: dict[Any, Any],
    /,
    character_limit: int | None,
) -> str | None:
    dict_log: str = ""
    for key, element in value.items():
        dict_log += f"\n|  + {key}: {_value_report(element, character_limit=character_limit)}"

    return dict_log.replace("\n", "\n|  ")


def _list_report(
    value: list[Any],
    /,
    character_limit: int | None,
) -> str | None:
    list_log: str = ""
    for idx, element in enumerate(value):
        list_log += f"\n|  [{idx}] {_value_report(element, character_limit=character_limit)}"

    return list_log.replace("\n", "\n|  ")


def _raw_value_report(
    value: Any,
    /,
    character_limit: int | None,
) -> str | None:
    if is_missing(value):
        return None  # skip missing

    # workaround for pydantic models
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _dict_report(
            value.model_dump(),
            character_limit=character_limit,
        )

    else:
        value_log = str(value)
        if not value_log:
            return None  # skip empty logs

        if (character_limit := character_limit) and len(value_log) > character_limit:
            return value_log.replace("\n", " ")[:character_limit] + "..."

        else:
            return value_log.replace("\n", "\n|  ")


def _value_report(
    value: Any,
    /,
    character_limit: int | None,
) -> str | None:
    # try unpack dicts
    if isinstance(value, dict):
        return _dict_report(
            cast(dict[Any, Any], value),
            character_limit=character_limit,
        )

    # try unpack lists
    elif isinstance(value, list):
        return _list_report(
            cast(list[Any], value),
            character_limit=character_limit,
        )

    # try unpack state
    elif isinstance(value, ParametrizedState):
        return _state_report(
            value,
            character_limit=character_limit,
        )

    else:
        return _raw_value_report(value, character_limit=character_limit)
