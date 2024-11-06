from collections.abc import Callable, Coroutine
from logging import INFO
from typing import Any, cast

from haiway import Missing, ScopeMetrics, State, is_missing

from draive.metrics.tokens import TokenUsage

__all__ = [
    "usage_metrics_logger",
]


def usage_metrics_logger(
    list_items_limit: int | None = None,
    item_character_limit: int | None = None,
) -> Callable[[ScopeMetrics], Coroutine[None, None, None]]:
    async def logger(metrics: ScopeMetrics) -> None:
        report_log: str | None = _usage_log(
            metrics,
            list_items_limit=list_items_limit,
            item_character_limit=item_character_limit,
        )
        metrics.log(
            INFO,
            f"Usage metrics:\n{report_log or 'N/A'}",
        )

    return logger


def _usage_metrics_merge(
    current: State | Missing,
    nested: State,
) -> State | Missing:
    if not isinstance(nested, TokenUsage):
        return current  # do not merge other metrics than TokenUsage

    if not isinstance(current, TokenUsage):
        return nested  # use nested TokenUsage if missing current

    return current + nested  # add TokenUsage from current and nested


def _usage_log(
    metrics: ScopeMetrics,
    list_items_limit: int | None,
    item_character_limit: int | None,
) -> str:
    log: str = f"@{metrics.label}[{metrics.identifier}]({metrics.time:.2f}s):"

    for metric in metrics.metrics(merge=_usage_metrics_merge):
        metric_log: str = ""
        for key, value in vars(metric).items():
            if value_log := _value_log(
                value,
                list_items_limit=list_items_limit,
                item_character_limit=item_character_limit,
            ):
                metric_log += f"\n|  + {key}: {value_log}"

            else:
                continue  # skip missing values

        if not metric_log:
            continue  # skip empty logs

        log += f"\n• {type(metric).__qualname__}:{metric_log}"

    # making use of private api of haiway - draive is very entangled already
    for nested in metrics._nested:  # pyright: ignore[reportPrivateUsage]
        nested_log: str = _usage_log(
            nested,
            list_items_limit=list_items_limit,
            item_character_limit=item_character_limit,
        ).replace("\n", "\n|  ")

        log += f"\n{nested_log}"

    return log.strip()


def _state_log(
    value: State,
    /,
    list_items_limit: int | None,
    item_character_limit: int | None,
) -> str | None:
    state_log: str = ""
    for key, element in vars(value).items():
        element_log: str | None = _value_log(
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


def _dict_log(
    value: dict[Any, Any],
    /,
    list_items_limit: int | None,
    item_character_limit: int | None,
) -> str | None:
    dict_log: str = ""
    for key, element in value.items():
        element_log: str | None = _value_log(
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


def _list_log(
    value: list[Any],
    /,
    list_items_limit: int | None,
    item_character_limit: int | None,
) -> str | None:
    list_log: str = ""
    enumerated: list[tuple[int, Any]] = list(enumerate(value))
    if list_items_limit:
        if list_items_limit > 0:
            enumerated = enumerated[:list_items_limit]

        else:
            enumerated = enumerated[list_items_limit:]

    for idx, element in enumerated:
        element_log: str | None = _value_log(
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


def _raw_value_log(
    value: Any,
    /,
    item_character_limit: int | None,
) -> str | None:
    if is_missing(value):
        return None  # skip missing

    value_log = str(value)
    if not value_log:
        return None  # skip empty logs

    if (item_character_limit := item_character_limit) and len(value_log) > item_character_limit:
        return value_log.replace("\n", " ")[:item_character_limit] + "..."

    else:
        return value_log.replace("\n", "\n|  ")


def _value_log(
    value: Any,
    /,
    list_items_limit: int | None,
    item_character_limit: int | None,
) -> str | None:
    # try unpack dicts
    if isinstance(value, dict):
        return _dict_log(
            cast(dict[Any, Any], value),
            list_items_limit=list_items_limit,
            item_character_limit=item_character_limit,
        )

    # try unpack lists
    elif isinstance(value, list):
        return _list_log(
            cast(list[Any], value),
            list_items_limit=list_items_limit,
            item_character_limit=item_character_limit,
        )

    # try unpack state
    elif isinstance(value, State):
        return _state_log(
            value,
            list_items_limit=list_items_limit,
            item_character_limit=item_character_limit,
        )

    else:
        return _raw_value_log(
            value,
            item_character_limit=item_character_limit,
        )
