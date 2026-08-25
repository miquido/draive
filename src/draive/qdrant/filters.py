from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any, cast, overload
from uuid import UUID

from haiway import AttributeRequirement, State
from qdrant_client.models import (
    Condition,
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchAny,
    MatchText,
    MatchValue,
    Range,
)

from draive.utils.attributes import attribute_path_segments

__all__ = ("prepare_filter",)


@overload
def prepare_filter[Model: State](
    requirements: AttributeRequirement[Model] | None,
) -> Filter | None: ...


@overload
def prepare_filter[Model: State](
    requirements: AttributeRequirement[Model] | None,
    *,
    default: Filter,
) -> Filter: ...


def prepare_filter[Model: State](
    requirements: AttributeRequirement[Model] | None,
    *,
    default: Filter | None = None,
) -> Filter | None:
    if requirements := requirements:
        return _convert(requirements)

    else:
        return default


def _convert[Model: State](  # noqa: PLR0911
    requirements: AttributeRequirement[Model],
    /,
) -> Filter:
    match requirements.operator:
        case "equal":
            return Filter(
                must=[
                    _match_condition(
                        _payload_key(requirements.lhs),
                        requirements.rhs,
                    )
                ]
            )

        case "text_match":
            return Filter(
                must=[
                    FieldCondition(
                        key=_payload_key(requirements.lhs),
                        match=MatchText(text=requirements.rhs),
                    )
                ]
            )

        case "not_equal":
            return Filter(
                must_not=[
                    _match_condition(
                        _payload_key(requirements.lhs),
                        requirements.rhs,
                    )
                ]
            )

        case "contained_in":
            # 'contained_in' is the only operator built with its operands swapped -
            # `lhs` holds the collection of allowed values, `rhs` holds the path
            return _any_filter(
                _payload_key(requirements.rhs),
                cast(Iterable[Any], requirements.lhs),
            )

        case "contains_any":
            return _any_filter(
                _payload_key(requirements.lhs),
                cast(Iterable[Any], requirements.rhs),
            )

        case "and":
            return Filter(
                must=[
                    _convert(requirements.lhs),
                    _convert(requirements.rhs),
                ]
            )

        case "or":
            return Filter(
                should=[
                    _convert(requirements.lhs),
                    _convert(requirements.rhs),
                ]
            )

        case "contains":
            # matching an array payload field against a single value
            # matches when any of its elements is equal to that value
            return Filter(
                must=[
                    _match_condition(
                        _payload_key(requirements.lhs),
                        requirements.rhs,
                    )
                ]
            )


def _payload_key(
    path: object,
    /,
) -> str:
    # payloads are stored serialized, attribute aliases have to be applied
    return ".".join(attribute_path_segments(path))


def _match_condition(
    path: str,
    value: Any,
    /,
) -> FieldCondition:
    match value:
        case bool() | str():
            return FieldCondition(
                key=path,
                match=MatchValue(value=value),
            )

        case int() | float():
            # `MatchValue` accepts strictly bool, int and str, and matches an integer
            # only against an integer payload - a float payload of an equal value is
            # not matched at all. A degenerate range compares numbers by value,
            # which covers both payload representations.
            return FieldCondition(
                key=path,
                range=Range(
                    gte=value,
                    lte=value,
                ),
            )

        case datetime() | date():
            # payload datetimes are stored using their own representation, only a
            # datetime range compares them by value instead of by exact text
            return FieldCondition(
                key=path,
                range=DatetimeRange(
                    gte=value,
                    lte=value,
                ),
            )

        case UUID():
            return FieldCondition(
                key=path,
                match=MatchValue(value=str(value)),
            )

        case _:
            raise NotImplementedError(
                f"Unsupported Qdrant requirement value: {type(value).__name__}"
            )


def _any_filter(
    path: str,
    values: Iterable[Any],
    /,
) -> Filter:
    conditions: Sequence[FieldCondition] = tuple(_match_condition(path, value) for value in values)
    match_values: Sequence[bool | int | str] = tuple(
        condition.match.value for condition in conditions if isinstance(condition.match, MatchValue)
    )
    if len(match_values) == len(conditions):
        # a single MatchAny is served better by a payload index than a disjunction,
        # numbers are excluded from it - they are matched through ranges
        return Filter(
            must=[
                FieldCondition(
                    key=path,
                    match=MatchAny(any=cast(Any, list(match_values))),
                )
            ]
        )

    # numeric range conditions cannot be expressed by MatchAny
    return Filter(should=cast(list[Condition], list(conditions)))
