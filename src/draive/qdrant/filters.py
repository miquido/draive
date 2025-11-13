from typing import overload

from haiway import AttributeRequirement
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchText, MatchValue

from draive.parameters import DataModel

__all__ = ("prepare_filter",)


@overload
def prepare_filter[Model: DataModel](
    requirements: AttributeRequirement[Model] | None,
) -> Filter | None: ...


@overload
def prepare_filter[Model: DataModel](
    requirements: AttributeRequirement[Model] | None,
    *,
    default: Filter,
) -> Filter: ...


def prepare_filter[Model: DataModel](
    requirements: AttributeRequirement[Model] | None,
    *,
    default: Filter | None = None,
) -> Filter | None:
    if requirements := requirements:
        return _convert(requirements)

    else:
        return default


def _convert[Model: DataModel](  # noqa: PLR0911
    requirements: AttributeRequirement[Model],
    /,
) -> Filter:
    match requirements.operator:
        case "equal":
            return Filter(
                must=[
                    FieldCondition(
                        key=str(requirements.lhs),
                        match=MatchValue(value=requirements.rhs),
                    )
                ]
            )

        case "text_match":
            return Filter(
                must=[
                    FieldCondition(
                        key=str(requirements.lhs),
                        match=MatchText(text=requirements.rhs),
                    )
                ]
            )

        case "not_equal":
            return Filter(
                must_not=[
                    FieldCondition(
                        key=str(requirements.lhs),
                        match=MatchValue(value=requirements.rhs),
                    )
                ]
            )

        case "contained_in":
            return Filter(
                must=[
                    FieldCondition(
                        key=str(requirements.rhs),
                        match=MatchAny(any=requirements.lhs),
                    )
                ]
            )

        case "contains_any":
            return Filter(
                must=[
                    FieldCondition(
                        key=str(requirements.lhs),
                        match=MatchAny(any=requirements.rhs),
                    )
                ]
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
            raise NotImplementedError("'contains' requirement is supported with Qdrant")
