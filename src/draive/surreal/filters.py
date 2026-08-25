from collections.abc import Mapping, Sequence

from haiway import AttributeRequirement, State

from draive.surreal.types import SurrealValue
from draive.surreal.utils import surreal_value
from draive.utils.attributes import attribute_path_segments

__all__ = ("prepare_filter",)


def prepare_filter[Model: State](
    requirements: AttributeRequirement[Model] | None,
    /,
) -> tuple[str, Mapping[str, SurrealValue]]:
    if requirements is None:
        return ("", {})

    return _convert(requirements, index=0)


def _convert[Model: State](  # noqa: C901, PLR0911
    requirements: AttributeRequirement[Model],
    /,
    *,
    index: int,
) -> tuple[str, Mapping[str, SurrealValue]]:
    match requirements.operator:
        case "and":
            left_clause, left_values = _convert(requirements.lhs, index=index)
            right_clause, right_values = _convert(
                requirements.rhs,
                index=index + len(left_values),
            )
            return (
                f"({left_clause}) AND ({right_clause})",
                {**left_values, **right_values},
            )

        case "or":
            left_clause, left_values = _convert(requirements.lhs, index=index)
            right_clause, right_values = _convert(
                requirements.rhs,
                index=index + len(left_values),
            )
            return (
                f"({left_clause}) OR ({right_clause})",
                {**left_values, **right_values},
            )

        case "equal":
            parameter: str = f"_f{index}"
            return (
                _field_reference(requirements.lhs) + f" = ${parameter}",
                {parameter: surreal_value(requirements.rhs)},
            )

        case "not_equal":
            parameter: str = f"_f{index}"
            return (
                _field_reference(requirements.lhs) + f" != ${parameter}",
                {parameter: surreal_value(requirements.rhs)},
            )

        case "contained_in":
            # 'contained_in' is the only operator built with its operands swapped
            parameter: str = f"_f{index}"
            values: SurrealValue = surreal_value(requirements.lhs)
            if not isinstance(values, Sequence) or isinstance(values, str | bytes):
                raise ValueError("'contained_in' requires a sequence")

            return (
                _field_reference(requirements.rhs) + f" INSIDE ${parameter}",
                {parameter: values},
            )

        case "contains_any":
            parameter: str = f"_f{index}"
            values: SurrealValue = surreal_value(requirements.rhs)
            if not isinstance(values, Sequence) or isinstance(values, str | bytes):
                raise ValueError("'contains_any' requires a sequence")

            return (
                _field_reference(requirements.lhs) + f" CONTAINSANY ${parameter}",
                {parameter: values},
            )

        case "contains":
            parameter: str = f"_f{index}"
            return (
                _field_reference(requirements.lhs) + f" CONTAINS ${parameter}",
                {parameter: surreal_value(requirements.rhs)},
            )

        case "text_match":
            # `string(...)` is not a valid function path, values have to be cast instead
            parameter: str = f"_f{index}"
            field: str = _field_reference(requirements.lhs)
            return (
                f"string::contains(<string>{field}, <string>${parameter})",
                {parameter: str(requirements.rhs)},
            )

        case _:
            raise NotImplementedError(
                f"Unsupported SurrealDB requirement operator: {requirements.operator}"
            )


def _field_reference(
    path: object,
    /,
) -> str:
    # stored documents use attribute aliases, filters have to follow them
    return ".".join(attribute_path_segments(path))
