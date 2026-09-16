from collections.abc import Mapping, Sequence
from typing import Any

from haiway import AttributePath, AttributeRequirement, State

from draive.surreal.types import SurrealValue
from draive.surreal.utils import surreal_value
from draive.utils.attributes import attribute_path_components

__all__ = ("prepare_filter",)


def prepare_filter[Model: State](
    requirements: AttributeRequirement[Model] | None,
    /,
    *,
    scoped: bool = False,
) -> tuple[str, Mapping[str, SurrealValue]]:
    """Translate typed requirements into a parameterized SurrealDB filter.

    Parameters
    ----------
    requirements : AttributeRequirement | None
        Conditions whose paths refer to the original model.
    scoped : bool, default=False
        Prefix each resolved path with the vector index's storage field, ``content``.

    Returns
    -------
    tuple[str, Mapping[str, SurrealValue]]
        Filter expression and bound values, or an empty expression without conditions.

    Raises
    ------
    AssertionError
        If a condition uses a string instead of an AttributePath.
    ValueError
        If a collection operator receives an invalid operand.
    NotImplementedError
        If an operator is unsupported.
    """
    if requirements is None:
        return ("", {})

    return _convert(requirements, index=0, scoped=scoped)


def _convert[Model: State](  # noqa: C901, PLR0911
    requirements: AttributeRequirement[Model],
    /,
    *,
    index: int,
    scoped: bool,
) -> tuple[str, Mapping[str, SurrealValue]]:
    match requirements.operator:
        case "and":
            left_clause, left_values = _convert(requirements.lhs, index=index, scoped=scoped)
            right_clause, right_values = _convert(
                requirements.rhs,
                index=index + len(left_values),
                scoped=scoped,
            )
            return (
                f"({left_clause}) AND ({right_clause})",
                {**left_values, **right_values},
            )

        case "or":
            left_clause, left_values = _convert(requirements.lhs, index=index, scoped=scoped)
            right_clause, right_values = _convert(
                requirements.rhs,
                index=index + len(left_values),
                scoped=scoped,
            )
            return (
                f"({left_clause}) OR ({right_clause})",
                {**left_values, **right_values},
            )

        case "equal":
            parameter: str = f"_f{index}"
            return (
                _field_reference(requirements.lhs, scoped=scoped) + f" = ${parameter}",
                {parameter: surreal_value(requirements.rhs)},
            )

        case "not_equal":
            parameter: str = f"_f{index}"
            return (
                _field_reference(requirements.lhs, scoped=scoped) + f" != ${parameter}",
                {parameter: surreal_value(requirements.rhs)},
            )

        case "contained_in":
            # 'contained_in' is the only operator built with its operands swapped
            parameter: str = f"_f{index}"
            values: SurrealValue = surreal_value(requirements.lhs)
            if not isinstance(values, Sequence) or isinstance(values, str | bytes):
                raise ValueError("'contained_in' requires a sequence")

            return (
                _field_reference(requirements.rhs, scoped=scoped) + f" INSIDE ${parameter}",
                {parameter: values},
            )

        case "contains_any":
            parameter: str = f"_f{index}"
            values: SurrealValue = surreal_value(requirements.rhs)
            if not isinstance(values, Sequence) or isinstance(values, str | bytes):
                raise ValueError("'contains_any' requires a sequence")

            return (
                _field_reference(requirements.lhs, scoped=scoped) + f" CONTAINSANY ${parameter}",
                {parameter: values},
            )

        case "contains":
            parameter: str = f"_f{index}"
            return (
                _field_reference(requirements.lhs, scoped=scoped) + f" CONTAINS ${parameter}",
                {parameter: surreal_value(requirements.rhs)},
            )

        case "text_match":
            # `string(...)` is not a valid function path, values have to be cast instead
            parameter: str = f"_f{index}"
            field: str = _field_reference(requirements.lhs, scoped=scoped)
            return (
                f"string::contains(<string>{field}, <string>${parameter})",
                {parameter: str(requirements.rhs)},
            )

        case _:
            raise NotImplementedError(
                f"Unsupported SurrealDB requirement operator: {requirements.operator}"
            )


def _field_reference(
    path: AttributePath[Any, Any],
    /,
    *,
    scoped: bool,
) -> str:
    # stored documents use attribute aliases, filters have to follow them
    resolved: str = ""
    for component in attribute_path_components(path):
        if isinstance(component, int):
            resolved += f"[{component}]"
        else:
            resolved += f".{component}" if resolved else component
    return f"content.{resolved}" if scoped else resolved
