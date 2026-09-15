from collections.abc import Sequence
from typing import Any

from haiway import AttributeAnnotation, AttributePath, State
from haiway.attributes.annotations import (
    MappingAttribute,
    NoneAttribute,
    ObjectAttribute,
    SequenceAttribute,
    TupleAttribute,
    TypedDictAttribute,
    UnionAttribute,
    ValidableAttribute,
)

__all__ = ("attribute_path_components", "attribute_path_segments")


def attribute_path_segments(
    path: AttributePath[Any, Any],
    /,
) -> Sequence[str]:
    """Resolve an attribute path into its serialized field names.

    Parameters
    ----------
    path : AttributePath
        Typed attribute path. Strings are not supported.

    Returns
    -------
    Sequence[str]
        Serialized field names, mapping keys and collection indices, with aliases
        resolved through nested models and collection elements.

    Raises
    ------
    AssertionError
        If the argument is not an AttributePath.
    """
    return tuple(str(component) for component in attribute_path_components(path))


def attribute_path_components(
    path: AttributePath[Any, Any],
    /,
) -> Sequence[str | int]:
    """Resolve serialized names while preserving sequence indices as integers.

    Parameters
    ----------
    path : AttributePath
        Typed attribute path to resolve.

    Returns
    -------
    Sequence[str | int]
        Aliased field names and mapping keys as strings, sequence indices as ints.

    Raises
    ------
    AssertionError
        If the argument is not an AttributePath.
    """
    assert isinstance(path, AttributePath), "Expected an AttributePath"  # nosec: B101
    resolved: list[str | int] = []
    annotation: AttributeAnnotation | None = (
        path.__root__.__SELF_ATTRIBUTE__ if issubclass(path.__root__, State) else None
    )
    for component in path.components:
        annotation = _required_annotation(annotation)
        if isinstance(annotation, ObjectAttribute | TypedDictAttribute):
            annotation = annotation.attributes.get(component)
            resolved.append(annotation.alias or component if annotation else component)

        elif isinstance(annotation, MappingAttribute):
            resolved.append(component[1:-1])
            annotation = annotation.values

        elif isinstance(annotation, SequenceAttribute):
            resolved.append(int(component[1:-1]))
            annotation = annotation.values

        elif isinstance(annotation, TupleAttribute):
            resolved.append(int(component[1:-1]))
            annotation = annotation.values[int(component[1:-1])]

        else:
            resolved.append(component)
            annotation = None

    return tuple(resolved)


def _required_annotation(
    annotation: AttributeAnnotation | None,
    /,
) -> AttributeAnnotation | None:
    if isinstance(annotation, ValidableAttribute):
        annotation = annotation.attribute

    if isinstance(annotation, UnionAttribute):
        alternatives: Sequence[AttributeAnnotation] = tuple(
            alternative
            for alternative in annotation.alternatives
            if not isinstance(alternative, NoneAttribute)
        )
        if len(alternatives) == 1:
            return _required_annotation(alternatives[0])

    if (
        annotation is not None
        and isinstance(annotation.base, type)
        and issubclass(annotation.base, State)
    ):
        return annotation.base.__SELF_ATTRIBUTE__

    return annotation
