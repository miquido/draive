import re
from collections.abc import Sequence
from typing import Any, Final

from haiway import AttributeAnnotation

__all__ = ("attribute_path_segments",)

_PATH_SEPARATOR_PATTERN: Final[re.Pattern[str]] = re.compile(r"[.\[\]]+")


def attribute_path_segments(
    path: Any,
    /,
) -> Sequence[str]:
    """Resolve an attribute path into its serialized field names.

    Stored representations of ``State`` values use attribute aliases where those
    were declared, while attribute paths are written with python attribute names.
    Resolving a path through the model definition keeps queries aligned with the
    stored data instead of silently matching nothing.

    Parameters
    ----------
    path : Any
        Attribute path to resolve. Plain strings are only split into segments,
        they are assumed to be already expressed in the stored representation.

    Returns
    -------
    Sequence[str]
        Path segments using declared aliases where available. Collection indices
        and segments not declared by the model are preserved as provided.
    """
    segments: Sequence[str] = tuple(
        segment for segment in _PATH_SEPARATOR_PATTERN.split(str(path).strip(".")) if segment
    )
    root: Any = getattr(path, "__root__", None)
    if root is None:
        return segments  # plain string paths are used as provided

    resolved: list[str] = []
    current: Any = root
    for segment in segments:
        annotation: AttributeAnnotation | None = _field_annotation(current, segment)
        if annotation is None:
            # collection index or an attribute not declared by the model
            resolved.append(segment)
            current = None
            continue

        resolved.append(annotation.alias or segment)
        current = annotation.base

    return tuple(resolved)


def _field_annotation(
    model: Any,
    name: str,
    /,
) -> AttributeAnnotation | None:
    fields: Any = getattr(model, "__FIELDS__", None)
    if fields is None:
        return None

    return next(
        (field.annotation for field in fields if field.name == name),
        None,
    )
