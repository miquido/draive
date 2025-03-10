from collections.abc import Mapping, Sequence

from haiway import AttributePath

__all__ = [
    "Meta",
    "MetaPath",
    "MetaValue",
]

type MetaValue = Mapping[str, MetaValue] | Sequence[MetaValue] | str | float | int | bool | None

type Meta = Mapping[str, MetaValue]

MetaPath: Meta = AttributePath(Meta, attribute=Meta)  # pyright: ignore[reportArgumentType, reportAssignmentType, reportCallIssue]
