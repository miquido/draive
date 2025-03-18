from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from haiway import AttributePath

__all__ = [
    "META_EMPTY",
    "Meta",
    "MetaPath",
    "MetaValue",
]

type MetaValue = Mapping[str, MetaValue] | Sequence[MetaValue] | str | float | int | bool | None

type Meta = Mapping[str, MetaValue]

MetaPath: Meta = AttributePath(Meta, attribute=Meta)  # pyright: ignore[reportArgumentType, reportAssignmentType, reportCallIssue]

# using mapping proxy to make sure it won't be mutated as there is no frozendict
META_EMPTY: Final[Meta] = MappingProxyType({})
