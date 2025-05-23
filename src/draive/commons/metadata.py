from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from haiway import AttributePath

__all__ = (
    "META_EMPTY",
    "Meta",
    "MetaPath",
    "MetaTags",
    "MetaValue",
    "check_meta_tags",
    "meta_tags",
    "with_meta_tags",
)

type MetaValue = Mapping[str, MetaValue] | Sequence[MetaValue] | str | float | int | bool | None

type Meta = Mapping[str, MetaValue]
type MetaTags = Sequence[str]

MetaPath: Meta = AttributePath(Meta, attribute=Meta)  # pyright: ignore[reportArgumentType, reportAssignmentType, reportCallIssue]

# using mapping proxy to make sure it won't be mutated as there is no frozendict
META_EMPTY: Final[Meta] = MappingProxyType({})


def meta_tags(
    meta: Meta,
    /,
) -> MetaTags:
    match meta.get("tags"):
        case [*tags]:
            return tuple(tag for tag in tags if isinstance(tag, str))

        case _:
            return ()


def with_meta_tags(
    meta: Meta,
    /,
    tags: MetaTags,
) -> Meta:
    match meta.get("tags"):
        case [*current_tags]:
            return {
                **meta,
                "tags": (*current_tags, *(tag for tag in tags if tag not in current_tags)),
            }

        case _:
            return {**meta, "tags": tags}


def check_meta_tags(
    meta: Meta,
    /,
    tags: MetaTags,
) -> bool:
    match meta.get("tags"):
        case [*meta_tags]:
            return all(tag in meta_tags for tag in tags)

        case _:
            return False
