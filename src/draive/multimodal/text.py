from typing import Self

from draive.commons import META_EMPTY, Meta, MetaValues
from draive.parameters import DataModel

__all__ = ("TextContent",)


class TextContent(DataModel):
    @classmethod
    def of(
        cls,
        text: str,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            text=text,
            meta=Meta.of(meta),
        )

    text: str
    meta: Meta = META_EMPTY

    def __bool__(self) -> bool:
        return len(self.text) > 0
