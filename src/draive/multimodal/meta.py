from typing import Self

from draive.commons import Meta, MetaValues
from draive.multimodal.media import MediaContent
from draive.multimodal.text import TextContent
from draive.parameters import DataModel

__all__ = ("MetaContent",)


class MetaContent(DataModel):
    @classmethod
    def of(
        cls,
        category: str,
        /,
        *,
        content: TextContent | MediaContent | DataModel | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            category=category,
            content=content,
            meta=Meta.of(meta),
        )

    category: str
    content: TextContent | MediaContent | DataModel | None
    meta: Meta
