from typing import Self

from draive.commons import META_EMPTY, Meta
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
        meta: Meta | None = None,
    ) -> Self:
        return cls(
            category=category,
            content=content,
            meta=meta if meta is not None else META_EMPTY,
        )

    category: str
    content: TextContent | MediaContent | DataModel | None
    meta: Meta
