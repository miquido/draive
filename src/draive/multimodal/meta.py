from draive.commons import Meta
from draive.multimodal.media import MediaContent
from draive.multimodal.text import TextContent
from draive.parameters import DataModel

__all__ = [
    "MetaContent",
]


class MetaContent(DataModel):
    category: str
    content: TextContent | MediaContent | DataModel
    meta: Meta

    def __bool__(self) -> bool:
        return bool(self.content)
