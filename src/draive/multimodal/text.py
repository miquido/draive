from haiway import Default

from draive.commons import META_EMPTY, Meta
from draive.parameters import DataModel

__all__ = ("TextContent",)


class TextContent(DataModel):
    text: str
    meta: Meta = Default(META_EMPTY)

    def __bool__(self) -> bool:
        return len(self.text) > 0
