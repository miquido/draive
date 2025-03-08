from draive.commons import Meta
from draive.parameters import DataModel

__all__ = [
    "TextContent",
]


class TextContent(DataModel):
    text: str
    meta: Meta | None = None

    def __bool__(self) -> bool:
        return len(self.text) > 0
