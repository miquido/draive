from collections.abc import Mapping

from draive.parameters import DataModel

__all__ = [
    "TextContent",
]


class TextContent(DataModel):
    text: str
    meta: Mapping[str, str | float | int | bool | None] | None = None

    def __bool__(self) -> bool:
        return len(self.text) > 0
