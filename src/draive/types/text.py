from draive.parameters import DataModel

__all__ = [
    "TextContent",
]


class TextContent(DataModel):
    text: str
    meta: dict[str, str | float | int | bool | None] | None = None

    def __bool__(self) -> bool:
        return bool(self.text)
