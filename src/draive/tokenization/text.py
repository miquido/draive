from typing import Protocol, runtime_checkable

__all__ = [
    "TextTokenizer",
]


@runtime_checkable
class TextTokenizer(Protocol):
    def __call__(
        self,
        text: str,
    ) -> list[int]:
        ...
