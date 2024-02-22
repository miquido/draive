from typing import Protocol

__all__ = [
    "TextTokenCounter",
]


class TextTokenCounter(Protocol):
    def __call__(
        self,
        text: str,
    ) -> int:
        ...
