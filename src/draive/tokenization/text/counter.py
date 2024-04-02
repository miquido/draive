from typing import Protocol, runtime_checkable

__all__ = [
    "TextTokenCounter",
]


@runtime_checkable
class TextTokenCounter(Protocol):
    def __call__(
        self,
        text: str,
    ) -> int:
        ...
