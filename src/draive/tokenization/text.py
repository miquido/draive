from typing import Any, Protocol, runtime_checkable

__all__ = [
    "TextTokenizer",
]


@runtime_checkable
class TextTokenizer(Protocol):
    def __call__(
        self,
        text: str,
        **extra: Any,
    ) -> list[int]: ...
