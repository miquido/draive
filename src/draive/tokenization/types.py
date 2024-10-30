from typing import Any, Protocol, runtime_checkable

__all__ = [
    "TextTokenizing",
]


@runtime_checkable
class TextTokenizing(Protocol):
    def __call__(
        self,
        text: str,
        **extra: Any,
    ) -> list[int]: ...
