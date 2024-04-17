from typing import Any, final

from draive.helpers import freeze

__all__ = [
    "KeyPathComponent",
]


@final
class KeyPathComponent:
    def __init__(
        self,
        name: str,
        expected: type[Any],
    ) -> None:
        self.name: str = name
        self._expected: type[Any] = expected

        freeze(self)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.name}: {self._expected}"
