from typing import (
    Any,
    Final,
    final,
)

__all__ = [
    "MissingParameter",
    "MISSING_PARAMETER",
]


@final
class MissingParameter:
    def __bool__(self) -> bool:
        return False

    def __str__(self) -> str:
        return "MISSING"

    def __repr__(self) -> str:
        return "MISSING"

    def __setattr__(
        self,
        __name: str,
        __value: Any,
    ) -> None:
        raise RuntimeError("MissingParameter can't be modified")

    def __delattr__(
        self,
        __name: str,
    ) -> None:
        raise RuntimeError("MissingParameter can't be modified")


MISSING_PARAMETER: Final[MissingParameter] = MissingParameter()
