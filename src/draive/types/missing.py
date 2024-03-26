from typing import Any, Final, final

__all__ = [
    "MissingValue",
    "MISSING",
]


@final
class MissingValue:
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
        raise RuntimeError("MissingValue can't be modified")


MISSING: Final[MissingValue] = MissingValue()
