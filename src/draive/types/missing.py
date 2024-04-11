from typing import Any, Final, TypeGuard, TypeVar, final, overload

__all__ = [
    "MISSING",
    "MissingValue",
    "when_missing",
    "is_missing",
    "not_missing",
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

_Value_T = TypeVar("_Value_T")
_Default_T = TypeVar("_Default_T")
_Cast_T = TypeVar("_Cast_T")


def is_missing(value: object | MissingValue) -> TypeGuard[MissingValue]:
    return value is MISSING


def not_missing(value: _Value_T | MissingValue) -> TypeGuard[_Value_T]:
    return value is not MISSING


@overload
def when_missing(
    value: _Value_T | MissingValue,
    /,
    default: _Default_T,
) -> _Value_T | _Default_T: ...


@overload
def when_missing(
    value: Any | MissingValue,
    /,
    default: Any,
    cast: type[_Cast_T],
) -> _Cast_T: ...


def when_missing(
    value: _Value_T | MissingValue,
    /,
    default: _Default_T,
    cast: type[_Cast_T] | None = None,
) -> _Value_T | _Default_T | _Cast_T:
    if not_missing(value):
        return value
    else:
        return default
