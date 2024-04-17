from typing import Any, Final, TypeGuard, final, overload

__all__ = [
    "MISSING",
    "Missing",
    "when_missing",
    "is_missing",
    "not_missing",
]


@final
class Missing:
    def __bool__(self) -> bool:
        return False

    def __eq__(
        self,
        value: object,
    ) -> bool:
        return isinstance(value, Missing)

    def __hash__(self) -> int:
        return hash(Missing)

    def __str__(self) -> str:
        return "MISSING"

    def __repr__(self) -> str:
        return "MISSING"

    def __setattr__(
        self,
        __name: str,
        __value: Any,
    ) -> None:
        raise RuntimeError("Missing can't be modified")

    def __delattr__(
        self,
        __name: str,
    ) -> None:
        raise RuntimeError("Missing can't be modified")


MISSING: Final[Missing] = Missing()


def is_missing(value: object | Missing) -> TypeGuard[Missing]:
    return isinstance(value, Missing)


def not_missing[Value](value: Value | Missing) -> TypeGuard[Value]:
    return not isinstance(value, Missing)


@overload
def when_missing[Value, Default](
    value: Value | Missing,
    /,
    default: Default,
) -> Value | Default: ...


@overload
def when_missing[Casted](
    value: Any | Missing,
    /,
    default: Any,
    cast: type[Casted],
) -> Casted: ...


def when_missing[Value, Casted, Default](
    value: Value | Missing,
    /,
    default: Default,
    cast: type[Casted] | None = None,
) -> Value | Default | Casted:
    if not_missing(value):
        return value
    else:
        return default
