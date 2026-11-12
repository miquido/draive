from typing import Any, NoReturn, Self

__all__ = ("ImmutableRecord",)


class ImmutableRecord:
    """Base of slotted records which hold what they are given and refuse changes.

    Subclasses declare ``__slots__`` and bind their attributes in ``__init__`` through
    ``object.__setattr__``. Equality and hashing are by identity, ``copy`` and ``deepcopy``
    return the instance itself.
    """

    __slots__ = ()

    def __setattr__(
        self,
        name: str,
        value: Any,
    ) -> NoReturn:
        raise AttributeError(
            f"Can't modify immutable {self.__class__.__qualname__}"
            f" attribute - '{name}' cannot be modified"
        )

    def __delattr__(
        self,
        name: str,
    ) -> NoReturn:
        raise AttributeError(
            f"Can't modify immutable {self.__class__.__qualname__}"
            f" attribute - '{name}' cannot be deleted"
        )

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(
        self,
        memo: dict[int, Any] | None = None,
    ) -> Self:
        return self
