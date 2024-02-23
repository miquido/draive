from dataclasses import Field, asdict, dataclass, field
from typing import Any, Self, dataclass_transform, final

__all__ = [
    "State",
]


@dataclass_transform(
    kw_only_default=True,
    frozen_default=True,
    field_specifiers=(Field, field),  # pyright: ignore[reportUnknownArgumentType]
)
class StateMeta(type):
    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        classdict: dict[str, Any],
        **kwargs: Any,
    ) -> type[Any]:
        # TODO: ensure properties belong to supported types:
        # bool, int, float, str, list[bool|int|float|str], dict[str, bool|int|float|str]
        # or other instances of State, additionally we need to support functions in state
        return final(  # pyright: ignore[reportUnknownVariableType]
            dataclass(  # pyright: ignore[reportGeneralTypeIssues, reportUnknownArgumentType, reportCallIssue]
                type.__new__(
                    cls,
                    name,
                    bases,
                    classdict,
                    **kwargs,
                ),
                frozen=True,
                kw_only=True,
            ),
        )


class State(metaclass=StateMeta):
    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
    ) -> Self:
        try:
            # TODO: add data validation step here
            # TODO: ensure nested objects conversion
            return cls(**value)

        except Exception as exc:
            raise ValueError(f"Failed to decode {cls.__name__} from dict:\n{value}") from exc

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return str(asdict(self))

    def __repr__(self) -> str:
        return self.__str__()

    def metric_summary(self) -> str:
        return self.__str__()

    # TODO: find a way to generate signature similar to dataclass __init__
    def updated(
        self,
        **kwargs: Any,
    ) -> Self:
        return self.__class__(**{**vars(self), **kwargs})
