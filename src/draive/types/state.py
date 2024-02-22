import json
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
    def from_json(
        cls,
        value: str | bytes,
    ) -> Self:
        try:
            return cls.from_dict(values=json.loads(value))

        except Exception as exc:
            raise ValueError(f"Failed to decode {cls.__name__} from json:\n{value}") from exc

    @classmethod
    def from_dict(
        cls,
        values: dict[str, object],
    ) -> Self:
        try:
            # TODO: add data validation step here
            return cls(**values)

        except Exception as exc:
            raise ValueError(f"Failed to decode {cls.__name__} from dict:\n{values}") from exc

    def as_json(self) -> str:
        try:
            return json.dumps(asdict(self))
        except Exception as exc:
            raise ValueError(
                f"Failed to encode {self.__class__.__name__} to json:\n{self}"
            ) from exc

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def __str__(self) -> str:
        return self.as_json()

    def __repr__(self) -> str:
        try:
            return f"{self.__class__.__name__} {json.dumps(asdict(self), indent=2)}"

        except Exception:
            return f"{self.__class__.__name__} {asdict(self)}"

    def metric_summary(self) -> str:
        return self.__repr__()

    # TODO: find a way to generate signature similar to dataclass __init__
    def updated(
        self,
        **kwargs: Any,
    ) -> Self:
        return self.__class__(**{**vars(self), **kwargs})
