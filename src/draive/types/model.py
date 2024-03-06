import json
from dataclasses import asdict
from typing import Self

from draive.types.state import State

__all__ = [
    "Model",
]


class Model(State):
    @classmethod
    def specification_json(cls) -> str:
        if not hasattr(cls, "_specification_json"):
            cls._specification_json: str = json.dumps(
                cls.specification(),
                indent=2,
            )

        return cls._specification_json

    @classmethod
    def from_json(
        cls,
        value: str | bytes,
        strict: bool = False,
    ) -> Self:
        try:
            return cls.from_dict(
                value=json.loads(value),
                strict=strict,
            )

        except Exception as exc:
            raise ValueError(f"Failed to decode {cls.__name__} from json:\n{value}") from exc

    def as_json(self) -> str:
        try:
            return json.dumps(self.as_dict())
        except Exception as exc:
            raise ValueError(
                f"Failed to encode {self.__class__.__name__} to json:\n{self}"
            ) from exc

    def __str__(self) -> str:
        return json.dumps(asdict(self), indent=2)
