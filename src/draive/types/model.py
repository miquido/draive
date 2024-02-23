import json
from dataclasses import asdict
from typing import Self

from draive.helpers import extract_parameters_specification
from draive.types.state import State

__all__ = [
    "Model",
]


class Model(State):
    @classmethod
    def specification(cls) -> str:
        if not hasattr(cls, "_specification"):
            cls._specification: str = json.dumps(
                extract_parameters_specification(cls.__init__),
                indent=2,
            )

        return cls._specification

    @classmethod
    def from_json(
        cls,
        value: str | bytes,
    ) -> Self:
        try:
            return cls.from_dict(value=json.loads(value))

        except Exception as exc:
            raise ValueError(f"Failed to decode {cls.__name__} from json:\n{value}") from exc

    def as_json(self) -> str:
        try:
            return json.dumps(asdict(self))
        except Exception as exc:
            raise ValueError(
                f"Failed to encode {self.__class__.__name__} to json:\n{self}"
            ) from exc

    def __str__(self) -> str:
        return json.dumps(asdict(self), indent=2)
