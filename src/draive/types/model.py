import json
from typing import Self

from draive.types.missing import MISSING
from draive.types.parameters import ParameterSpecification
from draive.types.state import State

__all__ = [
    "Model",
]


class Model(State):
    @classmethod
    def specification(cls) -> ParameterSpecification:
        if not hasattr(cls, "_specification"):
            cls._specification: ParameterSpecification = {
                "type": "object",
                "properties": {
                    parameter.alias or parameter.name: parameter.specification()
                    for parameter in cls._parameters()
                },
                "required": [
                    parameter.alias or parameter.name
                    for parameter in cls._parameters()
                    if parameter.default is MISSING
                ],
            }

        return cls._specification

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
    ) -> Self:
        try:
            return cls.from_dict(values=json.loads(value))

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
        return json.dumps(
            self.as_dict(),
            indent=2,
        )
