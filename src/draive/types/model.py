import json
from typing import Self, get_origin

from draive.types.parameters import ParameterSpecification, parameter_specification
from draive.types.state import State, StateField

__all__ = [
    "Model",
]


class Model(State):
    @classmethod
    def specification(cls) -> ParameterSpecification:
        if not hasattr(cls, "_specification"):
            fields: dict[str, StateField] = cls._fields()

            cls._specification: ParameterSpecification = {
                "type": "object",
                "properties": {
                    field.alias or field.name: parameter_specification(
                        annotation=field.annotation,
                        origin=get_origin(field.annotation),
                        description=field.description,
                    )
                    for field in fields.values()
                },
                "required": [field.alias or field.name for field in fields.values()],
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
        return json.dumps(self.as_dict(), indent=2)
