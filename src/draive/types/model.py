import json
from dataclasses import asdict
from typing import Self

from draive.types.specification import ParametrizedModel

__all__ = [
    "Model",
]


class Model(ParametrizedModel):
    @classmethod  # avoid making json each time
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
            return cls.validated(**json.loads(value))

        except Exception as exc:
            raise ValueError(f"Failed to decode {cls.__name__} from json:\n{value}") from exc

    def as_json(
        self,
        indent: int | None = None,
    ) -> str:
        try:
            return json.dumps(
                self.__class__.aliased_parameters(asdict(self)),
                indent=indent,
            )

        except Exception as exc:
            raise ValueError(
                f"Failed to encode {self.__class__.__name__} to json:\n{asdict(self)}"
            ) from exc

    def __str__(self) -> str:
        return self.as_json(indent=2)
