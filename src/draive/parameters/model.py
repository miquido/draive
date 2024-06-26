import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Self
from uuid import UUID

from draive.parameters.data import ParametrizedData
from draive.parameters.schema import json_schema, simplified_schema
from draive.utils import Missing, cache, not_missing

__all__ = [
    "DataModel",
]


class ModelJSONEncoder(json.JSONEncoder):
    def default(self, o: object) -> Any:
        if isinstance(o, Missing):
            return None
        elif isinstance(o, datetime):
            return o.isoformat()
        elif isinstance(o, UUID):
            return o.hex
        else:
            return json.JSONEncoder.default(self, o)


class DataModel(ParametrizedData):
    @classmethod
    @cache(limit=2)
    def json_schema(
        cls,
        indent: int | None = None,
    ) -> str:
        if not_missing(cls.__PARAMETERS_SPECIFICATION__):
            return json_schema(
                cls.__PARAMETERS_SPECIFICATION__,
                indent=indent,
            )

        else:
            raise TypeError(f"{cls.__qualname__} can't be represented using json schema")

    @classmethod
    @cache(limit=2)
    def simplified_schema(
        cls,
        indent: int | None = None,
    ) -> str:
        if not_missing(cls.__PARAMETERS_SPECIFICATION__):
            return simplified_schema(
                cls.__PARAMETERS_SPECIFICATION__,
                indent=indent,
            )

        else:
            raise TypeError(f"{cls.__qualname__} can't be represented using simplified schema")

    @classmethod
    def from_json(
        cls,
        value: str | bytes,
        /,
        decoder: type[json.JSONDecoder] = json.JSONDecoder,
    ) -> Self:
        try:
            return cls(
                **json.loads(
                    value,
                    cls=decoder,
                )
            )

        except Exception as exc:
            raise ValueError(f"Failed to decode {cls.__name__} from json:\n{value}") from exc

    def as_json(
        self,
        aliased: bool = True,
        indent: int | None = None,
        encoder_class: type[json.JSONEncoder] = ModelJSONEncoder,
    ) -> str:
        try:
            return json.dumps(
                self.as_dict(aliased=aliased),
                indent=indent,
                cls=encoder_class,
            )

        except Exception as exc:
            raise ValueError(
                f"Failed to encode {self.__class__.__name__} to json:\n{asdict(self)}"
            ) from exc
