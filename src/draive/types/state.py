from collections.abc import Callable
from dataclasses import (
    _MISSING_TYPE as DATACLASS_MISSING_TYPE,  # pyright: ignore[reportPrivateUsage]
)
from dataclasses import MISSING as DATACLASS_MISSING
from dataclasses import Field as DataclassField
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields as dataclass_fields
from typing import Any, Self, TypeVar, cast, dataclass_transform, final

from draive.types.missing import MISSING
from draive.types.parameters import ParameterDefinition

__all__ = [
    "State",
    "Field",
]

_FieldType_T = TypeVar("_FieldType_T")


def Field(
    *,
    alias: str | None = None,
    description: str | None = None,
    default: Callable[[], _FieldType_T] | _FieldType_T | DATACLASS_MISSING_TYPE = DATACLASS_MISSING,
    validator: Callable[[_FieldType_T], None] | None = None,
) -> _FieldType_T:  # it is actually a dataclass.Field, but type checker has to be fooled
    metadata: dict[str, Any] = {}
    if alias:
        metadata["alias"] = alias
    if description:
        metadata["description"] = description
    if validator:
        metadata["validator"] = validator
    if callable(default):
        return dataclass_field(
            default_factory=default,
            metadata=metadata,
        )
    else:
        return cast(
            _FieldType_T,
            dataclass_field(
                default=default,
                metadata=metadata,
            ),
        )


@dataclass_transform(
    kw_only_default=True,
    frozen_default=True,
    field_specifiers=(DataclassField, dataclass_field),  # pyright: ignore[reportUnknownArgumentType]
)
class StateMeta(type):
    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        classdict: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        # TODO: ensure properties belong to supported types:
        # bool, int, float, str, list[bool|int|float|str], dict[str, bool|int|float|str]
        # or other instances of State, additionally we need to support functions in state
        # TODO: trigger validation from fields on init
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
    def _parameters(cls) -> list[ParameterDefinition]:
        if not hasattr(cls, "__parameters"):
            cls.__parameters: list[ParameterDefinition] = []
            for field in dataclass_fields(cls):
                default: Any
                if field.default is not DATACLASS_MISSING:
                    default = field.default
                elif field.default_factory is not DATACLASS_MISSING:
                    default = field.default_factory
                else:
                    default = MISSING

                cls.__parameters.append(
                    ParameterDefinition(
                        name=field.name,
                        alias=field.metadata.get("alias"),
                        description=field.metadata.get("description"),
                        default=default,
                        annotation=field.type,
                        validator=field.metadata.get("validator"),
                    )
                )

        return cls.__parameters

    @classmethod
    def _validated(
        cls,
        *,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        validated: dict[str, Any] = {}
        for parameter in cls._parameters():
            if parameter.name in values:
                validated[parameter.name] = parameter.validated_value(values.get(parameter.name))
            elif (alias := parameter.alias) and (alias in values):
                validated[parameter.name] = parameter.validated_value(values.get(alias))
            else:
                default_value = parameter.default_value()
                if default_value is MISSING:
                    raise ValueError("Missing required value", parameter.name)
                else:
                    validated[parameter.name] = default_value

        return validated

    @classmethod
    def from_dict(
        cls,
        values: dict[str, Any],
    ) -> Self:
        try:
            return cls(**cls._validated(values=values))
        except Exception as exc:
            raise ValueError(f"Failed to decode {cls.__name__} from dict:\n{values}") from exc

    def as_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = asdict(self)
        for parameter in self.__class__._parameters():
            if alias := parameter.alias:
                values[alias] = values[parameter.name]
                del values[parameter.name]  # remove aliased value

        return values

    # TODO: find a way to generate signature similar to dataclass __init__
    def updated(
        self,
        **kwargs: Any,
    ) -> Self:
        return self.__class__(
            **self.__class__._validated(values={**vars(self), **kwargs}),
        )
