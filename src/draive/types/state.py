import types
import typing
from collections.abc import Callable
from copy import copy
from dataclasses import (
    _MISSING_TYPE,  # pyright: ignore[reportPrivateUsage]
    MISSING,
    asdict,
    dataclass,
)
from dataclasses import (
    Field as DataclassField,
)
from dataclasses import (
    field as dataclass_field,
)
from dataclasses import (
    fields as dataclass_fields,
)
from typing import Any, Self, TypeVar, cast, dataclass_transform, final, get_args, get_origin

from draive.types.parameters import (
    ParametersSpecification,
    extract_specification,
)

__all__ = [
    "State",
    "Field",
]

_FieldType_T = TypeVar("_FieldType_T")


def Field(
    *,
    alias: str | None = None,
    default: _FieldType_T | _MISSING_TYPE = MISSING,
    validator: Callable[[_FieldType_T], None] | None = None,
) -> _FieldType_T:  # it is actually a dataclass.Field, but type checker has to be fooled
    metadata: dict[str, Any] = {}
    if alias:
        metadata["alias"] = alias
    if validator:
        metadata["validator"] = validator
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
    ) -> type[Any]:
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
    def specification(cls) -> ParametersSpecification:
        if not hasattr(cls, "_specification"):
            # TODO: allow aliases in specification?
            cls._specification: ParametersSpecification = extract_specification(cls.__init__)

        return cls._specification

    @classmethod
    def _fields_dict(cls) -> dict[str, DataclassField[Any]]:
        if not hasattr(cls, "_fields_dictionary"):
            cls._fields_dictionary: dict[str, DataclassField[Any]] = {}
            cls._fields_aliases: list[tuple[str, str]] = []
            for field in dataclass_fields(cls):
                cls._fields_dictionary[field.name] = field
                if (alias := field.metadata.get("alias")) and alias != field.name:
                    assert alias not in cls._fields_dictionary, "Field name duplicated by alias"  # nosec: B101
                    cls._fields_dictionary[alias] = field
                    cls._fields_aliases.append((field.name, alias))

        return cls._fields_dictionary

    @classmethod
    def _aliases(cls) -> list[tuple[str, str]]:
        if not hasattr(cls, "_fields_aliases"):
            cls._fields_dict()  # initialize when needed
        return cls._fields_aliases

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        strict: bool = False,
    ) -> Self:
        try:
            return cls(
                **cls.validated(
                    values=value,
                    strict=strict,
                )
            )

        except Exception as exc:
            raise ValueError(f"Failed to decode {cls.__name__} from dict:\n{value}") from exc

    @classmethod
    def validated(
        cls,
        values: dict[str, Any],
        strict: bool = False,
    ) -> dict[str, Any]:
        fields: dict[str, DataclassField[Any]] = cls._fields_dict()

        for key, value in copy(values).items():
            if field := fields.get(key):
                try:
                    if validated := _validated(
                        annotation=field.type,
                        value=value,
                    ):
                        if validator := field.metadata.get("validator"):
                            validator(validated)
                        del values[key]  # remove previous value
                        values[field.name] = validated
                    elif key != field.name:  # handle aliases
                        if validator := field.metadata.get("validator"):
                            validator(value)
                        del values[key]  # remove previous value
                        values[field.name] = value
                    else:
                        if validator := field.metadata.get("validator"):
                            validator(value)
                        continue  # keep it as is
                except TypeError as exc:
                    raise ValueError("Invalid value", key, value) from exc
            elif strict:
                raise ValueError("Unexpected value", key, value)
            else:
                del values[key]  # remove unwanted values
        # missing and default values are handled by init
        return values

    def as_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = asdict(self)
        for field, alias in self.__class__._aliases():
            values[alias] = values[field]
            del values[field]  # remove previous value

        return values

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
        return self.__class__(
            **self.__class__.validated(
                values={**vars(self), **kwargs},
                strict=True,
            ),
        )


def _validated(  # noqa: PLR0911, C901, PLR0912
    annotation: Any,
    value: Any,
) -> Any | None:
    # TODO: validate function/callable values
    match get_origin(annotation) or annotation:
        case typing.Annotated:
            match get_args(annotation):
                case [annotated, *_]:
                    return _validated(
                        annotation=annotated,
                        value=value,
                    )
                case annotated:
                    raise TypeError("Unsupported annotated type", annotated)
        case typing.Literal:
            if value in get_args(annotation):
                return None
            else:
                raise TypeError("Invalid value", annotation, value)
        case types.UnionType | typing.Union:
            for alternative in get_args(annotation):
                try:
                    return _validated(
                        annotation=alternative,
                        value=value,
                    )
                except TypeError:
                    continue  # check next alternative

            raise TypeError("Invalid value", annotation, value)

        case expected_type:
            if isinstance(value, expected_type):
                return None  # keep it as is
            elif isinstance(value, dict) and issubclass(expected_type, State):
                return expected_type.from_dict(value=cast(dict[str, Any], value))
            elif isinstance(value, float) and expected_type == int:
                return int(value)  # auto convert float to int
            elif isinstance(value, int) and expected_type == float:
                return float(value)  # auto convert int to float
            elif callable(expected_type):  # check functions
                return None  # TODO: add function signature validation
            else:
                raise TypeError("Invalid value", expected_type, value)  # pyright: ignore[reportUnknownArgumentType]
