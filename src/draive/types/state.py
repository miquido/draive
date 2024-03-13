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

__all__ = [
    "State",
    "Field",
    "StateField",
]

_FieldType_T = TypeVar("_FieldType_T")


def Field(
    *,
    alias: str | None = None,
    description: str | None = None,
    default: _FieldType_T | _MISSING_TYPE = MISSING,
    validator: Callable[[_FieldType_T], None] | None = None,
) -> _FieldType_T:  # it is actually a dataclass.Field, but type checker has to be fooled
    metadata: dict[str, Any] = {}
    if alias:
        metadata["alias"] = alias
    if description:
        metadata["description"] = description
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


@final
class StateField:
    def __init__(  # noqa: PLR0913
        self,
        name: str,
        alias: str | None,
        description: str | None,
        annotation: Any,
        required: bool,
        validator: Callable[[Any], None] | None,
    ) -> None:
        self.name: str = name
        self.alias: str | None = alias
        self.description: str | None = description
        self.annotation: Any = annotation
        self.required: bool = required
        self.validated: Callable[[Any], Any] = _prepare_validator(
            annotation=annotation,
            additional=validator,
        )


class State(metaclass=StateMeta):
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
        fields: dict[str, StateField] = cls._fields()
        aliases: dict[str, StateField] = cls._aliases()

        for key, value in copy(values).items():
            if field := fields.get(key):
                try:
                    values[field.name] = field.validated(value)
                except TypeError as exc:
                    raise ValueError("Invalid value", key, value) from exc
            elif field := aliases.get(key):
                try:
                    del values[key]  # remove previous value
                    values[field.name] = field.validated(value)
                except TypeError as exc:
                    raise ValueError("Invalid value", key, value) from exc
            elif strict:
                raise ValueError("Unexpected value", key, value)
            else:
                del values[key]  # remove unwanted values
        # missing and default values are handled by init
        return values

    @classmethod
    def _fields(cls) -> dict[str, StateField]:
        if not hasattr(cls, "__fields"):
            cls.__fields: dict[str, StateField] = {}
            cls.__aliases: dict[str, StateField] = {}
            for field in dataclass_fields(cls):
                alias: str | None = field.metadata.get("alias")
                cls.__fields[field.name] = StateField(
                    name=field.name,
                    alias=alias,
                    description=field.metadata.get("description"),
                    annotation=field.type,
                    required=field.default is MISSING and field.default_factory is MISSING,
                    validator=field.metadata.get("validator"),
                )
                if (alias := alias) and alias != field.name:
                    assert alias not in cls.__fields, "Field name duplicated by alias"  # nosec: B101
                    cls.__aliases[alias] = cls.__fields[field.name]

        return cls.__fields

    @classmethod
    def _aliases(cls) -> dict[str, StateField]:
        if not hasattr(cls, "__aliases"):
            cls._fields()  # initialize when needed
        return cls.__aliases

    def as_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = asdict(self)
        for field in self.__class__._aliases().values():
            assert field.alias  # nosec: B101
            values[field.alias] = values[field.name]
            del values[field.name]  # remove previous value

        return values

    def __str__(self) -> str:
        return str(asdict(self))

    def __repr__(self) -> str:
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


def _prepare_validator(  # noqa: C901
    annotation: Any,
    additional: Callable[[Any], None] | None,
) -> Callable[[Any], Any]:
    match get_origin(annotation) or annotation:
        case typing.Annotated:
            match get_args(annotation):
                case [annotated, *_]:
                    return _prepare_validator(
                        annotation=annotated,
                        additional=additional,
                    )
                case annotated:
                    raise TypeError("Unsupported annotated type", annotated)

        case typing.Literal:

            def validated(value: Any) -> Any:
                if value in get_args(annotation):
                    if validate := additional:
                        validate(value)
                    return value
                else:
                    raise TypeError("Invalid value", annotation, value)

            return validated

        case types.UnionType | typing.Union:
            validators: list[Callable[[Any], Any]] = [
                _prepare_validator(
                    annotation=alternative,
                    additional=additional,
                )
                for alternative in get_args(annotation)
            ]

            def validated(value: Any) -> Any:
                for validator in validators:
                    try:
                        return validator(value)
                    except TypeError:
                        continue  # check next alternative

                raise TypeError("Invalid value", annotation, value)

            return validated

        case state_type if issubclass(state_type, State):

            def validated(value: Any) -> Any:
                state: State
                if isinstance(value, dict):
                    state = state_type.from_dict(value=cast(dict[str, Any], value))
                elif isinstance(value, state_type):
                    state = value
                else:
                    raise TypeError("Invalid value", annotation, value)
                if validate := additional:
                    validate(state)
                return state

            return validated

        case typed_dict_type if typing.is_typeddict(typed_dict_type):

            def validated(value: Any) -> Any:
                typed_dict: dict[Any, Any]
                if isinstance(value, dict):
                    typed_dict = typed_dict_type(**value)
                else:
                    raise TypeError("Invalid value", annotation, value)
                if validate := additional:
                    validate(typed_dict)
                return typed_dict

            return validated

        case other_type:

            def validated(value: Any) -> Any:
                if isinstance(value, other_type):
                    if validate := additional:
                        validate(value)
                    return value
                elif isinstance(value, float) and other_type == int:
                    # auto convert float to int - json does not distinguish those
                    converted_int: int = int(value)
                    if validate := additional:
                        validate(converted_int)
                    return converted_int
                elif isinstance(value, int) and other_type == float:
                    # auto convert int to float - json does not distinguish those
                    converted_float: float = float(value)
                    if validate := additional:
                        validate(converted_float)
                    return converted_float
                # TODO: validate function/callable values
                elif callable(value):
                    if validate := additional:
                        validate(value)
                    return value
                else:
                    raise TypeError("Invalid value", annotation, value)

            return validated
