import builtins
import inspect
import types
import typing
from collections import abc as collections_abc
from collections.abc import Callable, Iterable, Iterator
from dataclasses import (
    _MISSING_TYPE as DATACLASS_MISSING_TYPE,  # pyright: ignore[reportPrivateUsage]
)
from dataclasses import MISSING as DATACLASS_MISSING
from dataclasses import Field as DataclassField
from dataclasses import asdict, dataclass, is_dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields as dataclass_fields
from typing import (
    Any,
    ForwardRef,
    Generic,
    ParamSpec,
    Protocol,
    Self,
    TypeVar,
    cast,
    dataclass_transform,
    final,
    get_args,
    get_origin,
)

import typing_extensions

from draive.types.missing import MISSING, MissingValue

__all__ = [
    "ParameterDefinition",
    "ParametersDefinition",
    "ParametrizedMeta",
    "Field",
    "ParametrizedState",
    "Function",
    "Argument",
    "ParametrizedFunction",
]

_ArgumentType_T = TypeVar("_ArgumentType_T")


@final
class ParameterDefinition:
    def __init__(  # noqa: PLR0913
        self,
        name: str,
        alias: str | None,
        description: str | None,
        annotation: Any,
        module: Any,
        default: _ArgumentType_T | MissingValue,
        validator: Callable[[_ArgumentType_T], None] | None,
    ) -> None:
        assert name != alias, "Alias can't be the same as name"  # nosec: B101
        self.name: str = name
        self.alias: str | None = alias
        self.description: str | None = description
        self.annotation: Any = annotation
        self.default: _ArgumentType_T | MissingValue = default
        self.validator: Callable[[_ArgumentType_T], _ArgumentType_T] = _parameter_validator(
            annotation,
            additional=validator,
            module=module,
        )

        def frozen(
            __name: str,
            __value: Any,
        ) -> None:
            raise RuntimeError("ParameterDefinition can't be modified")

        self.__setattr__ = frozen

    def default_value(self) -> Any | MissingValue:
        return self.default

    def validate_value(
        self,
        value: Any | MissingValue,
        /,
    ) -> Any:
        if value is MISSING:
            default: Any | MissingValue = self.default_value()
            if default is MISSING:
                raise ValueError("Missing required value of %s", self.name)
            else:
                return self.validator(default)  # type: ignore[reportArgumentType]
        else:
            return self.validator(value)  # type: ignore[reportArgumentType]


@final
class ParametersDefinition:
    def __init__(
        self,
        parameters: Iterator[ParameterDefinition] | Iterable[ParameterDefinition],
    ) -> None:
        self.parameters: list[ParameterDefinition] = []
        self._alias_map: dict[str, str] = {}
        self.aliased_required: list[str] = []
        names: set[str] = set()
        aliases: set[str] = set()
        for parameter in parameters:
            assert (  # nosec: B101
                parameter.name not in names
            ), f"Parameter names can't overlap with each other: {parameter.name}"
            assert (  # nosec: B101
                parameter.name not in aliases
            ), f"Parameter aliases can't overlap with names: {parameter.name}"
            names.add(parameter.name)
            if alias := parameter.alias:
                assert alias not in names, f"Parameter aliases can't overlap with names: {alias}"  # nosec: B101
                assert alias not in aliases, f"Duplicate parameter aliases are not allowed: {alias}"  # nosec: B101
                aliases.add(alias)
                self._alias_map[parameter.name] = alias

            self.parameters.append(parameter)
            if parameter.default is MISSING:
                self.aliased_required.append(parameter.alias or parameter.name)

        assert names.isdisjoint(aliases), "Aliases can't overlap regular names"  # nosec: B101

        def frozen(
            __name: str,
            __value: Any,
        ) -> None:
            raise RuntimeError("ParametersDefinition can't be modified")

        self.__setattr__ = frozen

    def validated(
        self,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        validated: dict[str, Any] = {}
        for parameter in self.parameters:
            if parameter.name in values:
                validated[parameter.name] = parameter.validate_value(values.get(parameter.name))
            elif (alias := parameter.alias) and (alias in values):
                validated[parameter.name] = parameter.validate_value(values.get(alias))
            else:
                validated[parameter.name] = parameter.validate_value(MISSING)
        return validated

    def aliased(
        self,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        return {self._alias_map.get(key, key): value for key, value in values.items()}


class ParametrizedMeta(type):
    def parameters(cls) -> ParametersDefinition:
        raise NotImplementedError("Parameters have to be provided by: %s", cls)

    def validated_parameters(
        cls,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        return cls.parameters().validated(values=values)

    def aliased_parameters(
        cls,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        return cls.parameters().aliased(values=values)


_FieldType_T = TypeVar("_FieldType_T")


def Field(
    *,
    alias: str | None = None,
    description: str | None = None,
    default: _FieldType_T | DATACLASS_MISSING_TYPE = DATACLASS_MISSING,
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


def _field_parameter(
    field: DataclassField[Any],
    /,
    module: Any,
) -> ParameterDefinition:
    default: Any
    if field.default is not DATACLASS_MISSING:
        default = field.default
    elif field.default_factory is not DATACLASS_MISSING:
        raise NotImplementedError("default_factory is not supported yet")
    else:
        default = MISSING

    return ParameterDefinition(
        name=field.name,
        alias=field.metadata.get("alias"),
        description=field.metadata.get("description"),
        default=default,
        annotation=field.type,
        module=module,
        validator=field.metadata.get("validator"),
    )


@dataclass_transform(
    kw_only_default=True,
    frozen_default=True,
    field_specifiers=(DataclassField, dataclass_field),  # pyright: ignore[reportUnknownArgumentType]
)
class ParametrizedStateMeta(ParametrizedMeta):
    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        classdict: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        return dataclass(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
            type.__new__(
                cls,
                name,
                bases,
                classdict,
                **kwargs,
            ),
            frozen=True,
            kw_only=True,
        )


class ParametrizedState(metaclass=ParametrizedStateMeta):
    @classmethod
    def parameters(cls) -> ParametersDefinition:
        if not hasattr(cls, "_parameters"):
            cls._parameters: ParametersDefinition = ParametersDefinition(
                parameters=(
                    _field_parameter(
                        field,
                        module=cls.__module__,
                    )
                    for field in dataclass_fields(cls)
                )
            )

        return cls._parameters

    @classmethod
    def validated(cls, **kwargs: Any) -> Self:
        return cls(**cls.validated_parameters(values=kwargs))

    @classmethod
    def from_dict(
        cls,
        values: dict[str, Any],
    ) -> Self:
        try:
            return cls.validated(**values)
        except Exception as exc:
            raise ValueError(f"Failed to decode {cls.__name__} from dict:\n{values}") from exc

    def as_dict(self) -> dict[str, Any]:
        return self.__class__.aliased_parameters(asdict(self))


def Argument(  # Ruff - noqa: B008
    *,
    alias: str | None = None,
    description: str | None = None,
    default: _ArgumentType_T | MissingValue = MISSING,
    validator: Callable[[_ArgumentType_T], None] | None = None,
) -> _ArgumentType_T:  # it is actually a FunctionParameter, but type checker has to be fooled
    return cast(
        _ArgumentType_T,
        FunctionArgument(
            alias=alias,
            description=description,
            default=default,
            validator=validator,
        ),
    )


@final
class FunctionArgument:
    def __init__(
        self,
        alias: str | None,
        description: str | None,
        default: _ArgumentType_T | MissingValue,
        validator: Callable[[_ArgumentType_T], None] | None,
    ) -> None:
        self.alias: str | None = alias
        self.description: str | None = description
        self.default: _ArgumentType_T | MissingValue = default
        self.validator: Callable[[_ArgumentType_T], None] | None = validator


FunctionArgs = ParamSpec(
    name="FunctionArgs",
    # bound= - ideally it should be bound to allowed types, not implemented in python yet
)
FunctionResult = TypeVar(
    name="FunctionResult",
)
FunctionResult_co = TypeVar(
    name="FunctionResult_co",
    covariant=True,
)


class Function(Protocol[FunctionArgs, FunctionResult_co]):
    @property
    def __name__(self) -> str:
        ...

    def __call__(
        self,
        *args: FunctionArgs.args,
        **kwargs: FunctionArgs.kwargs,
    ) -> FunctionResult_co:
        ...


class ParametrizedFunction(Generic[FunctionArgs, FunctionResult]):
    def __init__(
        self,
        function: Function[FunctionArgs, FunctionResult],
    ) -> None:
        assert not isinstance(  # nosec: B101
            function, ParametrizedFunction
        ), "Nested function wrapping is not allowed"
        # mimic function attributes if able
        try:
            self.__module__ = function.__module__
        except AttributeError:
            pass
        try:
            self.__name__ = function.__name__
        except AttributeError:
            pass
        try:
            self.__qualname__ = function.__qualname__
        except AttributeError:
            pass
        try:
            self.__doc__ = function.__doc__
        except AttributeError:
            pass
        try:
            self.__annotations__ = function.__annotations__
        except AttributeError:
            pass
        try:
            self.__dict__.update(function.__dict__)
        except AttributeError:
            pass

        self._call: Function[FunctionArgs, FunctionResult] = function
        self.parameters: ParametersDefinition = ParametersDefinition(
            parameters=(
                _function_argument(argument, module=self.__module__)
                for argument in inspect.signature(function).parameters.values()
            )
        )

    def validated_parameters(
        self,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        return self.parameters.validated(values=values)

    def __call__(
        self,
        *args: FunctionArgs.args,
        **kwargs: FunctionArgs.kwargs,
    ) -> FunctionResult:
        assert not args, "Positional unkeyed arguments are not supported"  # nosec: B101
        return self._call(*args, **self.validated_parameters(values=kwargs))


def _function_argument(
    argument: Any,
    /,
    module: Any,
) -> ParameterDefinition:
    if argument.annotation is inspect._empty:  # pyright: ignore[reportPrivateUsage]
        raise TypeError(
            "Untyped argument %s",
            argument.name,
        )

    if isinstance(argument.default, FunctionArgument):
        return ParameterDefinition(
            name=argument.name,
            alias=argument.default.alias,
            description=argument.default.description,
            default=argument.default.default,
            annotation=argument.annotation,
            module=module,
            validator=argument.default.validator,
        )
    else:  # use regular annotation
        return ParameterDefinition(
            name=argument.name,
            alias=None,
            description=None,
            default=MISSING
            if argument.default is inspect._empty  # pyright: ignore[reportPrivateUsage]
            else argument.default,
            annotation=argument.annotation,
            module=module,
            validator=None,
        )


def _parameter_validator(  # noqa: PLR0915, PLR0912, PLR0911, C901
    parameter_annotation: Any,
    /,
    additional: Callable[[Any], None] | None,
    module: Any,
) -> Callable[[Any], Any]:
    annotation: Any
    if isinstance(parameter_annotation, str):
        # TODO: FIXME: resolve in valid context
        annotation = ForwardRef(parameter_annotation, module=module)._evaluate(  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            globalns=None,
            localns=None,
            recursive_guard=frozenset(),
        )
    elif isinstance(parameter_annotation, ForwardRef):
        # TODO: FIXME: resolve in valid context
        annotation = parameter_annotation._evaluate(  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            globalns=None,
            localns=None,
            recursive_guard=frozenset(),
        )
    else:
        annotation = parameter_annotation
    match get_origin(annotation) or annotation:
        case types.NoneType:
            if validate := additional:

                def validated_none(value: Any) -> None:
                    if value is None:
                        validate(value)
                        return value
                    else:
                        raise TypeError("Invalid value", annotation, value)
            else:

                def validated_none(value: Any) -> None:
                    if value is None:
                        return value
                    else:
                        raise TypeError("Invalid value", annotation, value)

            return validated_none

        case builtins.str:
            if validate := additional:

                def validated_string(value: Any) -> str:
                    if isinstance(value, str):
                        validate(value)
                        return value
                    else:
                        raise TypeError("Invalid value", annotation, value)
            else:

                def validated_string(value: Any) -> str:
                    if isinstance(value, str):
                        return value
                    else:
                        raise TypeError("Invalid value", annotation, value)

            return validated_string

        case builtins.int:
            if validate := additional:

                def validated_int(value: Any) -> int:
                    if isinstance(value, int):
                        validate(value)
                        return value
                    elif isinstance(value, float) and value.is_integer():
                        # auto-convert from float
                        converted: int = int(value)
                        validate(converted)
                        return converted
                    else:
                        raise TypeError("Invalid value", annotation, value)
            else:

                def validated_int(value: Any) -> int:
                    if isinstance(value, int):
                        return value
                    elif isinstance(value, float) and value.is_integer():
                        # auto-convert from float
                        return int(value)
                    else:
                        raise TypeError("Invalid value", annotation, value)

            return validated_int

        case builtins.float:
            if validate := additional:

                def validated_float(value: Any) -> float:
                    if isinstance(value, float):
                        validate(value)
                        return value
                    elif isinstance(value, int):
                        # auto-convert from int
                        converted: int = int(value)
                        validate(converted)
                        return converted
                    else:
                        raise TypeError("Invalid value", annotation, value)
            else:

                def validated_float(value: Any) -> float:
                    if isinstance(value, float):
                        return value
                    elif isinstance(value, int):
                        # auto-convert from float
                        return float(value)
                    else:
                        raise TypeError("Invalid value", annotation, value)

            return validated_float

        case builtins.bool:
            if validate := additional:

                def validated_bool(value: Any) -> bool:
                    if isinstance(value, bool):
                        validate(value)
                        return value
                    else:
                        raise TypeError("Invalid value", annotation, value)
            else:

                def validated_bool(value: Any) -> bool:
                    if isinstance(value, bool):
                        return value
                    else:
                        raise TypeError("Invalid value", annotation, value)

            return validated_bool

        case types.UnionType | typing.Union:
            # TODO: try to use dict/match instead of loop over list
            validators: list[Callable[[Any], Any]] = [
                _parameter_validator(
                    alternative,
                    additional=additional,
                    module=module,
                )
                for alternative in get_args(annotation)
            ]
            if validate := additional:

                def validated_union(value: Any) -> Any:
                    for validator in validators:
                        try:
                            validated: Any = validator(value)
                            validate(validated)
                            return validated
                        except (ValueError, TypeError):
                            continue

                    raise TypeError("Invalid value", annotation, value)
            else:

                def validated_union(value: Any) -> Any:
                    for validator in validators:
                        try:
                            return validator(value)
                        except (ValueError, TypeError):
                            continue

                    raise TypeError("Invalid value", annotation, value)

            return validated_union

        case (
            builtins.list  # pyright: ignore[reportUnknownMemberType]
            | typing.List  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
        ):
            validate_element: Callable[[Any], Any]
            match get_args(annotation):
                case (element_annotation,):
                    validate_element = _parameter_validator(
                        element_annotation,
                        additional=None,
                        module=module,
                    )

                case ():  # pyright: ignore[reportUnnecessaryComparison] fallback to untyped list
                    validate_element = lambda value: value  # noqa: E731

                case other:  # pyright: ignore[reportUnnecessaryComparison]
                    raise TypeError("Unsupported list type annotation", other)

            if validate := additional:

                def validated_list(value: Any) -> list[Any]:
                    if isinstance(value, list):
                        values_list: builtins.list[Any] = [
                            validate_element(element)
                            for element in value  # pyright: ignore[reportUnknownVariableType]
                        ]
                        validate(values_list)
                        return values_list
                    else:
                        raise TypeError("Invalid value", annotation, value)
            else:

                def validated_list(value: Any) -> list[Any]:
                    if isinstance(value, list):
                        return [
                            validate_element(element)
                            for element in value  # pyright: ignore[reportUnknownVariableType]
                        ]
                    else:
                        raise TypeError("Invalid value", annotation, value)

            return validated_list

        case typing.Literal:
            options: tuple[Any, ...] = get_args(annotation)
            if validate := additional:

                def validated_literal(value: Any) -> Any:
                    if value in options:
                        validate(value)
                        return value
                    else:
                        raise TypeError("Invalid value", annotation, value)
            else:

                def validated_literal(value: Any) -> Any:
                    if value in options:
                        return value
                    else:
                        raise TypeError("Invalid value", annotation, value)

            return validated_literal

        case parametrized if issubclass(parametrized, ParametrizedState):
            if validate := additional:

                def validated(value: Any) -> Any:
                    if isinstance(value, parametrized):
                        validate(value)
                        return value
                    elif isinstance(value, dict):
                        validated: ParametrizedState = parametrized.validated(**value)
                        validate(validated)
                        return validated
                    else:
                        raise TypeError("Invalid value", annotation, value)
            else:

                def validated(value: Any) -> Any:
                    if isinstance(value, parametrized):
                        return value
                    elif isinstance(value, dict):
                        return parametrized.validated(**value)
                    else:
                        raise TypeError("Invalid value", annotation, value)

            return validated

        case typed_dict if typing.is_typeddict(typed_dict) or typing_extensions.is_typeddict(
            typed_dict
        ):
            # TODO: validate typed dict internals (i.e. nested typed dicts)
            if validate := additional:

                def validated_typed_dict(value: Any) -> dict[str, Any]:
                    if isinstance(value, dict):
                        validated: Any = typed_dict(**value)
                        validate(validated)
                        return validated
                    else:
                        raise TypeError("Invalid value", annotation, value)
            else:

                def validated_typed_dict(value: Any) -> dict[str, Any]:
                    if isinstance(value, dict):
                        return typed_dict(**value)
                    else:
                        raise TypeError("Invalid value", annotation, value)

            return validated_typed_dict

        case (
            builtins.dict  # pyright: ignore[reportUnknownMemberType]
            | typing.Dict  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
        ):
            validate_element: Callable[[Any], Any]
            match get_args(annotation):
                case (builtins.str, element_annotation):
                    validate_element = _parameter_validator(
                        element_annotation,
                        additional=None,
                        module=module,
                    )

                case other:  # pyright: ignore[reportUnnecessaryComparison]
                    raise TypeError("Unsupported dict type annotation", other)

            if validate := additional:

                def validated_dict(value: Any) -> dict[str, Any]:
                    if isinstance(value, dict):
                        values_dict: builtins.dict[str, Any] = {
                            key: validate_element(element)
                            for key, element in value.items()  # pyright: ignore[reportUnknownVariableType]
                            if isinstance(key, str)
                        }
                        validate(values_dict)
                        return values_dict
                    else:
                        raise TypeError("Invalid value", annotation, value)
            else:

                def validated_dict(value: Any) -> dict[str, Any]:
                    if isinstance(value, dict):
                        return {
                            key: validate_element(element)
                            for key, element in value.items()  # pyright: ignore[reportUnknownVariableType]
                            if isinstance(key, str)
                        }
                    else:
                        raise TypeError("Invalid value", annotation, value)

            return validated_dict

        case data_class if is_dataclass(data_class):
            # TODO: validate dataclass internals (i.e. nested dataclass)
            if validate := additional:

                def validated_dataclass(value: Any) -> Any:
                    if isinstance(value, dict):
                        validated: Any = data_class(**value)
                        validate(validated)
                        return validated
                    else:
                        raise TypeError("Invalid value", annotation, value)
            else:

                def validated_dataclass(value: Any) -> Any:
                    if isinstance(value, dict):
                        return data_class(**value)
                    else:
                        raise TypeError("Invalid value", annotation, value)

            return validated_dataclass

        case typing.Required | typing.NotRequired:
            match get_args(annotation):
                case [other, *_]:
                    return _parameter_validator(
                        other,
                        additional=additional,
                        module=module,
                    )

                case other:
                    raise TypeError("Unsupported type annotation", annotation)

        case typing.Annotated:
            match get_args(annotation):
                case [other, *_]:
                    return _parameter_validator(
                        other,
                        additional=additional,
                        module=module,
                    )

                case other:
                    raise TypeError("Unsupported annotated type", annotation)

        case collections_abc.Callable:  # pyright: ignore[reportUnknownMemberType]
            # TODO: validate callable signature
            if validate := additional:

                def validated_callable(value: Any) -> Any:
                    if callable(value) or inspect.isfunction(value):
                        validate(value)
                        return value
                    else:
                        raise TypeError("Invalid value", annotation, value)
            else:

                def validated_callable(value: Any) -> Any:
                    if callable(value) or inspect.isfunction(value):
                        return value
                    else:
                        raise TypeError("Invalid value", annotation, value)

            return validated_callable

        case protocol if issubclass(protocol, Protocol):
            # TODO: validate protocol details if needed
            if validate := additional:

                def validated_protocol(value: Any) -> Any:
                    if isinstance(value, protocol):
                        validate(value)
                        return value
                    else:
                        raise TypeError("Invalid value", annotation, value)
            else:

                def validated_protocol(value: Any) -> Any:
                    if isinstance(value, protocol):
                        return value
                    else:
                        raise TypeError("Invalid value", annotation, value)

            return validated_protocol

        case other:
            raise TypeError("Unsupported type annotation", other)
