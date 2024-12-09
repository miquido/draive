from collections.abc import Callable
from copy import deepcopy
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from types import EllipsisType, GenericAlias
from typing import (
    Any,
    ClassVar,
    Generic,
    Self,
    TypeVar,
    cast,
    dataclass_transform,
    final,
    get_origin,
    overload,
)
from weakref import WeakValueDictionary

from haiway import MISSING, AttributePath, Missing
from haiway.state import AttributeAnnotation, attribute_annotations

from draive.parameters.parameter import Parameter
from draive.parameters.specification import (
    ParameterSpecification,
    ParametersSpecification,
)
from draive.parameters.types import ParameterConverter, ParameterDefaultFactory
from draive.parameters.validation import (
    ParameterValidationContext,
    ParameterValidator,
    ParameterVerifier,
)

__all__ = [
    "Field",
    "ParametrizedData",
]


@overload
def Field[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default: Value | Missing = MISSING,
    validator: ParameterValidator[Value] | Missing = MISSING,
    converter: ParameterConverter[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Field[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default_factory: ParameterDefaultFactory[Value] | Missing = MISSING,
    validator: ParameterValidator[Value] | Missing = MISSING,
    converter: ParameterConverter[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Field[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default: Value | Missing = MISSING,
    verifier: ParameterVerifier[Value] | Missing = MISSING,
    converter: ParameterConverter[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Field[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default_factory: ParameterDefaultFactory[Value] | Missing = MISSING,
    verifier: ParameterVerifier[Value] | Missing = MISSING,
    converter: ParameterConverter[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


def Field[Value](  # noqa: PLR0913
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default: Value | Missing = MISSING,
    default_factory: ParameterDefaultFactory[Value] | Missing = MISSING,
    validator: ParameterValidator[Value] | Missing = MISSING,
    verifier: ParameterVerifier[Value] | Missing = MISSING,
    converter: ParameterConverter[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value:  # it is actually a DataField, but type checker has to be fooled
    assert (  # nosec: B101
        default is MISSING or default_factory is MISSING
    ), "Can't specify both default value and factory"
    assert (  # nosec: B101
        description is MISSING or specification is MISSING
    ), "Can't specify both description and specification"
    assert (  # nosec: B101
        validator is MISSING or verifier is MISSING
    ), "Can't specify both validator and verifier"

    return cast(
        Value,
        DataField(
            aliased=aliased,
            description=description,
            default=default,
            default_factory=default_factory,
            validator=validator,
            verifier=verifier,
            converter=converter,
            specification=specification,
        ),
    )


@final
class DataField:
    def __init__(  # noqa: PLR0913
        self,
        aliased: str | None,
        description: str | Missing,
        default: Any | Missing,
        default_factory: ParameterDefaultFactory[Any] | Missing,
        validator: ParameterValidator[Any] | Missing,
        verifier: ParameterVerifier[Any] | Missing,
        converter: ParameterConverter[Any] | Missing,
        specification: ParameterSpecification | Missing,
    ) -> None:
        self.aliased: str | None = aliased
        self.description: str | Missing = description
        self.default: Any | Missing = default
        self.default_factory: Callable[[], Any] | Missing = default_factory
        self.validator: ParameterValidator[Any] | Missing = validator
        self.verifier: ParameterVerifier[Any] | Missing = verifier
        self.converter: ParameterConverter[Any] | Missing = converter
        self.specification: ParameterSpecification | Missing = specification


def _resolve_field(
    attribute: AttributeAnnotation,
    /,
    *,
    name: str,
    default: Any,
) -> Parameter[Any]:
    match default:
        case DataField() as data_field:
            return Parameter[Any].of(
                attribute,
                name=name,
                alias=data_field.aliased,
                description=data_field.description,
                default_value=data_field.default,
                default_factory=data_field.default_factory,
                validator=data_field.validator,
                verifier=data_field.verifier,
                converter=MISSING,
                specification=data_field.specification,
            )

        case default:
            return Parameter[Any].of(
                attribute,
                name=name,
                alias=None,
                description=MISSING,
                default_value=default,
                default_factory=MISSING,
                validator=MISSING,
                verifier=MISSING,
                converter=MISSING,
                specification=MISSING,
            )


@dataclass_transform(
    kw_only_default=True,
    frozen_default=True,
    field_specifiers=(Field,),
)
class ParametrizedDataMeta(type):
    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        type_base: type[Any] | None = None,
        type_parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        data_type = super().__new__(
            cls,
            name,
            bases,
            namespace,
            **kwargs,
        )

        parameters: dict[str, Parameter[Any]] = {}
        parameters_specification: dict[str, ParameterSpecification] = {}
        parameters_specification_required: list[str] = []  # TODO: FIXME: !!!
        if bases:  # handle base class
            for key, annotation in attribute_annotations(
                data_type,
                type_parameters=type_parameters,
            ).items():
                # do not include ClassVars and dunder items
                if ((get_origin(annotation) or annotation) is ClassVar) or key.startswith("__"):
                    continue

                parameter: Parameter[Any] = _resolve_field(
                    annotation,
                    name=key,
                    default=getattr(data_type, key, MISSING),
                )
                parameters_specification[key] = parameter.specification

                parameters[key] = parameter

        data_type.__PARAMETERS__ = parameters  # pyright: ignore[reportAttributeAccessIssue]
        if parameters_specification:
            data_type.__PARAMETERS_SPECIFICATION__ = {  # pyright: ignore[reportAttributeAccessIssue]
                "type": "object",
                "properties": parameters_specification,
                "required": parameters_specification_required,
            }

        else:
            data_type.__PARAMETERS_SPECIFICATION__ = {  # pyright: ignore[reportAttributeAccessIssue]
                "type": "object",
                "additionalProperties": True,
            }

        data_type.__slots__ = frozenset(parameters.keys())  # pyright: ignore[reportAttributeAccessIssue]
        data_type.__match_args__ = data_type.__slots__  # pyright: ignore[reportAttributeAccessIssue]
        data_type._ = AttributePath(data_type, attribute=data_type)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

        return data_type


_types_cache: WeakValueDictionary[
    tuple[
        Any,
        tuple[Any, ...],
    ],
    Any,
] = WeakValueDictionary()


class ParametrizedData(metaclass=ParametrizedDataMeta):
    _: ClassVar[Self]
    __IMMUTABLE__: ClassVar[EllipsisType] = ...
    __PARAMETERS__: ClassVar[dict[str, Parameter[Any]]]
    __PARAMETERS_SPECIFICATION__: ClassVar[ParametersSpecification]

    def __class_getitem__(
        cls,
        type_argument: tuple[type[Any], ...] | type[Any],
    ) -> type[Self]:
        assert Generic in cls.__bases__, "Can't specialize non generic type!"  # nosec: B101

        type_arguments: tuple[type[Any], ...]
        match type_argument:
            case [*arguments]:
                type_arguments = tuple(arguments)

            case argument:
                type_arguments = (argument,)

        if any(isinstance(argument, TypeVar) for argument in type_arguments):  # pyright: ignore[reportUnnecessaryIsInstance]
            # if we got unfinished type treat it as an alias instead of resolving
            return cast(type[Self], GenericAlias(cls, type_arguments))

        assert len(type_arguments) == len(  # nosec: B101
            cls.__type_params__
        ), "Type arguments count has to match type parameters count"

        if cached := _types_cache.get((cls, type_arguments)):
            return cached

        type_parameters: dict[str, Any] = {
            parameter.__name__: argument
            for (parameter, argument) in zip(
                cls.__type_params__ or (),
                type_arguments or (),
                strict=False,
            )
        }

        parameter_names: str = ",".join(
            getattr(
                argument,
                "__name__",
                str(argument),
            )
            for argument in type_arguments
        )
        name: str = f"{cls.__name__}[{parameter_names}]"
        bases: tuple[type[Self]] = (cls,)

        parametrized_type: type[Self] = ParametrizedDataMeta.__new__(
            cls.__class__,
            name=name,
            bases=bases,
            namespace={"__module__": cls.__module__},
            type_parameters=type_parameters,
        )
        _types_cache[(cls, type_arguments)] = parametrized_type
        return parametrized_type

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        if self.__PARAMETERS__:
            with ParameterValidationContext().scope(self.__class__.__qualname__) as context:
                for name, parameter in self.__PARAMETERS__.items():
                    object.__setattr__(
                        self,  # pyright: ignore[reportUnknownArgumentType]
                        name,
                        parameter.validated(
                            parameter.find(kwargs),
                            context=context,
                        ),
                    )

        else:
            for key, value in kwargs.items():
                object.__setattr__(
                    self,  # pyright: ignore[reportUnknownArgumentType]
                    key,
                    value,
                )

    @classmethod
    def validate(
        cls,
        value: Any,
        /,
        context: ParameterValidationContext,
    ) -> Self:
        match value:
            case valid if isinstance(valid, cls):
                return valid

            case {**values}:
                return cls(**values)

            case _:
                raise TypeError(f"Expected '{cls.__name__}', received '{type(value).__name__}'")

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        /,
    ) -> Self:
        return cls(**value)

    def as_dict(
        self,
        aliased: bool = True,
    ) -> dict[str, Any]:
        return _data_dict(
            self,
            aliased=aliased,
            converter=None,
        )

    def updating[Value](
        self,
        path: AttributePath[Self, Value] | Value,
        /,
        value: Value,
    ) -> Self:
        assert isinstance(  # nosec: B101
            path, AttributePath
        ), "Prepare parameter path by using Self._.path.to.property or explicitly"

        return cast(AttributePath[Self, Value], path)(self, updated=value)

    def updated(
        self,
        **kwargs: Any,
    ) -> Self:
        return self.__replace__(**kwargs)

    def __replace__(
        self,
        /,
        **parameters: Any,
    ) -> Self:
        if not parameters or parameters.keys().isdisjoint(self.__PARAMETERS__.keys()):
            return self  # do not make a copy when nothing will be updated

        with ParameterValidationContext().scope(self.__class__.__qualname__) as context:
            updated: Self = self.__new__(self.__class__)
            for parameter in self.__PARAMETERS__.values():
                validated_value: Any
                if parameter.name in parameters:
                    validated_value = parameter.validated(
                        parameters[parameter.name],
                        context=context,
                    )

                elif parameter.alias and parameter.alias in parameters:
                    validated_value = parameter.validated(
                        parameters[parameter.alias],
                        context=context,
                    )

                else:  # no need to validate again when reusing current value
                    validated_value = object.__getattribute__(
                        self,
                        parameter.name,
                    )

                object.__setattr__(
                    updated,
                    parameter.name,
                    validated_value,
                )

            return updated

    def __str__(self) -> str:
        return _data_str(self, aliased=True, converter=None).strip()

    def __repr__(self) -> str:
        return str(self.as_dict())

    def __eq__(self, other: Any) -> bool:
        if other.__class__ != self.__class__:
            return False

        return all(
            getattr(self, key, MISSING) == getattr(other, key, MISSING)
            for key in self.__PARAMETERS__.keys()
        )

    def __contains__(
        self,
        element: Any,
    ) -> bool:
        return element in vars(self)

    def __getitem__(
        self,
        name: str,
    ) -> Any | Missing:
        return vars(self).get(name, MISSING)

    @overload
    def get(
        self,
        name: str,
    ) -> Any | Missing: ...

    @overload
    def get[Default](
        self,
        name: str,
        default: Default,
    ) -> Any | Default: ...

    def get(
        self,
        name: str,
        default: Any | Missing = MISSING,
    ) -> Any | Missing:
        if self.__contains__(name):
            return self.__getitem__(name)

        else:
            return default

    def __setattr__(
        self,
        __name: str,
        __value: Any,
    ) -> None:
        raise AttributeError(f"{self.__class__.__qualname__} is frozen and can't be modified")

    def __delattr__(
        self,
        __name: str,
    ) -> None:
        raise AttributeError(f"{self.__class__.__qualname__} is frozen and can't be modified")

    def __copy__(self) -> Self:
        return self.__class__(**vars(self))


# based on python dataclass asdict but simplified
def _data_dict(  # noqa: PLR0911
    data: Any,
    /,
    aliased: bool,
    converter: ParameterConverter[Any] | None,
) -> Any:
    # use converter if able
    if converter := converter:
        return converter(data)

    match data:
        case str() | None | int() | float() | bool():
            return data  # use basic value types as they are

        case parametrized_data if hasattr(parametrized_data, "__PARAMETERS__"):
            # convert parametrized data to dict
            if not parametrized_data.__PARAMETERS__:  # if base - use all variables
                return cast(dict[str, Any], vars(parametrized_data))

            elif aliased:  # alias if needed
                return {
                    field.aliased or field.name: _data_dict(
                        getattr(parametrized_data, field.name),
                        aliased=aliased,
                        converter=field.converter,
                    )
                    for field in parametrized_data.__PARAMETERS__.values()
                }

            else:
                return {
                    field.name: _data_dict(
                        getattr(parametrized_data, field.name),
                        aliased=aliased,
                        converter=field.converter,
                    )
                    for field in parametrized_data.__PARAMETERS__.values()
                }

        case {**elements}:  # replace mapping with dict
            return {
                key: _data_dict(
                    value,
                    aliased=aliased,
                    converter=None,
                )
                for key, value in elements.items()
            }

        case [*values]:  # replace sequence with list
            return [
                _data_dict(
                    value,
                    aliased=aliased,
                    converter=None,
                )
                for value in values
            ]

        case dataclass if is_dataclass(dataclass):
            return {
                f.name: _data_dict(
                    getattr(dataclass, f.name),
                    aliased=aliased,
                    converter=None,
                )
                for f in dataclass_fields(dataclass)
            }

        case other:  # for other types use deepcopy
            return deepcopy(other)


def _data_str(  # noqa: PLR0911, PLR0912, C901
    data: Any,
    /,
    aliased: bool,
    converter: ParameterConverter[Any] | None,
) -> str:
    # use converter if able
    if converter := converter:
        return _data_str(
            converter(data),
            aliased=aliased,
            converter=None,
        )

    match data:
        case str() as string:
            return string

        case None | int() | float() | bool():
            return str(data)

        case parametrized_data if hasattr(parametrized_data, "__PARAMETERS__"):
            # convert parametrized data to dict
            if not parametrized_data.__PARAMETERS__:  # if base - use all variables
                string: str = ""
                for key, value in vars(parametrized_data).items():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                    element: str = _data_str(
                        value,
                        aliased=aliased,
                        converter=None,
                    ).replace("\n", "\n  ")

                    string += f"\n{key}: {element}"

                return string

            elif aliased:  # alias if needed
                string: str = ""
                for field in parametrized_data.__PARAMETERS__.values():
                    element: str = _data_str(
                        getattr(parametrized_data, field.name),
                        aliased=aliased,
                        converter=field.converter,
                    ).replace("\n", "\n  ")

                    string += f"\n{field.aliased or field.name}: {element}"

                return string

            else:
                string: str = ""
                for field in parametrized_data.__PARAMETERS__.values():
                    element: str = _data_str(
                        getattr(parametrized_data, field.name),
                        aliased=aliased,
                        converter=field.converter,
                    ).replace("\n", "\n  ")

                    string += f"\n{field.name}: {element}"

                return string

        case {**elements}:  # replace mapping with dict
            string: str = ""
            for key, value in elements.items():
                element: str = _data_str(
                    value,
                    aliased=aliased,
                    converter=None,
                ).replace("\n", "\n  ")

                string += f"\n{key}: {element}"

            return string

        case [*values]:  # replace sequence with list
            string: str = ""
            for value in values:
                element: str = (
                    _data_str(
                        value,
                        aliased=aliased,
                        converter=None,
                    )
                    .replace("\n", "\n  ")
                    .strip()
                )

                string += f"\n- {element}"

            return string

        case dataclass if is_dataclass(dataclass):
            string: str = ""
            for field in dataclass_fields(dataclass):
                element: str = _data_str(
                    getattr(parametrized_data, field.name),
                    aliased=aliased,
                    converter=None,
                ).replace("\n", "\n  ")

                string += f"\n{field.name}: {element}"

            return string

        case other:  # for other types use its str
            return str(other)
