import sys
from abc import ABCMeta
from collections.abc import Callable
from copy import deepcopy
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from types import GenericAlias
from typing import (
    Any,
    ClassVar,
    Generic,
    ParamSpec,
    Self,
    TypeVar,
    TypeVarTuple,
    cast,
    dataclass_transform,
    final,
    get_origin,
    overload,
)
from weakref import WeakValueDictionary

from draive.parameters.annotations import (
    ParameterDefaultFactory,
    allows_missing,
    object_annotations,
)
from draive.parameters.basic import BasicValue
from draive.parameters.path import ParameterPath
from draive.parameters.specification import (
    ParameterSpecification,
    parameter_specification,
)
from draive.parameters.validation import (
    ParameterValidationContext,
    ParameterValidationError,
    ParameterValidator,
    ParameterVerifier,
    parameter_validator,
)
from draive.utils import MISSING, Missing, freeze, is_missing, not_missing

__all__ = [
    "Field",
    "ParametrizedData",
]


@overload
def Field[Value](
    *,
    aliased: str | None = None,
    description: str | None = None,
    default: Value | Missing = MISSING,
    validator: ParameterValidator[Value] | Missing = MISSING,
    converter: Callable[[Value], BasicValue] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Field[Value](
    *,
    aliased: str | None = None,
    description: str | None = None,
    default_factory: Callable[[], Value] | Missing = MISSING,
    validator: ParameterValidator[Value] | Missing = MISSING,
    converter: Callable[[Value], BasicValue] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Field[Value](
    *,
    aliased: str | None = None,
    description: str | None = None,
    default: Value | Missing = MISSING,
    verifier: ParameterVerifier[Value] | None = None,
    converter: Callable[[Value], BasicValue] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Field[Value](
    *,
    aliased: str | None = None,
    description: str | None = None,
    default_factory: Callable[[], Value] | Missing = MISSING,
    verifier: ParameterVerifier[Value] | None = None,
    converter: Callable[[Value], BasicValue] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


def Field[Value](  # noqa: PLR0913 # Ruff - noqa: B008
    *,
    aliased: str | None = None,
    description: str | None = None,
    default: Value | Missing = MISSING,
    default_factory: Callable[[], Value] | Missing = MISSING,
    validator: ParameterValidator[Value] | Missing = MISSING,
    verifier: ParameterVerifier[Value] | None = None,
    converter: Callable[[Value], BasicValue] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value:  # it is actually a DataField, but type checker has to be fooled
    assert (  # nosec: B101
        is_missing(default_factory) or is_missing(default)
    ), "Can't specify both default value and factory"
    assert (  # nosec: B101
        is_missing(validator) or verifier is None
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
        aliased: str | None = None,
        description: str | None = None,
        default: Any | Missing = MISSING,
        default_factory: ParameterDefaultFactory[Any] | Missing = MISSING,
        validator: ParameterValidator[Any] | Missing = MISSING,
        verifier: ParameterVerifier[Any] | None = None,
        converter: Callable[[Any], BasicValue] | Missing = MISSING,
        specification: ParameterSpecification | Missing = MISSING,
    ) -> None:
        self.aliased: str | None = aliased
        self.description: str | None = description
        self.default: Any | Missing = default
        self.default_factory: Callable[[], Any] | Missing = default_factory
        self.validator: ParameterValidator[Any] | Missing = validator
        self.verifier: ParameterVerifier[Any] | None = verifier
        self.converter: Callable[[Any], BasicValue] | Missing = converter
        self.specification: ParameterSpecification | Missing = specification


@final
class DataParameter:
    @classmethod
    def of(  # noqa: PLR0913
        cls,
        annotation: Any,
        /,
        name: str,
        default: Any,
        prepare_specification: bool,
        type_arguments: dict[str, Any],
        globalns: dict[str, Any],
        localns: dict[str, Any] | None,
        recursion_guard: frozenset[type[Any]],
    ) -> Self:
        match default:
            case DataField() as data_field:
                return cls(
                    name=name,
                    aliased=data_field.aliased,
                    description=data_field.description,
                    annotation=annotation,
                    default=data_field.default,
                    default_factory=data_field.default_factory,
                    allows_missing=allows_missing(
                        annotation,
                        type_arguments=type_arguments,
                        globalns=globalns,
                        localns=localns,
                    ),
                    validator=data_field.validator
                    if not_missing(data_field.validator)
                    else parameter_validator(
                        annotation,
                        verifier=data_field.verifier,
                        type_arguments=type_arguments,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=recursion_guard,
                    ),
                    converter=data_field.converter,
                    specification=(
                        data_field.specification
                        if not_missing(data_field.specification)
                        else parameter_specification(
                            annotation,
                            description=data_field.description,
                            type_arguments=type_arguments,
                            globalns=globalns,
                            localns=localns,
                            recursion_guard=recursion_guard,
                        )
                    )
                    if prepare_specification
                    else MISSING,
                )

            case default:
                return cls(
                    name=name,
                    aliased=None,
                    description=None,
                    annotation=annotation,
                    default=default,
                    default_factory=MISSING,
                    allows_missing=allows_missing(
                        annotation,
                        type_arguments=type_arguments,
                        globalns=globalns,
                        localns=localns,
                    ),
                    validator=parameter_validator(
                        annotation,
                        verifier=None,
                        type_arguments=type_arguments,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=recursion_guard,
                    ),
                    converter=MISSING,
                    specification=parameter_specification(
                        annotation,
                        description=None,
                        type_arguments=type_arguments,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=recursion_guard,
                    )
                    if prepare_specification
                    else MISSING,
                )

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        aliased: str | None,
        description: str | None,
        annotation: Any,
        default: Any | Missing,
        default_factory: ParameterDefaultFactory[Any] | Missing,
        allows_missing: bool,
        validator: ParameterValidator[Any],
        converter: Callable[[Any], BasicValue] | Missing,
        specification: ParameterSpecification | Missing,
    ) -> None:
        self.name: str = name
        self.aliased: str | None = aliased
        self.description: str | None = description
        self.annotation: Any = annotation
        self.default_value: Callable[[], Any | Missing]
        if not_missing(default_factory):
            self.default_value = default_factory

        elif not_missing(default):
            self.default_value = lambda: default

        else:
            self.default_value = lambda: MISSING

        self.has_default: bool = not_missing(default_factory) or not_missing(default)
        self.allows_missing: bool = allows_missing
        self.validator: ParameterValidator[Any] = validator
        self.converter: Callable[[Any], BasicValue] | Missing = converter
        self.specification: ParameterSpecification | Missing = specification

        freeze(self)

    def validated(
        self,
        value: Any,
        /,
        context: ParameterValidationContext,
    ) -> Any:
        if is_missing(value):
            if self.has_default:
                return self.validator(self.default_value(), context.appending_path(f".{self.name}"))

            elif self.allows_missing:
                return MISSING

            else:
                raise ParameterValidationError.missing(
                    context=context.appending_path(f".{self.name}")
                )

        else:
            return self.validator(value, context.appending_path(f".{self.name}"))


@dataclass_transform(
    kw_only_default=True,
    frozen_default=True,
    field_specifiers=(Field,),
)
class ParametrizedDataMeta(ABCMeta):
    def __new__(  # noqa: PLR0913
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        type_base: type[Any] | None = None,
        type_parameters: tuple[TypeVar | ParamSpec | TypeVarTuple, ...] | None = None,
        type_arguments: tuple[type[Any], ...] | None = None,
        **kwargs: Any,
    ) -> Any:
        data_type = super().__new__(
            cls,
            name,
            bases,
            namespace,
            **kwargs,
        )

        data_type.__TYPE_BASE__ = type_base or data_type  # pyright: ignore[reportAttributeAccessIssue]
        data_type.__TYPE_ARGUMENTS__ = type_arguments or ()  # pyright: ignore[reportAttributeAccessIssue]

        if not bases:  # handle base class
            return data_type

        globalns: dict[str, Any]
        if data_type.__module__:
            globalns = sys.modules.get(data_type.__module__).__dict__

        else:
            globalns = {}

        localns: dict[str, Any] = {data_type.__name__: data_type}
        recursion_guard: frozenset[type[Any]] = frozenset({data_type})
        resolved_type_parameters: dict[str, Any] = {
            parameter.__name__: argument
            for (parameter, argument) in zip(
                type_parameters or (),
                type_arguments or (),
                strict=False,
            )
        }
        parameters: dict[str, DataParameter] = {}
        prepare_specification: bool = hasattr(
            data_type,
            "__PARAMETERS_SPECIFICATION__",
        )
        properties_specification: dict[str, ParameterSpecification] = {}
        aliased_required_specification: list[str] = []

        for key, annotation in object_annotations(
            data_type,
            globalns,
            localns,
        ).items():
            # do not include ClassVars and private or dunder items
            if ((get_origin(annotation) or annotation) is ClassVar) or key.startswith("__"):
                continue

            parameter: DataParameter = DataParameter.of(
                annotation,
                name=key,
                default=getattr(data_type, key, MISSING),
                # prepare specification only for the data models
                prepare_specification=prepare_specification,
                type_arguments=resolved_type_parameters,
                globalns=globalns,
                localns=localns,
                recursion_guard=recursion_guard,
            )
            parameters[key] = parameter

            if prepare_specification:  # check and use specification when needed
                if not_missing(parameter.specification):
                    properties_specification[parameter.name] = parameter.specification

                    if not (parameter.has_default or parameter.allows_missing):
                        aliased_required_specification.append(key)

                else:
                    # if any parameter does not have specification then whole type does not have one
                    raise TypeError("Property %s of %s, has no specification!", key, annotation)

        data_type.__PARAMETERS__ = parameters  # pyright: ignore[reportAttributeAccessIssue]
        data_type.__PARAMETERS_LIST__ = list(parameters.values())  # pyright: ignore[reportAttributeAccessIssue]

        if prepare_specification:
            if properties_specification:
                data_type.__PARAMETERS_SPECIFICATION__ = {  # pyright: ignore[reportAttributeAccessIssue]
                    "type": "object",
                    "properties": properties_specification,
                    "required": aliased_required_specification,
                }

            else:
                data_type.__PARAMETERS_SPECIFICATION__ = {  # pyright: ignore[reportAttributeAccessIssue]
                    "type": "object",
                    "additionalProperties": True,
                }

        data_type.__slots__ = frozenset(parameters.keys())  # pyright: ignore[reportAttributeAccessIssue]
        data_type.__match_args__ = data_type.__slots__  # pyright: ignore[reportAttributeAccessIssue]
        data_type._ = ParameterPath(root=data_type, parameter=data_type)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        return data_type

    def __instancecheck__(
        cls,
        instance: Any,
    ) -> bool:
        # Avoid calling ABC _abc_subclasscheck | https://github.com/python/cpython/issues/92810
        if not hasattr(instance, "__PARAMETERS__"):
            return False  # early exit if not subclass of ParametrizedData

        return super().__instancecheck__(instance)

    def __subclasscheck__(
        cls,
        subclass: Any,
    ) -> bool:
        # Avoid calling ABC _abc_subclasscheck | https://github.com/python/cpython/issues/92810
        if not hasattr(subclass, "__PARAMETERS__"):
            return False  # early exit if not subclass of ParametrizedData

        # check for parametrized subtypes within matching parameters
        if cls.__TYPE_BASE__ == subclass.__TYPE_BASE__:  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            # check for parametrized variant by checking parameters
            return all(
                issubclass(subclass_param, cls_param)
                for (cls_param, subclass_param) in zip(  # pyright: ignore[reportUnknownVariableType]
                    cls.__TYPE_ARGUMENTS__,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
                    subclass.__TYPE_ARGUMENTS__,
                    strict=False,
                )
            )

        return super().__subclasscheck__(subclass)


_types_cache: WeakValueDictionary[
    tuple[
        Any,
        tuple[Any, ...],
    ],
    Any,
] = WeakValueDictionary()


class ParametrizedData(metaclass=ParametrizedDataMeta):
    _: ClassVar[Self]
    __TYPE_BASE__: ClassVar[type[ParametrizedDataMeta]]
    __TYPE_ARGUMENTS__: ClassVar[tuple[type[Any], ...]]
    __PARAMETERS__: ClassVar[dict[str, DataParameter]]
    __PARAMETERS_LIST__: ClassVar[list[DataParameter]]

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        if self.__PARAMETERS_LIST__:
            for parameter in self.__PARAMETERS_LIST__:
                parameter_context: ParameterValidationContext = ParameterValidationContext(
                    path=(self.__class__.__qualname__,)
                )
                object.__setattr__(
                    self,  # pyright: ignore[reportUnknownArgumentType]
                    parameter.name,
                    parameter.validated(
                        kwargs.get(
                            parameter.name,
                            kwargs.get(
                                parameter.aliased,
                                MISSING,
                            )
                            if parameter.aliased
                            else MISSING,
                        ),
                        context=parameter_context,
                    ),
                )

        else:
            for key, value in kwargs.items():
                object.__setattr__(
                    self,  # pyright: ignore[reportUnknownArgumentType]
                    key,
                    value,
                )

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

        if any(isinstance(argument, TypeVar) for argument in type_arguments):
            # if we got unfinished type treat it as an alias
            return cast(type[Self], GenericAlias(cls, type_arguments))

        assert len(type_arguments) == len(  # nosec: B101
            cls.__type_params__
        ), "Type arguments count has to match parameters count"

        # TODO: add UnionType ordering for equal types
        if cached := _types_cache.get((cls, type_arguments)):
            return cached

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
            cls=cls.__class__,
            name=name,
            bases=bases,
            namespace={
                "__module__": cls.__module__,
            },
            type_base=cls,
            type_parameters=cls.__type_params__,  # pyright: ignore[reportArgumentType]
            type_arguments=type_arguments,
        )
        _types_cache[(cls, type_arguments)] = parametrized_type
        return parametrized_type

    @classmethod
    def path[Parameter](
        cls,
        path: Parameter,
        /,
    ) -> ParameterPath[Self, Parameter]:
        assert isinstance(  # nosec: B101
            path, ParameterPath
        ), "Prepare parameter path by using Self._.path.to.property"

        return cast(ParameterPath[Self, Parameter], path)

    @classmethod
    def validator(
        cls,
        /,
        value: Any,
        context: ParameterValidationContext,
    ) -> Self:
        match value:
            case valid if isinstance(valid, cls):
                return valid

            case {**values}:
                return cls(**values)

            case _:
                raise ParameterValidationError.invalid_type(
                    expected=cls,
                    received=value,
                    context=context,
                )

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

    # TODO: find a way to generate signature similar to dataclass __init__
    # or try using ParameterPath if found a way to generate setters
    def updated(
        self,
        /,
        **parameters: Any,
    ) -> Self:
        if not parameters or parameters.keys().isdisjoint(self.__PARAMETERS__.keys()):
            return self  # do not make a copy when nothing will be updated

        updated: Self = self.__new__(self.__class__)
        for parameter in self.__PARAMETERS_LIST__:
            parameter_context: ParameterValidationContext = ParameterValidationContext(
                path=(self.__class__.__qualname__,),
            )
            validated_value: Any
            if parameter.name in parameters:
                validated_value = parameter.validated(
                    parameters[parameter.name],
                    context=parameter_context,
                )

            elif parameter.aliased and parameter.aliased in parameters:
                validated_value = parameter.validated(
                    parameters[parameter.aliased],
                    context=parameter_context,
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

    def updating[Value](
        self,
        path: ParameterPath[Self, Value] | Value,
        /,
        value: Value,
    ) -> Self:
        assert isinstance(  # nosec: B101
            path, ParameterPath
        ), "Prepare parameter path by using Self._.path.to.property"

        return cast(ParameterPath[Self, Value], path)(self, updated=value)

    def updating_parameter(
        self,
        name: str,
        /,
        value: Any,
    ) -> Self:
        assert (  # nosec: B101
            name in self.__PARAMETERS__
        ), f"Parameter {name} does not exist in {self.__class__.__qualname__}"

        updated: Self = self.__new__(self.__class__)
        for parameter in self.__PARAMETERS_LIST__:
            parameter_context: ParameterValidationContext = ParameterValidationContext(
                path=(self.__class__.__qualname__,),
            )
            validated_value: Any
            if parameter.name == name:
                validated_value = parameter.validated(
                    value,
                    context=parameter_context,
                )

            elif parameter.aliased and parameter.aliased == name:
                validated_value = parameter.validated(
                    value,
                    context=parameter_context,
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

    def __deepcopy__(
        self,
        memo: dict[int, Any] | None,
    ) -> Self:
        copy: Self = self.__class__(
            **{
                key: deepcopy(
                    value,
                    memo,
                )
                for key, value in vars(self).items()
            }
        )
        return copy


# based on python dataclass asdict but simplified
def _data_dict(  # noqa: PLR0911
    data: Any,
    /,
    aliased: bool,
    converter: Callable[[Any], BasicValue] | None,
) -> Any:
    # use converter if able
    if converter := converter:
        return converter(data)

    match data:
        case str() | None | int() | float() | bool():
            return data  # use basic value types as they are

        case parametrized_data if hasattr(parametrized_data, "__PARAMETERS_LIST__"):
            # convert parametrized data to dict
            if not parametrized_data.__PARAMETERS_LIST__:  # if base - use all variables
                return cast(dict[str, Any], vars(parametrized_data))

            elif aliased:  # alias if needed
                return {
                    field.aliased or field.name: _data_dict(
                        getattr(parametrized_data, field.name),
                        aliased=aliased,
                        converter=field.converter,
                    )
                    for field in parametrized_data.__PARAMETERS_LIST__
                }

            else:
                return {
                    field.name: _data_dict(
                        getattr(parametrized_data, field.name),
                        aliased=aliased,
                        converter=field.converter,
                    )
                    for field in parametrized_data.__PARAMETERS_LIST__
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
    converter: Callable[[Any], BasicValue] | None,
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

        case parametrized_data if hasattr(parametrized_data, "__PARAMETERS_LIST__"):
            # convert parametrized data to dict
            if not parametrized_data.__PARAMETERS_LIST__:  # if base - use all variables
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
                for field in parametrized_data.__PARAMETERS_LIST__:
                    element: str = _data_str(
                        getattr(parametrized_data, field.name),
                        aliased=aliased,
                        converter=field.converter,
                    ).replace("\n", "\n  ")

                    string += f"\n{field.aliased or field.name}: {element}"

                return string

            else:
                string: str = ""
                for field in parametrized_data.__PARAMETERS_LIST__:
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
