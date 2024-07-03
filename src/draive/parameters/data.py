import sys
from collections.abc import Callable
from copy import deepcopy
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any, ClassVar, Self, cast, dataclass_transform, final, get_origin, overload

from draive.parameters.annotations import (
    ParameterDefaultFactory,
    allows_missing,
    object_annotations,
)
from draive.parameters.basic import BasicValue
from draive.parameters.path import ParameterPath
from draive.parameters.specification import (
    ParameterSpecification,
    ParametersSpecification,
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
                return self.validator(self.default_value(), (*context, f".{self.name}"))

            elif self.allows_missing:
                return MISSING

            else:
                raise ParameterValidationError.missing(context=(*context, f".{self.name}"))

        else:
            return self.validator(value, (*context, f".{self.name}"))

    @classmethod
    def of(  # noqa: PLR0913
        cls,
        annotation: Any,
        /,
        name: str,
        default: Any,
        prepare_specification: bool,
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
                        globalns=globalns,
                        localns=localns,
                    ),
                    validator=data_field.validator
                    if not_missing(data_field.validator)
                    else parameter_validator(
                        annotation,
                        verifier=data_field.verifier,
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
                        globalns=globalns,
                        localns=localns,
                    ),
                    validator=parameter_validator(
                        annotation,
                        verifier=None,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=recursion_guard,
                    ),
                    converter=MISSING,
                    specification=parameter_specification(
                        annotation,
                        description=None,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=recursion_guard,
                    )
                    if prepare_specification
                    else MISSING,
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
        classdict: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        data_type = type.__new__(
            cls,
            name,
            bases,
            classdict,
            **kwargs,
        )
        if not bases:
            return data_type

        is_data_model: bool = "DataModel" in tuple(base.__name__ for base in bases)
        globalns: dict[str, Any] = sys.modules.get(data_type.__module__).__dict__
        localns: dict[str, Any] = {data_type.__name__: data_type}
        recursion_guard: frozenset[type[Any]] = frozenset({data_type})
        parameters: dict[str, DataParameter] = {}
        properties_specification: dict[str, ParameterSpecification] | Missing = {}
        aliased_required: list[str] = []
        for key, annotation in object_annotations(
            data_type,
            globalns,
            localns,
        ).items():
            # do not include ClassVars and private or dunder items
            if ((get_origin(annotation) or annotation) is ClassVar) or key.startswith("_"):
                continue

            parameter: DataParameter = DataParameter.of(
                annotation,
                name=key,
                default=getattr(data_type, key, MISSING),
                # prepare specification only for the data models
                prepare_specification=is_data_model,
                globalns=globalns,
                localns=localns,
                recursion_guard=recursion_guard,
            )
            parameters[key] = parameter

            if not_missing(properties_specification):
                if not_missing(parameter.specification):
                    properties_specification[parameter.name] = parameter.specification

                    if not (parameter.has_default or parameter.allows_missing):
                        aliased_required.append(key)

                else:
                    # if any parameter does not have specification
                    # then whole type does not have one
                    properties_specification = MISSING

            else:
                continue  # skip if we already have missing specification

        assert parameters or name in {  # nosec: B101
            "DataModel",
            "Stateless",
        }, "Can't prepare parametrized data without parameters!"
        data_type.__PARAMETERS__ = parameters  # pyright: ignore[reportAttributeAccessIssue]
        data_type.__PARAMETERS_LIST__ = list(parameters.values())  # pyright: ignore[reportAttributeAccessIssue]
        if not parameters and name == "DataModel":
            data_type.__PARAMETERS_SPECIFICATION__ = {  # pyright: ignore[reportAttributeAccessIssue]
                "type": "object",
                "additionalProperties": True,
            }

        elif not_missing(properties_specification):
            data_type.__PARAMETERS_SPECIFICATION__ = {  # pyright: ignore[reportAttributeAccessIssue]
                "type": "object",
                "properties": properties_specification,
                "required": aliased_required,
            }

        else:
            data_type.__PARAMETERS_SPECIFICATION__ = MISSING  # pyright: ignore[reportAttributeAccessIssue]

        data_type.__slots__ = frozenset(parameters.keys())  # pyright: ignore[reportAttributeAccessIssue]
        data_type.__match_args__ = data_type.__slots__  # pyright: ignore[reportAttributeAccessIssue]
        data_type._ = ParameterPath(data_type, data_type)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        return data_type


class ParametrizedData(metaclass=ParametrizedDataMeta):
    _: ClassVar[Self]
    __PARAMETERS__: ClassVar[dict[str, DataParameter]]
    __PARAMETERS_LIST__: ClassVar[list[DataParameter]]
    __PARAMETERS_SPECIFICATION__: ClassVar[ParametersSpecification | Missing]

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        if self.__class__.__PARAMETERS_LIST__:
            for parameter in self.__class__.__PARAMETERS_LIST__:
                parameter_context: ParameterValidationContext = (self.__class__.__qualname__,)
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
            case self if isinstance(self, cls):
                return self

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
        if not parameters:
            return self

        updated: Self = self.__class__.__new__(self.__class__)
        for parameter in self.__class__.__PARAMETERS_LIST__:
            parameter_context: ParameterValidationContext = (self.__class__.__qualname__,)
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
            name in self.__class__.__PARAMETERS__
        ), f"Parameter {name} does not exist in {self.__class__.__qualname__}"

        updated: Self = self.__class__.__new__(self.__class__)
        for parameter in self.__class__.__PARAMETERS_LIST__:
            parameter_context: ParameterValidationContext = (self.__class__.__qualname__,)
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
            for key in self.__class__.__PARAMETERS__.keys()
        )

    def __getitem__(
        self,
        name: str,
    ) -> Any | Missing:
        return vars(self).get(name, MISSING)

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

        case parametrized_data if hasattr(parametrized_data.__class__, "__PARAMETERS_LIST__"):
            # convert parametrized data to dict
            if not parametrized_data.__class__.__PARAMETERS_LIST__:  # if base - use all variables
                return cast(dict[str, Any], vars(parametrized_data))

            elif aliased:  # alias if needed
                return {
                    field.aliased or field.name: _data_dict(
                        getattr(parametrized_data, field.name),
                        aliased=aliased,
                        converter=field.converter,
                    )
                    for field in parametrized_data.__class__.__PARAMETERS_LIST__
                }

            else:
                return {
                    field.name: _data_dict(
                        getattr(parametrized_data, field.name),
                        aliased=aliased,
                        converter=field.converter,
                    )
                    for field in parametrized_data.__class__.__PARAMETERS_LIST__
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

        case parametrized_data if hasattr(parametrized_data.__class__, "__PARAMETERS_LIST__"):
            # convert parametrized data to dict
            if not parametrized_data.__class__.__PARAMETERS_LIST__:  # if base - use all variables
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
                for field in parametrized_data.__class__.__PARAMETERS_LIST__:
                    element: str = _data_str(
                        getattr(parametrized_data, field.name),
                        aliased=aliased,
                        converter=field.converter,
                    ).replace("\n", "\n  ")

                    string += f"\n{field.aliased or field.name}: {element}"

                return string

            else:
                string: str = ""
                for field in parametrized_data.__class__.__PARAMETERS_LIST__:
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
                element: str = _data_str(
                    value,
                    aliased=aliased,
                    converter=None,
                ).replace("\n", "\n  ")

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
