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
    alias: str | None = None,
    description: str | None = None,
    default: Value | Missing = MISSING,
    validator: ParameterValidator[Value] | Missing = MISSING,
    converter: Callable[[Value], BasicValue] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Field[Value](
    *,
    alias: str | None = None,
    description: str | None = None,
    default_factory: Callable[[], Value] | Missing = MISSING,
    validator: ParameterValidator[Value] | Missing = MISSING,
    converter: Callable[[Value], BasicValue] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Field[Value](
    *,
    alias: str | None = None,
    description: str | None = None,
    default: Value | Missing = MISSING,
    verifier: ParameterVerifier[Value] | None = None,
    converter: Callable[[Value], BasicValue] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Field[Value](
    *,
    alias: str | None = None,
    description: str | None = None,
    default_factory: Callable[[], Value] | Missing = MISSING,
    verifier: ParameterVerifier[Value] | None = None,
    converter: Callable[[Value], BasicValue] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


def Field[Value](  # noqa: PLR0913 # Ruff - noqa: B008
    *,
    alias: str | None = None,
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
            alias=alias,
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
        alias: str | None = None,
        description: str | None = None,
        default: Any | Missing = MISSING,
        default_factory: ParameterDefaultFactory[Any] | Missing = MISSING,
        validator: ParameterValidator[Any] | Missing = MISSING,
        verifier: ParameterVerifier[Any] | None = None,
        converter: Callable[[Any], BasicValue] | Missing = MISSING,
        specification: ParameterSpecification | Missing = MISSING,
    ) -> None:
        self.alias: str | None = alias
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
        alias: str | None,
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
        self.alias: str | None = alias
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
        globalns: dict[str, Any],
        localns: dict[str, Any] | None,
        recursion_guard: frozenset[type[Any]],
    ) -> Self:
        match default:
            case DataField() as data_field:
                return cls(
                    name=name,
                    alias=data_field.alias,
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
                    specification=data_field.specification
                    if not_missing(data_field.specification)
                    else parameter_specification(
                        annotation,
                        description=data_field.description,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=recursion_guard,
                    ),
                )

            case default:
                return cls(
                    name=name,
                    alias=None,
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
                    ),
                )


@dataclass_transform(
    kw_only_default=True,
    frozen_default=True,
    field_specifiers=(),
)
class ParametrizedDataMeta(type):
    _: Any
    __PARAMETERS__: dict[str, DataParameter]
    __PARAMETERS_SPECIFICATION__: ParametersSpecification | Missing

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
                    # if any parameter does not have specification then whole type does not have one
                    properties_specification = MISSING  # pyright: ignore[reportConstantRedefinition]

            else:
                continue  # skip if we already have missing specification

        data_type.__PARAMETERS__ = parameters  # pyright: ignore[reportConstantRedefinition]
        if not_missing(properties_specification):
            data_type.__PARAMETERS_SPECIFICATION__ = {  # pyright: ignore[reportConstantRedefinition]
                "type": "object",
                "properties": properties_specification,
                "required": aliased_required,
            }

        else:
            data_type.__PARAMETERS_SPECIFICATION__ = MISSING  # pyright: ignore[reportConstantRedefinition]

        data_type.__slots__ = frozenset(parameters.keys())  # pyright: ignore[reportAttributeAccessIssue]
        data_type.__match_args__ = data_type.__slots__  # pyright: ignore[reportAttributeAccessIssue]
        data_type._ = ParameterPath(data_type, data_type)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        return data_type


class ParametrizedData(metaclass=ParametrizedDataMeta):
    _: ClassVar[Self]
    __PARAMETERS__: ClassVar[dict[str, DataParameter]]
    __PARAMETERS_SPECIFICATION__: ClassVar[ParametersSpecification | Missing]

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        assert not args, "Positional unkeyed arguments are not supported"  # nosec: B101
        for parameter in self.__class__.__PARAMETERS__.values():
            parameter_context: ParameterValidationContext = (self.__class__.__qualname__,)
            object.__setattr__(
                self,  # pyright: ignore[reportUnknownArgumentType]
                parameter.name,
                parameter.validated(
                    kwargs.get(
                        parameter.name,
                        kwargs.get(
                            parameter.alias,
                            MISSING,
                        )
                        if parameter.alias
                        else MISSING,
                    ),
                    context=parameter_context,
                ),
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
            case self if self.__class__ == cls:
                return self

            case {**values}:
                return cls(**values)

            case self if cls == ParametrizedData and isinstance(self, ParametrizedData):
                # when the required type is the base ParametrizedData then allow any subclass
                return cast(Self, self)

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
        if parameters:
            return self.__class__(**{**vars(self), **parameters})

        else:
            return self

    def __str__(self) -> str:
        return str(self.as_dict())

    def __repr__(self) -> str:
        return str(self.as_dict())

    def __eq__(self, other: Any) -> bool:
        if other.__class__ != self.__class__:
            return False

        return all(
            getattr(self, key, MISSING) == getattr(other, key, MISSING)
            for key in self.__class__.__PARAMETERS__.keys()
        )

    def __setattr__(
        self,
        __name: str,
        __value: Any,
    ) -> None:
        raise RuntimeError(f"{self.__class__.__qualname__} is frozen and can't be modified")

    def __delattr__(
        self,
        __name: str,
    ) -> None:
        raise RuntimeError(f"{self.__class__.__qualname__} is frozen and can't be modified")


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

        case parametrized_data if hasattr(parametrized_data.__class__, "__PARAMETERS__"):
            # convert parametrized data to dict
            if aliased:
                return {
                    field.alias or field.name: _data_dict(
                        getattr(parametrized_data, field.name),
                        aliased=aliased,
                        converter=field.converter,
                    )
                    for field in parametrized_data.__class__.__PARAMETERS__.values()
                }

            else:
                return {
                    field.name: _data_dict(
                        getattr(parametrized_data, field.name),
                        aliased=aliased,
                        converter=field.converter,
                    )
                    for field in parametrized_data.__class__.__PARAMETERS__.values()
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
