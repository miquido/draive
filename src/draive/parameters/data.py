import sys
from collections.abc import Callable
from copy import deepcopy
from dataclasses import MISSING as DATACLASS_MISSING
from dataclasses import Field as DataclassField
from dataclasses import dataclass, is_dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields as dataclass_fields
from typing import Any, ClassVar, Self, cast, dataclass_transform, overload

from draive.parameters.basic import BasicValue
from draive.parameters.definition import ParameterDefinition, ParametersDefinition
from draive.parameters.path import ParameterPath
from draive.parameters.specification import ParameterSpecification
from draive.parameters.validation import (
    ParameterValidationContext,
    ParameterValidationError,
    ParameterValidator,
    ParameterVerifier,
    parameter_validator,
)
from draive.utils import MISSING, Missing, missing, not_missing

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
) -> Value:  # it is actually a dataclass.Field, but type checker has to be fooled
    assert (  # nosec: B101
        missing(default_factory) or missing(default)
    ), "Can't specify both default value and factory"
    assert (  # nosec: B101
        missing(validator) or verifier is None
    ), "Can't specify both validator and verifier"

    metadata: dict[str, Any] = {}
    if alias is not None:
        metadata["alias"] = alias
    if not_missing(description):
        metadata["description"] = description
    if not_missing(validator):
        metadata["validator"] = validator
    if verifier is not None:
        metadata["verifier"] = verifier
    if not_missing(converter):
        metadata["converter"] = converter
    if not_missing(specification):
        metadata["specification"] = specification

    if not_missing(default_factory):
        return dataclass_field(
            default_factory=default_factory,
            metadata=metadata,
        )

    elif not_missing(default):
        return dataclass_field(
            default=default,
            metadata=metadata,
        )

    else:
        return dataclass_field(metadata=metadata)


@dataclass_transform(
    kw_only_default=True,
    frozen_default=True,
    field_specifiers=(
        DataclassField,
        dataclass_field,
    ),
)
class ParametrizedDataMeta(type):
    __PARAMETERS__: ParametersDefinition

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        classdict: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        data_class: Any = dataclass(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
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
        if bases:
            globalns: dict[str, Any] = sys.modules.get(data_class.__module__).__dict__
            localns: dict[str, Any] = {data_class.__name__: data_class}
            recursion_guard: frozenset[type[Any]] = frozenset({data_class})
            data_class.__PARAMETERS__ = ParametersDefinition(
                data_class,
                (
                    _field_parameter(
                        field,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=recursion_guard,
                    )
                    for field in dataclass_fields(data_class)
                ),
            )

        else:
            data_class.__PARAMETERS__ = ParametersDefinition(
                data_class,
                parameters=[],
            )

        data_class._ = ParameterPath(data_class, data_class)

        return data_class


def _field_parameter(
    field: DataclassField[Any],
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
) -> ParameterDefinition[Any]:
    return ParameterDefinition(
        name=field.name,
        alias=field.metadata.get("alias", None),
        description=field.metadata.get("description", None),
        annotation=field.type,
        default=MISSING if field.default is DATACLASS_MISSING else field.default,
        default_factory=MISSING
        if field.default_factory is DATACLASS_MISSING
        else field.default_factory,
        validator=field.metadata.get(
            "validator",
            parameter_validator(
                field.type,
                verifier=field.metadata.get("verifier", None),
                globalns=globalns,
                localns=localns,
                recursion_guard=recursion_guard,
            ),
        ),
        specification=field.metadata.get("specification", MISSING),
    )


class ParametrizedData(metaclass=ParametrizedDataMeta):
    _: ClassVar[Self]

    # TODO: add validation on __init__

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
    def validated(
        cls,
        **values: Any,
    ) -> Self:
        return cls(
            **cls.__PARAMETERS__.validated(
                context=(cls.__qualname__,),
                **values,
            )
        )

    @classmethod
    def validator(
        cls,
        /,
        value: Any,
        context: ParameterValidationContext,
    ) -> Self:
        match value:
            case self if type(self) == cls:
                return self

            case {**values}:
                return cls(
                    **cls.__PARAMETERS__.validated(
                        context=context,
                        **values,
                    )
                )

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
        return cls.validated(**value)

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
            return self.__class__.validated(**{**vars(self), **parameters})

        else:
            return self


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
                    field.metadata.get("alias", field.name): _data_dict(
                        getattr(parametrized_data, field.name),
                        aliased=aliased,
                        converter=field.metadata.get("converter", None),
                    )
                    for field in dataclass_fields(parametrized_data)
                }

            else:
                return {
                    field.name: _data_dict(
                        getattr(parametrized_data, field.name),
                        aliased=aliased,
                        converter=field.metadata.get("converter", None),
                    )
                    for field in dataclass_fields(parametrized_data)
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
