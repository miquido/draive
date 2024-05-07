import sys
from collections.abc import Callable
from dataclasses import MISSING as DATACLASS_MISSING
from dataclasses import Field as DataclassField
from dataclasses import asdict as dataclass_asdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields as dataclass_fields
from typing import Any, ClassVar, Self, cast, dataclass_transform, overload

from draive.parameters.definition import ParameterDefinition, ParametersDefinition
from draive.parameters.missing import MISSING_PARAMETER, MissingParameter
from draive.parameters.path import ParameterPath
from draive.parameters.specification import ParameterSpecification
from draive.parameters.validation import parameter_validator

__all__ = [
    "Field",
    "ParametrizedData",
]


@overload
def Field[Value](
    *,
    alias: str | None = None,
    description: str | None = None,
    default: Value | MissingParameter = MISSING_PARAMETER,
    validator: Callable[[Any], Value] | None = None,
    specification: ParameterSpecification | None = None,
) -> Value: ...


@overload
def Field[Value](
    *,
    alias: str | None = None,
    description: str | None = None,
    default_factory: Callable[[], Value] | None = None,
    validator: Callable[[Any], Value] | None = None,
    specification: ParameterSpecification | None = None,
) -> Value: ...


@overload
def Field[Value](
    *,
    alias: str | None = None,
    description: str | None = None,
    default: Value | MissingParameter = MISSING_PARAMETER,
    verifier: Callable[[Value], None] | None = None,
    specification: ParameterSpecification | None = None,
) -> Value: ...


@overload
def Field[Value](
    *,
    alias: str | None = None,
    description: str | None = None,
    default_factory: Callable[[], Value] | None = None,
    verifier: Callable[[Value], None] | None = None,
    specification: ParameterSpecification | None = None,
) -> Value: ...


def Field[Value](  # noqa: PLR0913 # Ruff - noqa: B008
    *,
    alias: str | None = None,
    description: str | None = None,
    default: Value | MissingParameter = MISSING_PARAMETER,
    default_factory: Callable[[], Value] | None = None,
    validator: Callable[[Any], Value] | None = None,
    verifier: Callable[[Value], None] | None = None,
    specification: ParameterSpecification | None = None,
) -> Value:  # it is actually a dataclass.Field, but type checker has to be fooled
    assert (  # nosec: B101
        default_factory is None or default is MISSING_PARAMETER
    ), "Can't specify both default value and factory"
    assert (  # nosec: B101
        validator is None or verifier is None
    ), "Can't specify both validator and verifier"
    metadata: dict[str, Any] = {
        "alias": alias,
        "description": description,
        "validator": validator,
        "verifier": verifier,
        "specification": specification,
    }

    if default_factory := default_factory:
        return dataclass_field(
            default_factory=default_factory,
            metadata=metadata,
        )
    elif default is MISSING_PARAMETER:
        return cast(Value, dataclass_field(metadata=metadata))
    else:
        return cast(
            Value,
            dataclass_field(
                default=default,
                metadata=metadata,
            ),
        )


@dataclass_transform(
    kw_only_default=True,
    frozen_default=True,
    field_specifiers=(
        DataclassField,
        dataclass_field,
    ),
)
class ParametrizedDataMeta(type):
    __parameters_definition__: ParametersDefinition

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        classdict: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        if bases:
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
            globalns: dict[str, Any] = sys.modules.get(data_class.__module__).__dict__
            localns: dict[str, Any] = {data_class.__name__: data_class}
            recursion_guard: frozenset[type[Any]] = frozenset({data_class})
            data_class.__parameters_definition__ = ParametersDefinition(
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

            return data_class

        else:
            return type.__new__(
                cls,
                name,
                bases,
                classdict,
                **kwargs,
            )

    def _validated(
        cls,
        **parameters: Any,
    ) -> dict[str, Any]:
        return cls.__parameters_definition__.validated(**parameters)

    def _aliased(
        cls,
        **parameters: Any,
    ) -> dict[str, Any]:
        return cls.__parameters_definition__.aliased(**parameters)


def _field_parameter(
    field: DataclassField[Any],
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
) -> ParameterDefinition[Any]:
    return ParameterDefinition(
        name=field.name,
        alias=field.metadata.get("alias"),
        description=field.metadata.get("description"),
        annotation=field.type,
        default=MISSING_PARAMETER if field.default is DATACLASS_MISSING else field.default,
        default_factory=None
        if field.default_factory is DATACLASS_MISSING
        else field.default_factory,
        validator=field.metadata.get("validator")
        or parameter_validator(
            field.type,
            verifier=field.metadata.get("verifier"),
            globalns=globalns,
            localns=localns,
            recursion_guard=recursion_guard,
        ),
        specification=field.metadata.get("specification"),
    )


class ParametrizedData(metaclass=ParametrizedDataMeta):
    _: ClassVar[Self]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._: Self = cast(
            Self,
            ParameterPath(cls, cls),  # type: ignore
        )

    @classmethod
    def path(
        cls,
        /,
    ) -> Self:
        return cast(
            Self,
            ParameterPath(cls, cls),  # type: ignore
        )

    @classmethod
    def path_cast[Parameter](
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
        return cls(**cls._validated(**values))  # pyright: ignore[reportPrivateUsage]

    @classmethod
    def validator(
        cls,
        /,
        value: Any,
    ) -> Self:
        if isinstance(value, cls):
            return value

        elif isinstance(value, dict):
            return cls.validated(**value)

        else:
            raise TypeError("Invalid value %s", value)

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
        if aliased:
            return self.__class__._aliased(**dataclass_asdict(self))  # pyright: ignore[reportPrivateUsage]
        else:
            return dataclass_asdict(self)

    # TODO: find a way to generate signature similar to dataclass __init__
    def updated(
        self,
        /,
        **parameters: Any,
    ) -> Self:
        if parameters:
            return self.__class__.validated(**{**vars(self), **parameters})

        else:
            return self
