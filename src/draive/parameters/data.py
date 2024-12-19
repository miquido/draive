import json
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from datetime import date, datetime, time
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
    overload,
)
from uuid import UUID
from weakref import WeakValueDictionary

from haiway import MISSING, AttributePath, Missing, cache, not_missing
from haiway.state import AttributeAnnotation, attribute_annotations

from draive.parameters.parameter import Parameter
from draive.parameters.schema import json_schema, simplified_schema
from draive.parameters.specification import (
    ParameterSpecification,
    ParametersSpecification,
)
from draive.parameters.types import (
    ParameterConversion,
    ParameterDefaultFactory,
    ParameterValidation,
    ParameterValidationContext,
    ParameterVerification,
)

__all__ = [
    "DataModel",
    "Field",
]


@overload
def Field[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default: Value | Missing = MISSING,
    validator: ParameterValidation[Value] | Missing = MISSING,
    converter: ParameterConversion[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Field[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default_factory: ParameterDefaultFactory[Value] | Missing = MISSING,
    validator: ParameterValidation[Value] | Missing = MISSING,
    converter: ParameterConversion[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Field[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default: Value | Missing = MISSING,
    verifier: ParameterVerification[Value] | Missing = MISSING,
    converter: ParameterConversion[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Field[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default_factory: ParameterDefaultFactory[Value] | Missing = MISSING,
    verifier: ParameterVerification[Value] | Missing = MISSING,
    converter: ParameterConversion[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


def Field[Value](  # noqa: PLR0913
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default: Value | Missing = MISSING,
    default_factory: ParameterDefaultFactory[Value] | Missing = MISSING,
    validator: ParameterValidation[Value] | Missing = MISSING,
    verifier: ParameterVerification[Value] | Missing = MISSING,
    converter: ParameterConversion[Value] | Missing = MISSING,
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
        validator: ParameterValidation[Any] | Missing,
        verifier: ParameterVerification[Any] | Missing,
        converter: ParameterConversion[Any] | Missing,
        specification: ParameterSpecification | Missing,
    ) -> None:
        self.aliased: str | None = aliased
        self.description: str | Missing = description
        self.default: Any | Missing = default
        self.default_factory: Callable[[], Any] | Missing = default_factory
        self.validator: ParameterValidation[Any] | Missing = validator
        self.verifier: ParameterVerification[Any] | Missing = verifier
        self.converter: ParameterConversion[Any] | Missing = converter
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
                converter=data_field.converter,
                specification=data_field.specification,
                required=attribute.required
                and data_field.default is MISSING
                and data_field.default_factory is MISSING,
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
                required=attribute.required,
            )


@dataclass_transform(
    kw_only_default=True,
    frozen_default=True,
    field_specifiers=(Field,),
)
class DataModelMeta(type):
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
        parameters_specification_required: list[str] = []
        for key, annotation in attribute_annotations(
            data_type,
            type_parameters=type_parameters or {},
        ).items():
            default: Any = getattr(data_type, key, MISSING)
            updated_annotation = annotation.update_required(_check_required(default))
            parameter: Parameter[Any] = _resolve_field(
                updated_annotation,
                name=key,
                default=default,
            )
            parameters_specification[key] = parameter.specification

            if parameter.required:
                parameters_specification_required.append(key)

            parameters[key] = parameter

        if bases:
            data_type.__PARAMETERS__ = parameters  # pyright: ignore[reportAttributeAccessIssue]

        else:
            assert len(parameters) == 0  # nosec: B101
            data_type.__PARAMETERS__ = MISSING  # pyright: ignore[reportAttributeAccessIssue]

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
        data_type._ = AttributePath(data_type, attribute=data_type)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportCallIssue]

        return data_type


def _check_required(default: Any) -> bool:
    if default is MISSING:
        return True

    elif isinstance(default, DataField):
        return default.default is MISSING and default.default_factory is MISSING

    else:
        return False


_types_cache: WeakValueDictionary[
    tuple[
        Any,
        tuple[Any, ...],
    ],
    Any,
] = WeakValueDictionary()


class DataModel(metaclass=DataModelMeta):
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

        parametrized_type: type[Self] = DataModelMeta.__new__(
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
        if self.__PARAMETERS__ is MISSING:
            for key, value in kwargs.items():
                object.__setattr__(
                    self,  # pyright: ignore[reportUnknownArgumentType]
                    key,
                    value,
                )

        else:
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

    @classmethod
    def model_validator(
        cls,
        *,
        verifier: ParameterVerification[Self] | None,
    ) -> ParameterValidation[Self]:
        if verifier := verifier:

            def validator(
                value: Any,
                /,
                context: ParameterValidationContext,
            ) -> Self:
                match value:
                    case validated if isinstance(validated, cls):
                        verifier(validated)
                        return validated

                    case {**values}:
                        validated: Self = cls(**values)
                        verifier(validated)
                        return validated

                    case _:
                        raise TypeError(
                            f"Expected '{cls.__name__}', received '{type(value).__name__}'"
                        )
        else:

            def validator(
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
                        raise TypeError(
                            f"Expected '{cls.__name__}', received '{type(value).__name__}'"
                        )

        return validator

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

        if self.__PARAMETERS__ is MISSING:
            return all(value == getattr(other, key, MISSING) for key, value in vars(self).items())

        else:
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

    def __iter__(self) -> Iterator[str]:
        yield from vars(self)

    def __len__(self) -> int:
        return len(vars(self))

    def __copy__(self) -> Self:
        return self.__class__(**vars(self))

    @classmethod
    @cache(limit=2)
    def json_schema(
        cls,
        indent: int | None = None,
    ) -> str:
        return json_schema(
            cls.__PARAMETERS_SPECIFICATION__,
            indent=indent,
        )

    @classmethod
    @cache(limit=2)
    def simplified_schema(
        cls,
        indent: int | None = None,
    ) -> str:
        assert not_missing(  # nosec: B101
            cls.__PARAMETERS_SPECIFICATION__
        ), f"{cls.__qualname__} can't be represented using simplified schema"

        return simplified_schema(
            cls.__PARAMETERS_SPECIFICATION__,
            indent=indent,
        )

    @classmethod
    def from_json(
        cls,
        value: str | bytes,
        /,
        decoder: type[json.JSONDecoder] = json.JSONDecoder,
    ) -> Self:
        try:
            return cls(
                **json.loads(
                    value,
                    cls=decoder,
                )
            )

        except Exception as exc:
            raise ValueError(f"Failed to decode {cls.__name__} from json:\n{value}") from exc

    def as_json(
        self,
        aliased: bool = True,
        indent: int | None = None,
        encoder_class: type[json.JSONEncoder] | None = None,
    ) -> str:
        try:
            return json.dumps(
                self.as_dict(aliased=aliased),
                indent=indent,
                cls=encoder_class or ModelJSONEncoder,
            )

        except Exception as exc:
            raise ValueError(
                f"Failed to encode {self.__class__.__name__} to json:\n{self.as_dict()}"
            ) from exc


class ModelJSONEncoder(json.JSONEncoder):
    def default(self, o: object) -> Any:
        if isinstance(o, UUID):
            return o.hex

        elif isinstance(o, datetime):
            return o.isoformat()

        elif isinstance(o, time):
            return o.isoformat()

        elif isinstance(o, date):
            return o.isoformat()

        else:
            return json.JSONEncoder.default(self, o)


# based on python dataclass asdict but simplified
def _data_dict(  # noqa: PLR0911
    data: Any,
    /,
    aliased: bool,
    converter: ParameterConversion[Any] | None,
) -> Any:
    # use converter if able
    if converter := converter:
        return converter(data)

    match data:
        case str() | None | int() | float() | bool():
            return data  # use basic value types as they are

        case parametrized_data if hasattr(parametrized_data, "__PARAMETERS__"):
            # convert parametrized data to dict
            if parametrized_data.__PARAMETERS__ is MISSING:  # if base - use all variables
                return cast(
                    dict[str, Any],
                    {
                        key: value
                        for key, value in vars(parametrized_data).items()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                        if value is not MISSING
                    },
                )

            elif aliased:  # alias if needed
                return {
                    field.alias or field.name: _data_dict(
                        getattr(parametrized_data, field.name),
                        aliased=aliased,
                        converter=field.converter,
                    )
                    for field in parametrized_data.__PARAMETERS__.values()
                    if getattr(parametrized_data, field.name, MISSING) is not MISSING
                }

            else:
                return {
                    field.name: _data_dict(
                        getattr(parametrized_data, field.name),
                        aliased=aliased,
                        converter=field.converter,
                    )
                    for field in parametrized_data.__PARAMETERS__.values()
                    if getattr(parametrized_data, field.name, MISSING) is not MISSING
                }

        case {**elements}:  # replace mapping with dict
            return {
                key: _data_dict(
                    value,
                    aliased=aliased,
                    converter=None,
                )
                for key, value in elements.items()
                if value is not MISSING
            }

        case [*values]:  # replace sequence with list
            return [
                _data_dict(
                    value,
                    aliased=aliased,
                    converter=None,
                )
                for value in values
                if value is not MISSING
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
    converter: ParameterConversion[Any] | None,
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
            if parametrized_data.__PARAMETERS__ is MISSING:  # if base - use all variables
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

                    string += f"\n{field.alias or field.name}: {element}"

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
