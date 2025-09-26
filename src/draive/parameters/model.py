import json
from collections.abc import Callable, Iterator, Mapping, MutableSequence, Sequence
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
    overload,
)
from weakref import WeakValueDictionary

from haiway import (
    MISSING,
    AttributeAnnotation,
    AttributePath,
    DefaultValue,
    Missing,
    ValidationContext,
    ValidationError,
    Validator,
    not_missing,
)
from haiway.attributes.annotations import ObjectAttribute, resolve_self_attribute

from draive.parameters.coding import ParametersJSONEncoder
from draive.parameters.parameter import Parameter
from draive.parameters.schema import json_schema, simplified_schema
from draive.parameters.specification import (
    ParameterSpecification,
    ParametersSpecification,
)
from draive.parameters.types import ParameterConversion, ParameterVerification

__all__ = (
    "DataModel",
    "Field",
)


@overload
def Field[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default: Value | Missing = MISSING,
    validator: Validator[Value] | Missing = MISSING,
    converter: ParameterConversion[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Field[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default_factory: Callable[[], Value] | Missing = MISSING,
    validator: Validator[Value] | Missing = MISSING,
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
    default_factory: Callable[[], Value] | Missing = MISSING,
    verifier: ParameterVerification[Value] | Missing = MISSING,
    converter: ParameterConversion[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Field[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default_env: str | Missing = MISSING,
    verifier: ParameterVerification[Value] | Missing = MISSING,
    converter: ParameterConversion[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


def Field[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default: Value | Missing = MISSING,
    default_factory: Callable[[], Value] | Missing = MISSING,
    default_env: str | Missing = MISSING,
    validator: Validator[Value] | Missing = MISSING,
    verifier: ParameterVerification[Value] | Missing = MISSING,
    converter: ParameterConversion[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value:  # it is actually a DataField, but type checker has to be fooled
    assert (  # nosec: B101
        default is MISSING or default_factory is MISSING or default_env is MISSING
    ), "Can't specify default value, factory and default_env"
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
            default_value=default,
            default_factory=default_factory,
            default_env=default_env,
            validator=validator,
            verifier=verifier,
            converter=converter,
            specification=specification,
        ),
    )


@final
class DataField:
    def __init__(
        self,
        aliased: str | None,
        description: str | Missing,
        default_value: Any | Missing,
        default_factory: Callable[[], Any] | Missing,
        default_env: str | Missing,
        validator: Validator[Any] | Missing,
        verifier: ParameterVerification[Any] | Missing,
        converter: ParameterConversion[Any] | Missing,
        specification: ParameterSpecification | Missing,
    ) -> None:
        self.aliased: str | None = aliased
        self.description: str | Missing = description
        self.default_value: Any | Missing = default_value
        self.default_factory: Callable[[], Any] | Missing = default_factory
        self.default_env: str | Missing = default_env
        self.validator: Validator[Any] | Missing = validator
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
            has_explicit_default: bool = (
                data_field.default_value is not MISSING
                or data_field.default_factory is not MISSING
                or data_field.default_env is not MISSING
            )

            return Parameter[Any].of(
                attribute,
                name=name,
                alias=data_field.aliased,
                description=data_field.description,
                default=DefaultValue(
                    data_field.default_value,
                    factory=data_field.default_factory,
                    env=data_field.default_env,
                ),
                validator=data_field.validator,
                verifier=data_field.verifier,
                converter=data_field.converter,
                specification=data_field.specification,
                required=not has_explicit_default,
            )

        case DefaultValue() as default:  # pyright: ignore[reportUnknownVariableType]
            return Parameter[Any].of(
                attribute,
                name=name,
                alias=None,
                description=MISSING,
                default=default,  # pyright: ignore[reportUnknownArgumentType]
                validator=MISSING,
                verifier=MISSING,
                converter=MISSING,
                specification=MISSING,
                required=False,
            )

        case value if value is MISSING:
            return Parameter[Any].of(
                attribute,
                name=name,
                alias=None,
                description=MISSING,
                default=DefaultValue(value),
                validator=MISSING,
                verifier=MISSING,
                converter=MISSING,
                specification=MISSING,
                required=True,
            )

        case value:
            return Parameter[Any].of(
                attribute,
                name=name,
                alias=None,
                description=MISSING,
                default=DefaultValue(value),
                validator=MISSING,
                verifier=MISSING,
                converter=MISSING,
                specification=MISSING,
                required=False,
            )


@dataclass_transform(
    kw_only_default=True,
    frozen_default=True,
    field_specifiers=(Field,),
)
class DataModelMeta(type):
    def __new__(
        mcs,
        /,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        type_parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        cls = type.__new__(
            mcs,
            name,
            bases,
            namespace,
            **kwargs,
        )
        self_attribute: ObjectAttribute = resolve_self_attribute(
            cls,
            parameters=type_parameters or {},
        )

        parameters: MutableSequence[Parameter[Any]] = []
        parameters_specification: dict[str, ParameterSpecification] = {}
        parameters_specification_required: list[str] = []
        for key, attribute in self_attribute.attributes.items():
            parameter: Parameter[Any] = _resolve_field(
                attribute,
                name=key,
                default=getattr(cls, key, MISSING),
            )
            # we are using aliased name for specification
            aliased_name: str = parameter.alias or parameter.name
            parameters_specification[aliased_name] = parameter.specification

            # and actual key/name for object itself
            parameters.append(parameter)

            if parameter.required:
                parameters_specification_required.append(aliased_name)

        cls.__SELF_ATTRIBUTE__ = self_attribute  # pyright: ignore[reportAttributeAccessIssue]

        if not bases:
            assert len(parameters) == 0  # nosec: B101
            cls.__TYPE_PARAMETERS__ = None  # pyright: ignore[reportAttributeAccessIssue]
            cls.__FIELDS__ = ()  # pyright: ignore[reportAttributeAccessIssue]

        else:
            cls.__TYPE_PARAMETERS__ = type_parameters  # pyright: ignore[reportAttributeAccessIssue]
            cls.__FIELDS__ = tuple(parameters)  # pyright: ignore[reportAttributeAccessIssue]
            cls.__slots__ = tuple(parameter.name for parameter in parameters)  # pyright: ignore[reportAttributeAccessIssue]
            cls.__match_args__ = cls.__slots__  # pyright: ignore[reportAttributeAccessIssue]

        if parameters_specification:
            cls.__PARAMETERS_SPECIFICATION__ = {  # pyright: ignore[reportAttributeAccessIssue]
                "type": "object",
                "properties": parameters_specification,
                "required": parameters_specification_required,
                "additionalProperties": False,
            }

        else:
            cls.__PARAMETERS_SPECIFICATION__ = {  # pyright: ignore[reportAttributeAccessIssue]
                "type": "object",
                "additionalProperties": True,
            }

        cls._ = AttributePath(cls, attribute=cls)  # pyright: ignore[reportCallIssue, reportUnknownMemberType, reportAttributeAccessIssue]

        return cls

    def validate(
        cls,
        value: Any,
    ) -> Any: ...

    def __instancecheck__(
        self,
        instance: Any,
    ) -> bool:
        instance_type: type[Any] = type(instance)
        if not self.__subclasscheck__(instance_type):
            return False

        if hasattr(self, "__origin__") or hasattr(instance_type, "__origin__"):
            try:  # TODO: find a better way to validate partially typed instances
                self(**vars(instance))

            except ValidationError:
                return False

        return True

    def __subclasscheck__(
        self,
        subclass: type[Any],
    ) -> bool:
        if self is subclass:
            return True

        self_origin: type[Any] = getattr(self, "__origin__", self)

        # Handle case where we're checking not parameterized type
        if self_origin is self:
            return type.__subclasscheck__(self, subclass)

        subclass_origin: type[Any] = getattr(subclass, "__origin__", subclass)

        # Both must be based on the same generic class
        if self_origin is not subclass_origin:
            return False

        return self._check_type_parameters(subclass)

    def _check_type_parameters(
        self,
        subclass: type[Any],
    ) -> bool:
        self_args: Sequence[Any] | None = getattr(
            self,
            "__args__",
            None,
        )
        subclass_args: Sequence[Any] | None = getattr(
            subclass,
            "__args__",
            None,
        )

        if self_args is None and subclass_args is None:
            return True

        if self_args is None:
            assert subclass_args is not None  # nosec: B101
            self_args = tuple(Any for _ in subclass_args)

        elif subclass_args is None:
            assert self_args is not None  # nosec: B101
            subclass_args = tuple(Any for _ in self_args)

        # Check if the type parameters are compatible (covariant)
        for self_arg, subclass_arg in zip(
            self_args,
            subclass_args,
            strict=True,
        ):
            if self_arg is Any or subclass_arg is Any:
                continue

            # For covariance: GenericState[Child] should be subclass of GenericState[Parent]
            # This means subclass_param should be a subclass of self_param
            if not issubclass(subclass_arg, self_arg):
                return False

        return True


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
    __TYPE_PARAMETERS__: ClassVar[Mapping[str, Any] | None] = None
    __SELF_ATTRIBUTE__: ClassVar[ObjectAttribute]
    __FIELDS__: ClassVar[Sequence[Parameter[Any]]]
    __PARAMETERS_SPECIFICATION__: ClassVar[ParametersSpecification]

    @classmethod
    def __class_getitem__(
        cls,
        type_argument: tuple[type[Any], ...] | type[Any],
    ) -> type[Self]:
        assert Generic in cls.__bases__, "Can't specialize non generic type!"  # nosec: B101
        assert cls.__TYPE_PARAMETERS__ is None, "Can't specialize already specialized type!"  # nosec: B101

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
        # Set origin for subclass checks
        parametrized_type.__origin__ = cls  # pyright: ignore[reportAttributeAccessIssue]
        parametrized_type.__args__ = type_arguments  # pyright: ignore[reportAttributeAccessIssue]
        _types_cache[(cls, type_arguments)] = parametrized_type
        return parametrized_type

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        for parameter in self.__FIELDS__:
            with ValidationContext.scope(f".{parameter.name}"):
                object.__setattr__(
                    self,  # pyright: ignore[reportUnknownArgumentType]
                    parameter.name,
                    parameter.validate(parameter.find(kwargs)),
                )

        if not getattr(self, "__slots__", ()):  # if we do not have slots accept everything
            assert not self.__FIELDS__  # nosec: B101
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
    ) -> Self:
        match value:
            case valid if isinstance(valid, cls):
                return valid

            case {**values}:
                return cls(**values)

            case _:
                raise TypeError(f"'{value}' is not matching expected type of '{cls}'")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        /,
    ) -> Self:
        return cls(**value)

    def to_str(self) -> str:
        return self.__str__()

    def to_mapping(
        self,
        aliased: bool = True,
    ) -> Mapping[str, Any]:
        return _data_dict(
            self,
            aliased=aliased,
            converter=None,
        )

    def updated(
        self,
        **kwargs: Any,
    ) -> Self:
        return self.__replace__(**kwargs)

    def __replace__(
        self,
        **kwargs: Any,
    ) -> Self:
        if not kwargs or kwargs.keys().isdisjoint(getattr(self, "__slots__", ())):
            return self  # do not make a copy when nothing will be updated

        if not kwargs:
            return self

        updated: Self = object.__new__(self.__class__)
        for field in self.__class__.__FIELDS__:
            update: Any | Missing = kwargs.get(field.name, MISSING)
            if update is MISSING:  # reuse missing elements
                object.__setattr__(
                    updated,
                    field.name,
                    getattr(self, field.name),
                )

            else:  # and validate updates
                with ValidationContext.scope(f".{field.name}"):
                    object.__setattr__(
                        updated,
                        field.name,
                        field.validate(update),
                    )

        return updated

    def __str__(self) -> str:
        return _data_str(
            self,
            aliased=True,
            converter=None,
        ).strip()

    def __repr__(self) -> str:
        return str(self.to_mapping())

    def __eq__(self, other: Any) -> bool:
        if other.__class__ != self.__class__:
            return False

        if self.__FIELDS__ is MISSING:
            return all(value == getattr(other, key, MISSING) for key, value in vars(self).items())

        else:
            return all(
                getattr(self, field.name, MISSING) == getattr(other, field.name, MISSING)
                for field in self.__FIELDS__
            )

    def __hash__(self) -> int:
        hash_values: list[int] = []
        for field in self.__FIELDS__:
            value: Any = getattr(self, field.name, MISSING)

            # Skip MISSING values to ensure consistent hashing
            if value is MISSING:
                continue

            # Convert to hashable representation
            try:
                hash_values.append(hash(value))

            except TypeError:
                continue  # skip unhashable

        return hash((self.__class__, tuple(hash_values)))

    def __contains__(
        self,
        element: Any,
    ) -> bool:
        return element in vars(self)

    def __getitem__(
        self,
        name: str,
    ) -> Any | Missing:
        return getattr(self, name, MISSING)

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
        return getattr(self, name, default)

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
        return self  # DataModel is immutable, no need to provide an actual copy

    def __deepcopy__(
        self,
        memo: dict[int, Any] | None,
    ) -> Self:
        return self  # DataModel is immutable, no need to provide an actual copy

    @classmethod
    def json_schema(
        cls,
        indent: int | None = None,
    ) -> str:
        return json_schema(
            cls.__PARAMETERS_SPECIFICATION__,
            indent=indent,
        )

    @classmethod
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
            raise ValueError(f"Failed to decode {cls.__name__} from json: {exc}") from exc

    @classmethod
    def from_json_array(
        cls,
        value: str | bytes,
        /,
        decoder: type[json.JSONDecoder] = json.JSONDecoder,
    ) -> Sequence[Self]:
        payload: Any
        try:
            payload = json.loads(
                value,
                cls=decoder,
            )

        except Exception as exc:
            raise ValueError(f"Failed to decode {cls.__name__} from json: {exc}") from exc

        match payload:
            case [*elements]:
                try:
                    return tuple(cls(**element) for element in elements)

                except Exception as exc:
                    raise ValueError(
                        f"Failed to decode {cls.__name__} from json array: {exc}"
                    ) from exc

            case _:
                raise ValueError("Provided json is not an array!")

    def to_json(
        self,
        aliased: bool = True,
        indent: int | None = None,
        encoder_class: type[json.JSONEncoder] = ParametersJSONEncoder,
    ) -> str:
        try:
            return json.dumps(
                self.to_mapping(aliased=aliased),
                indent=indent,
                cls=encoder_class,
            )

        except Exception as exc:
            raise ValueError(
                f"Failed to encode {self.__class__.__name__} to json:\n{self.to_mapping()}"
            ) from exc


# based on python dataclass asdict but simplified
def _data_dict(  # noqa: PLR0911, C901
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

        case DataModel() as model:
            fields: Sequence[Parameter[Any]] = model.__class__.__FIELDS__

            field_names: set[str] = {field.name for field in fields}
            mapping: dict[str, Any] = {}

            for field in fields:
                value: Any = getattr(model, field.name, MISSING)
                if value is MISSING:
                    continue

                key: str = field.alias if aliased and field.alias else field.name
                mapping[key] = _data_dict(
                    value,
                    aliased=aliased,
                    converter=field.converter,
                )

            # include dynamically attached attributes (when __slots__ is not defined)
            for key, value in getattr(model, "__dict__", {}).items():
                if key in field_names or value is MISSING:
                    continue

                mapping[key] = _data_dict(
                    value,
                    aliased=aliased,
                    converter=None,
                )

            return mapping

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

        case DataModel() as model:
            fields: Sequence[Parameter[Any]] = model.__class__.__FIELDS__
            field_names: set[str] = {field.name for field in fields}

            lines: list[str] = []

            for field in fields:
                value: Any = getattr(model, field.name, MISSING)
                if value is MISSING:
                    continue

                key: str = field.alias if aliased and field.alias else field.name
                element: str = _data_str(
                    value,
                    aliased=aliased,
                    converter=field.converter,
                ).replace("\n", "\n  ")

                lines.append(f"\n{key}: {element}")

            for key, value in getattr(model, "__dict__", {}).items():
                if key in field_names or value is MISSING:
                    continue

                element = _data_str(
                    value,
                    aliased=aliased,
                    converter=None,
                ).replace("\n", "\n  ")

                lines.append(f"\n{key}: {element}")

            return "".join(lines)

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
                    getattr(dataclass, field.name),
                    aliased=aliased,
                    converter=None,
                ).replace("\n", "\n  ")

                string += f"\n{field.name}: {element}"

            return string

        case other:  # for other types use its str
            return str(other)
