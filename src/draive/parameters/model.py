import json
import typing
from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from copy import deepcopy
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from types import GenericAlias
from typing import (
    Any,
    ClassVar,
    Generic,
    Self,
    TypeVar,
    cast,
    dataclass_transform,
    overload,
)
from weakref import WeakValueDictionary

from haiway import (
    MISSING,
    AttributeAnnotation,
    AttributePath,
    Default,
    DefaultValue,
    Missing,
    State,
    TypeSpecification,
    ValidationContext,
    ValidationError,
    not_missing,
)
from haiway.attributes import Attribute, AttributesJSONEncoder
from haiway.attributes.annotations import (
    ObjectAttribute,
    resolve_self_attribute,
)
from haiway.attributes.specification import type_specification

from draive.parameters.schema import simplified_schema

__all__ = ("DataModel",)


@dataclass_transform(
    kw_only_default=True,
    frozen_default=True,
    field_specifiers=(Default,),
)
class DataModelMeta(type):
    __SELF_ATTRIBUTE__: ObjectAttribute
    __TYPE_PARAMETERS__: Mapping[str, Any] | None
    __SPECIFICATION__: TypeSpecification
    __FIELDS__: Sequence[Attribute]

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

        specification_fields: MutableMapping[str, TypeSpecification] = {}
        required_fields: MutableSequence[str] = []
        fields: MutableSequence[Attribute] = []
        for key, element in self_attribute.attributes.items():
            default: Any = getattr(cls, key, MISSING)
            base_attribute: AttributeAnnotation = element
            attribute: AttributeAnnotation = base_attribute
            alias: str | None = attribute.alias
            description: str | None = attribute.description
            required: bool = attribute.required
            specification: TypeSpecification | None = attribute.specification

            if specification is None:
                specification = type_specification(
                    attribute,
                    description=description,
                )

            if specification is None:
                raise RuntimeError(
                    f"Attribute {key} of {name} does not provide a valid specification"
                )

            field: Attribute = Attribute(
                name=key,
                alias=alias,
                annotation=attribute,
                required=required,
                default=_resolve_default(default),
                specification=specification,
            )

            fields.append(field)

            if field.alias is not None:
                assert field.alias not in specification_fields  # nosec: B101
                assert field.name not in specification_fields  # nosec: B101
                specification_fields[field.alias] = specification
                if field.required:
                    required_fields.append(field.alias)

            else:
                assert field.name not in specification_fields  # nosec: B101
                specification_fields[field.name] = specification
                if field.required:
                    required_fields.append(field.name)

        cls.__SELF_ATTRIBUTE__ = self_attribute  # pyright: ignore[reportConstantRedefinition]
        cls.__TYPE_PARAMETERS__ = type_parameters  # pyright: ignore[reportConstantRedefinition]
        if not bases:
            assert not type_parameters  # nosec: B101
            cls.__FIELDS__ = ()  # pyright: ignore[reportAttributeAccessIssue, reportConstantRedefinition]
            cls.__SPECIFICATION__ = {  # pyright: ignore[reportConstantRedefinition]
                "type": "object",
                "additionalProperties": True,
            }

        else:
            cls.__FIELDS__ = tuple(fields)  # pyright: ignore[reportConstantRedefinition]
            cls.__SPECIFICATION__ = (  # pyright: ignore[reportAttributeAccessIssue, reportConstantRedefinition]
                {
                    "type": "object",
                    "properties": specification_fields,
                    "required": required_fields,
                    "additionalProperties": False,
                }
            )
            cls.__slots__ = tuple(field.name for field in fields)  # pyright: ignore[reportAttributeAccessIssue, reportConstantRedefinition]
            cls.__match_args__ = cls.__slots__  # pyright: ignore[reportAttributeAccessIssue, reportConstantRedefinition]

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
        instance_type: Any = cast(Any, type(instance))
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


def _resolve_default(
    value: DefaultValue | Any | Missing,
) -> DefaultValue:
    if isinstance(value, DefaultValue):
        return value

    return DefaultValue(
        default=value,
        default_factory=MISSING,
        env=MISSING,
    )


_types_cache: WeakValueDictionary[
    tuple[
        Any,
        tuple[Any, ...],
    ],
    Any,
] = WeakValueDictionary()


class DataModel(metaclass=DataModelMeta):
    _: ClassVar[Self]

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

    @classmethod
    def validate(
        cls,
        value: Any,
    ) -> Self:
        if isinstance(value, cls):
            return value

        elif isinstance(value, Mapping | typing.Mapping):
            return cls(**value)

        else:
            raise TypeError(f"'{value}' is not matching expected type of '{cls}'")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        /,
    ) -> Self:
        return cls(**value)

    @classmethod
    def simplified_schema(
        cls,
        indent: int | None = None,
    ) -> str:
        return simplified_schema(
            cls.__SPECIFICATION__,
            indent=indent,
        )

    @classmethod
    def json_schema(
        cls,
        *,
        indent: int | None = None,
    ) -> str:
        return json.dumps(
            cls.__SPECIFICATION__,
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
        indent: int | None = None,
        encoder_class: type[json.JSONEncoder] = AttributesJSONEncoder,
    ) -> str:
        mapping: Mapping[str, Any] = self.to_mapping()
        try:
            return json.dumps(
                mapping,
                indent=indent,
                cls=encoder_class,
            )

        except Exception as exc:
            raise ValueError(
                f"Failed to encode {self.__class__.__name__} to json:\n{mapping}"
            ) from exc

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        for parameter in self.__FIELDS__:
            with ValidationContext.scope(f".{parameter.name}"):
                object.__setattr__(
                    self,  # pyright: ignore[reportUnknownArgumentType]
                    parameter.name,
                    parameter.validate_from(kwargs),
                )

        if not getattr(self, "__slots__", ()):  # if we do not have slots accept everything
            assert not self.__FIELDS__  # nosec: B101
            for key, value in kwargs.items():
                object.__setattr__(
                    self,  # pyright: ignore[reportUnknownArgumentType]
                    key,
                    value,
                )

    def to_str(self) -> str:
        return self.__str__()

    def to_mapping(self) -> Mapping[str, Any]:
        dict_result: dict[str, Any] = {}
        if getattr(self.__class__, "__slots__", ()):
            for field in self.__FIELDS__:
                key: str = field.alias if field.alias is not None else field.name
                value: Any | Missing = getattr(self, field.name, MISSING)

                if not_missing(value):
                    dict_result[key] = _recursive_mapping(value)

        else:
            # include dynamically attached attributes (when __slots__ is not defined)
            for key, value in vars(self).items():
                if not_missing(value):
                    dict_result[key] = _recursive_mapping(value)

        return dict_result

    def updated(
        self,
        **kwargs: Any,
    ) -> Self:
        return self.__replace__(**kwargs)

    def __str__(self) -> str:
        return str(self.to_mapping())

    def __repr__(self) -> str:
        return str(self.to_mapping())

    def __eq__(self, other: Any) -> bool:
        if other.__class__ != self.__class__:
            return False

        if not self.__FIELDS__:
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

    def __replace__(
        self,
        **kwargs: Any,
    ) -> Self:
        if not kwargs:
            return self  # do not make a copy when nothing will be updated

        if not self.__class__.__FIELDS__:
            return self.__class__(**(vars(self) | kwargs))

        fields: Sequence[Attribute] = self.__class__.__FIELDS__
        alias_to_name: dict[str, str] = {
            field.alias if field.alias is not None else field.name: field.name for field in fields
        }
        valid_keys: set[str] = set(alias_to_name.keys()) | set(alias_to_name.values())

        if kwargs.keys().isdisjoint(valid_keys):
            return self  # do not make a copy when nothing will be updated

        canonical_updates: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in valid_keys:
                canonical_updates[alias_to_name.get(key, key)] = value

        if not canonical_updates:
            return self

        updated: Self = object.__new__(self.__class__)
        for field in fields:
            update: Any | Missing = canonical_updates.get(field.name, MISSING)
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


def _recursive_mapping(  # noqa: PLR0911
    value: Any,
) -> Any:
    if isinstance(value, str | bytes | float | int | bool | None):
        return value

    elif isinstance(value, State):
        return value.to_mapping(recursive=True)

    elif isinstance(value, DataModel):
        return value.to_mapping()

    elif is_dataclass(value):
        return {
            field.name: _recursive_mapping(getattr(value, field.name))
            for field in dataclass_fields(value)
        }

    elif isinstance(value, Mapping | typing.Mapping):
        return {
            key: _recursive_mapping(element)
            for key, element in cast(Mapping[Any, Any], value).items()
        }

    elif isinstance(value, Iterable | typing.Iterable):
        return [_recursive_mapping(element) for element in cast(Iterable[Any], value)]

    elif hasattr(value, "to_mapping") and callable(value.to_mapping):
        return value.to_mapping()

    else:
        return deepcopy(value)
