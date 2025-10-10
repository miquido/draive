from collections.abc import Callable, Iterable, Mapping
from inspect import Parameter as InspectParameter
from inspect import _empty as INSPECT_EMPTY  # pyright: ignore[reportPrivateUsage]
from inspect import signature
from typing import Any, cast, final, get_type_hints

from haiway import (
    MISSING,
    AttributeAnnotation,
    DefaultValue,
    Missing,
    TypeSpecification,
    ValidationContext,
    not_missing,
)
from haiway.attributes import Attribute
from haiway.attributes.annotations import resolve_attribute
from haiway.attributes.specification import type_specification
from haiway.utils import mimic_function

__all__ = (
    "Argument",
    "ParametrizedFunction",
)


def Argument[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default: Value | Missing = MISSING,
    default_factory: Callable[[], Value] | Missing = MISSING,
    default_env: str | Missing = MISSING,
    specification: TypeSpecification | Missing = MISSING,
) -> Value:  # it is actually a FunctionArgument, but type checker has to be fooled
    assert (  # nosec: B101
        default is MISSING or default_factory is MISSING or default_env is MISSING
    ), "Can't specify default value, factory and env"
    assert (  # nosec: B101
        description is MISSING or specification is MISSING
    ), "Can't specify both description and specification"
    return cast(
        Value,
        FunctionArgument(
            aliased=aliased,
            description=description,
            default=default,
            default_factory=default_factory,
            default_env=default_env,
            specification=specification,
        ),
    )


@final
class FunctionArgument:
    def __init__(
        self,
        aliased: str | None,
        description: str | Missing,
        default: Any | Missing,
        default_factory: Callable[[], Any] | Missing,
        default_env: str | Missing,
        specification: TypeSpecification | Missing,
    ) -> None:
        self.aliased: str | None = aliased
        self.description: str | Missing = description
        self.default: Any | Missing = default
        self.default_factory: Callable[[], Any] | Missing = default_factory
        self.default_env: str | Missing = default_env
        self.specification: TypeSpecification | Missing = specification


class ParametrizedFunction[**Args, Result]:
    __slots__ = (
        "__defaults__",
        "__doc__",
        "__globals__",
        "__kwdefaults__",
        "__name__",
        "__qualname__",
        "__wrapped__",
        "_call",
        "_name",
        "_parameters",
        "_variadic_keyword_parameters",
        "_write",
    )

    def __init__(
        self,
        function: Callable[Args, Result],
        /,
    ) -> None:
        assert (  # nosec: B101
            not isinstance(function, ParametrizedFunction)
        ), "Cannot parametrize the same function more than once!"

        self._call: Callable[Args, Result]
        object.__setattr__(
            self,
            "_call",
            function,
        )
        self._name: str
        object.__setattr__(
            self,
            "_name",
            function.__name__,
        )
        self._parameters: dict[str, Attribute[Any]]
        object.__setattr__(
            self,
            "_parameters",
            {},
        )
        self._variadic_keyword_parameters: Attribute[Any] | None
        object.__setattr__(
            self,
            "_variadic_keyword_parameters",
            None,
        )
        type_hints: Mapping[str, Any] = get_type_hints(
            function,
            include_extras=True,
        )
        for parameter in signature(function).parameters.values():
            match parameter.kind:
                case InspectParameter.POSITIONAL_ONLY | InspectParameter.VAR_POSITIONAL:
                    raise NotImplementedError("Positional only arguments are not supported yet.")

                case InspectParameter.VAR_KEYWORD:
                    assert self._variadic_keyword_parameters is None  # nosec: B101
                    object.__setattr__(
                        self,
                        "_variadic_keyword_parameters",
                        _resolve_argument(
                            parameter,
                            module=function.__module__,
                            type_hint=type_hints.get(parameter.name),
                        ),
                    )

                case _:
                    self._parameters[parameter.name] = _resolve_argument(
                        parameter,
                        module=function.__module__,
                        type_hint=type_hints.get(parameter.name),
                    )

        mimic_function(function, within=self)

    @property
    def arguments(self) -> Iterable[Attribute[Any]]:
        return self._parameters.values()

    def validate_arguments(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # TODO: add support for positional arguments
        validated: dict[str, Any] = {}
        if self._variadic_keyword_parameters is None:
            for parameter in self._parameters.values():
                with ValidationContext.scope(f".{parameter.name}"):
                    validated[parameter.name] = parameter.validate_from(kwargs)

        else:
            for parameter in self._parameters.values():
                with ValidationContext.scope(f".{parameter.name}"):
                    validated[parameter.name] = parameter.validate_from(kwargs)

            for key, value in kwargs.items():
                with ValidationContext.scope(f".{key}"):
                    validated[key] = self._variadic_keyword_parameters.validate(
                        value,
                    )

        return validated

    def __call__(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        assert not args, "Positional unkeyed arguments are not supported"  # nosec: B101
        return self._call(*args, **self.validate_arguments(**kwargs))  # pyright: ignore[reportCallIssue]

    def __setattr__(
        self,
        name: str,
        value: Any,
    ) -> Any:
        raise AttributeError(
            f"Can't modify immutable {self.__class__.__qualname__},"
            f" attribute - '{name}' cannot be modified"
        )

    def __delattr__(
        self,
        name: str,
    ) -> None:
        raise AttributeError(
            f"Can't modify immutable {self.__class__.__qualname__},"
            f" attribute - '{name}' cannot be deleted"
        )


def _resolve_argument(
    parameter: InspectParameter,
    /,
    *,
    module: str,
    type_hint: Any,
) -> Attribute[Any]:
    if parameter.annotation is INSPECT_EMPTY or type_hint is None:
        raise TypeError(f"Untyped argument {parameter.name}")

    attribute: AttributeAnnotation = resolve_attribute(
        type_hint,
        module=module,
        resolved_parameters={},
        recursion_guard={},
    )

    match parameter.default:
        case FunctionArgument() as argument:
            alias: str | None = (
                argument.aliased
                if argument.aliased is not None
                else attribute.alias
            )
            specification: TypeSpecification | None = None

            if not_missing(argument.specification):
                specification = argument.specification

            else:
                description: str | None = None
                if not_missing(argument.description):
                    description = argument.description

                else:
                    description = attribute.description

                specification = type_specification(
                    attribute,
                    description=description,
                )

            return Attribute(
                name=parameter.name,
                annotation=attribute,
                alias=alias,
                default=DefaultValue(
                    argument.default,
                    factory=argument.default_factory,
                    env=argument.default_env,
                ),
                specification=specification,
                required=argument.default is MISSING
                and argument.default_factory is MISSING
                and argument.default_env is MISSING,
            )

        case DefaultValue() as default:
            return Attribute(
                name=parameter.name,
                annotation=attribute,
                alias=attribute.alias,
                default=default,
                specification=type_specification(attribute, None),
                required=False,
            )

        case value:
            return Attribute(
                name=parameter.name,
                annotation=attribute,
                alias=attribute.alias,
                default=DefaultValue(MISSING if value is INSPECT_EMPTY else value),
                specification=type_specification(attribute, None),
                required=value is INSPECT_EMPTY and attribute.required,
            )
