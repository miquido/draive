from collections.abc import Callable, Iterable, Mapping
from inspect import Parameter as InspectParameter
from inspect import _empty as INSPECT_EMPTY  # pyright: ignore[reportPrivateUsage]
from inspect import signature
from typing import Any, get_type_hints

from haiway import (
    MISSING,
    AttributeAnnotation,
    DefaultValue,
    ValidationContext,
)
from haiway.attributes import Attribute
from haiway.attributes.annotations import resolve_attribute
from haiway.attributes.specification import type_specification
from haiway.utils import mimic_function

__all__ = ("ParametrizedFunction",)


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
        self._parameters: dict[str, Attribute]
        object.__setattr__(
            self,
            "_parameters",
            {},
        )
        self._variadic_keyword_parameters: Attribute | None
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
    def arguments(self) -> Iterable[Attribute]:
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
) -> Attribute:
    if parameter.annotation is INSPECT_EMPTY or type_hint is None:
        raise TypeError(f"Untyped argument {parameter.name}")

    attribute: AttributeAnnotation = resolve_attribute(
        type_hint,
        module=module,
        resolved_parameters={},
        recursion_guard={},
    )

    if isinstance(parameter.default, DefaultValue):
        return Attribute(
            name=parameter.name,
            annotation=attribute,
            alias=attribute.alias,
            default=parameter.default,
            specification=type_specification(attribute, None),
            required=False,
        )

    else:
        return Attribute(
            name=parameter.name,
            annotation=attribute,
            alias=attribute.alias,
            default=DefaultValue(
                default=MISSING if parameter.default is INSPECT_EMPTY else parameter.default,
            ),
            specification=type_specification(attribute, None),
            required=parameter.default is INSPECT_EMPTY and attribute.required,
        )
