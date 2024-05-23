import sys
from collections.abc import Callable
from inspect import Parameter, signature
from inspect import _empty as INSPECT_EMPTY  # pyright: ignore[reportPrivateUsage]
from typing import Any, Protocol, cast, final, overload

from draive.parameters.definition import ParameterDefinition, ParametersDefinition
from draive.parameters.specification import ParameterSpecification
from draive.parameters.validation import (
    ParameterValidator,
    ParameterVerifier,
    parameter_validator,
)
from draive.utils import MISSING, Missing, mimic_function, missing, not_missing

__all__ = [
    "Argument",
    "ParametrizedFunction",
]


class Function[**Args, Result](Protocol):
    @property
    def __name__(self) -> str: ...

    def __call__(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result: ...


@overload
def Argument[Value](
    *,
    alias: str | None = None,
    description: str | None = None,
    default: Value | Missing = MISSING,
    validator: ParameterValidator[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Argument[Value](
    *,
    alias: str | None = None,
    description: str | None = None,
    default_factory: Callable[[], Value] | Missing = MISSING,
    validator: ParameterValidator[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Argument[Value](
    *,
    alias: str | None = None,
    description: str | None = None,
    default: Value | Missing = MISSING,
    verifier: ParameterVerifier[Value] | None = None,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Argument[Value](
    *,
    alias: str | None = None,
    description: str | None = None,
    default_factory: Callable[[], Value] | Missing = MISSING,
    verifier: ParameterVerifier[Value] | None = None,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


def Argument[Value](  # noqa: PLR0913 # Ruff - noqa: B008
    *,
    alias: str | None = None,
    description: str | None = None,
    default: Value | Missing = MISSING,
    default_factory: Callable[[], Value] | Missing = MISSING,
    validator: ParameterValidator[Value] | Missing = MISSING,
    verifier: ParameterVerifier[Value] | None = None,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value:  # it is actually a FunctionArgument, but type checker has to be fooled
    assert (  # nosec: B101
        missing(default_factory) or missing(default)
    ), "Can't specify both default value and factory"
    assert (  # nosec: B101
        missing(validator) or verifier is None
    ), "Can't specify both validator and verifier"

    return cast(
        Value,
        FunctionArgument(
            alias=alias,
            description=description,
            default=default,
            default_factory=default_factory,
            validator=validator,
            verifier=verifier,
            specification=specification,
        ),
    )


@final
class FunctionArgument[Value]:
    def __init__(  # noqa: PLR0913
        self,
        alias: str | None,
        description: str | None,
        default: Value | Missing,
        default_factory: Callable[[], Value] | Missing,
        validator: ParameterValidator[Value] | Missing,
        verifier: ParameterVerifier[Value] | None,
        specification: ParameterSpecification | Missing,
    ) -> None:
        assert (  # nosec: B101
            missing(validator) or verifier is None
        ), "Can't specify both validator and verifier"

        self.alias: str | None = alias
        self.description: str | None = description
        self.default: Value | Missing = default
        self.default_factory: Callable[[], Value] | Missing = default_factory
        self.validator: ParameterValidator[Value] | Missing = validator
        self.verifier: ParameterVerifier[Value] | None = verifier
        self.specification: ParameterSpecification | Missing = specification


def _argument_parameter(
    parameter: Parameter,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
) -> ParameterDefinition[Any]:
    if parameter.annotation is INSPECT_EMPTY:
        raise TypeError(
            "Untyped argument %s",
            parameter.name,
        )

    if isinstance(parameter.default, FunctionArgument):
        argument: FunctionArgument[Any] = cast(FunctionArgument[Any], parameter.default)  # pyright: ignore[reportUnknownMemberType]
        return ParameterDefinition[Any](
            name=parameter.name,
            alias=argument.alias,
            description=argument.description,
            annotation=parameter.annotation,
            default=argument.default,
            default_factory=argument.default_factory,
            validator=argument.validator
            if not_missing(argument.validator)
            else parameter_validator(
                parameter.annotation,
                verifier=argument.verifier,
                globalns=globalns,
                localns=localns,
                recursion_guard=frozenset(),
            ),
            specification=argument.specification,
        )
    else:  # use regular type annotation
        return ParameterDefinition(
            name=parameter.name,
            alias=None,
            description=None,
            annotation=parameter.annotation,
            default=MISSING if parameter.default is INSPECT_EMPTY else parameter.default,
            default_factory=MISSING,
            validator=parameter_validator(
                parameter.annotation,
                verifier=None,
                globalns=globalns,
                localns=localns,
                recursion_guard=frozenset(),
            ),
            specification=MISSING,
        )


class ParametrizedFunction[**Args, Result]:
    def __init__(
        self,
        function: Function[Args, Result],
    ) -> None:
        if isinstance(function, ParametrizedFunction):
            self = function

        else:
            self._call: Function[Args, Result] = function
            globalns: dict[str, Any] = sys.modules.get(function.__module__).__dict__
            self._parameters: ParametersDefinition = ParametersDefinition(
                function.__module__,
                (
                    _argument_parameter(
                        parameter=parameter,
                        globalns=globalns,
                        localns=None,
                    )
                    for parameter in signature(function).parameters.values()
                ),
            )

            mimic_function(function, within=self)

    def validate_arguments(
        self,
        **arguments: Any,
    ) -> dict[str, Any]:
        return self._parameters.validated(
            context=(self.__qualname__,),
            **arguments,
        )

    def __call__(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        assert not args, "Positional unkeyed arguments are not supported"  # nosec: B101
        return self._call(*args, **self.validate_arguments(**kwargs))  # pyright: ignore[reportCallIssue]
