import sys
from collections.abc import Callable
from inspect import Parameter, signature
from inspect import _empty as INSPECT_EMPTY  # pyright: ignore[reportPrivateUsage]
from typing import Any, Protocol, cast, final, overload

from draive.helpers import mimic_function
from draive.parameters.definition import ParameterDefinition, ParametersDefinition
from draive.parameters.missing import MISSING_PARAMETER, MissingParameter
from draive.parameters.specification import ParameterSpecification
from draive.parameters.validation import parameter_validator

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
    default: Value | MissingParameter = MISSING_PARAMETER,
    validator: Callable[[Any], Value] | None = None,
    specification: ParameterSpecification | None = None,
) -> Value: ...


@overload
def Argument[Value](
    *,
    alias: str | None = None,
    description: str | None = None,
    default_factory: Callable[[], Value] | None = None,
    validator: Callable[[Any], Value] | None = None,
    specification: ParameterSpecification | None = None,
) -> Value: ...


@overload
def Argument[Value](
    *,
    alias: str | None = None,
    description: str | None = None,
    default: Value | MissingParameter = MISSING_PARAMETER,
    verifier: Callable[[Value], None] | None = None,
    specification: ParameterSpecification | None = None,
) -> Value: ...


@overload
def Argument[Value](
    *,
    alias: str | None = None,
    description: str | None = None,
    default_factory: Callable[[], Value] | None = None,
    verifier: Callable[[Value], None] | None = None,
    specification: ParameterSpecification | None = None,
) -> Value: ...


def Argument[Value](  # noqa: PLR0913 # Ruff - noqa: B008
    *,
    alias: str | None = None,
    description: str | None = None,
    default: Value | MissingParameter = MISSING_PARAMETER,
    default_factory: Callable[[], Value] | None = None,
    validator: Callable[[Any], Value] | None = None,
    verifier: Callable[[Value], None] | None = None,
    specification: ParameterSpecification | None = None,
) -> Value:  # it is actually a FunctionParameter, but type checker has to be fooled
    assert (  # nosec: B101
        default_factory is None or default is MISSING_PARAMETER
    ), "Can't specify both default value and factory"
    assert (  # nosec: B101
        validator is None or verifier is None
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
        default: Value | MissingParameter,
        default_factory: Callable[[], Value] | None,
        validator: Callable[[Any], Value] | None,
        verifier: Callable[[Value], None] | None,
        specification: ParameterSpecification | None,
    ) -> None:
        assert (  # nosec: B101
            validator is None or verifier is None
        ), "Can't specify both validator and verifier"
        self.alias: str | None = alias
        self.description: str | None = description
        self.default: Value | MissingParameter = default
        self.default_factory: Callable[[], Value] | None = default_factory
        self.validator: Callable[[Any], Value] | None = validator
        self.verifier: Callable[[Value], None] | None = verifier
        self.specification: ParameterSpecification | None = specification


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
            or parameter_validator(
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
            default=MISSING_PARAMETER if parameter.default is INSPECT_EMPTY else parameter.default,
            default_factory=None,
            validator=parameter_validator(
                parameter.annotation,
                verifier=None,
                globalns=globalns,
                localns=localns,
                recursion_guard=frozenset(),
            ),
            specification=None,
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
            self.parameters: ParametersDefinition = ParametersDefinition(
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

    def validated_arguments(
        self,
        **arguments: Any,
    ) -> dict[str, Any]:
        return self.parameters.validated(**arguments)

    def __call__(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        assert not args, "Positional unkeyed arguments are not supported"  # nosec: B101
        return self._call(*args, **self.validated_arguments(**kwargs))  # pyright: ignore[reportCallIssue]
