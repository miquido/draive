import sys
from collections.abc import Callable
from inspect import Parameter, signature
from inspect import _empty as INSPECT_EMPTY  # pyright: ignore[reportPrivateUsage]
from typing import Any, Self, cast, final, overload

from draive.parameters.annotations import ParameterDefaultFactory, allows_missing
from draive.parameters.specification import ParameterSpecification, parameter_specification
from draive.parameters.validation import (
    ParameterValidationContext,
    ParameterValidationError,
    ParameterValidator,
    ParameterVerifier,
    parameter_validator,
)
from draive.utils import MISSING, Missing, freeze, is_missing, mimic_function, not_missing

__all__ = [
    "Argument",
    "ParametrizedFunction",
]


@overload
def Argument[Value](
    *,
    aliased: str | None = None,
    description: str | None = None,
    default: Value | Missing = MISSING,
    validator: ParameterValidator[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Argument[Value](
    *,
    aliased: str | None = None,
    description: str | None = None,
    default_factory: ParameterDefaultFactory[Value] | Missing = MISSING,
    validator: ParameterValidator[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Argument[Value](
    *,
    aliased: str | None = None,
    description: str | None = None,
    default: Value | Missing = MISSING,
    verifier: ParameterVerifier[Value] | None = None,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Argument[Value](
    *,
    aliased: str | None = None,
    description: str | None = None,
    default_factory: ParameterDefaultFactory[Value] | Missing = MISSING,
    verifier: ParameterVerifier[Value] | None = None,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


def Argument[Value](  # noqa: PLR0913 # Ruff - noqa: B008
    *,
    aliased: str | None = None,
    description: str | None = None,
    default: Value | Missing = MISSING,
    default_factory: ParameterDefaultFactory[Value] | Missing = MISSING,
    validator: ParameterValidator[Value] | Missing = MISSING,
    verifier: ParameterVerifier[Value] | None = None,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value:  # it is actually a FunctionArgument, but type checker has to be fooled
    assert (  # nosec: B101
        is_missing(default_factory) or is_missing(default)
    ), "Can't specify both default value and factory"
    assert (  # nosec: B101
        is_missing(validator) or verifier is None
    ), "Can't specify both validator and verifier"

    return cast(
        Value,
        FunctionArgument(
            aliased=aliased,
            description=description,
            default=default,
            default_factory=default_factory,
            validator=validator,
            verifier=verifier,
            specification=specification,
        ),
    )


@final
class FunctionArgument:
    def __init__(  # noqa: PLR0913
        self,
        aliased: str | None,
        description: str | None,
        default: Any | Missing,
        default_factory: ParameterDefaultFactory[Any] | Missing,
        validator: ParameterValidator[Any] | Missing,
        verifier: ParameterVerifier[Any] | None,
        specification: ParameterSpecification | Missing,
    ) -> None:
        self.aliased: str | None = aliased
        self.description: str | None = description
        self.default: Any | Missing = default
        self.default_factory: Callable[[], Any] | Missing = default_factory
        self.validator: ParameterValidator[Any] | Missing = validator
        self.verifier: ParameterVerifier[Any] | None = verifier
        self.specification: ParameterSpecification | Missing = specification


class ParametrizedFunction[**Args, Result]:
    def __init__(
        self,
        function: Callable[Args, Result],
    ) -> None:
        assert (  # nosec: B101
            not isinstance(function, ParametrizedFunction)
        ), "Cannot parametrize the same function more than once!"

        self._call: Callable[Args, Result] = function
        globalns: dict[str, Any]
        if hasattr(function, "__globals__"):
            globalns = function.__globals__  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]

        else:
            globalns = sys.modules.get(function.__module__).__dict__

        self._parameters: dict[str, FunctionParameter] = {
            parameter.name: FunctionParameter.of(
                parameter,
                globalns=globalns,  # pyright: ignore[reportUnknownArgumentType]
                localns=None,
            )
            for parameter in signature(function).parameters.values()
        }

        mimic_function(function, within=self)

    def validate_arguments(
        self,
        **arguments: Any,
    ) -> dict[str, Any]:
        # TODO: add support for positional arguments
        context: ParameterValidationContext = (self.__qualname__,)
        validated: dict[str, Any] = {}
        for parameter in self._parameters.values():
            validated[parameter.name] = parameter.validated(
                arguments.get(
                    parameter.name,
                    arguments.get(
                        parameter.aliased,
                        MISSING,
                    )
                    if parameter.aliased
                    else MISSING,
                ),
                context=context,
            )

        return validated

    def __call__(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        assert not args, "Positional unkeyed arguments are not supported"  # nosec: B101
        return self._call(*args, **self.validate_arguments(**kwargs))  # pyright: ignore[reportCallIssue]


@final
class FunctionParameter:
    def __init__(  # noqa: PLR0913
        self,
        name: str,
        aliased: str | None,
        description: str | None,
        annotation: Any,
        default: Any | Missing,
        default_factory: ParameterDefaultFactory[Any] | Missing,
        allows_missing: bool,
        validator: ParameterValidator[Any],
        specification: ParameterSpecification | Missing,
    ) -> None:
        self.name: str = name
        self.aliased: str | None = aliased
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
                return self.validator(self.default_value(), (*context, f"@{self.name}"))

            elif self.allows_missing:
                return MISSING

            else:
                raise ParameterValidationError.missing(context=(*context, f"@{self.name}"))

        else:
            return self.validator(value, (*context, f"@{self.name}"))

    @classmethod
    def of(
        cls,
        parameter: Parameter,
        /,
        globalns: dict[str, Any],
        localns: dict[str, Any] | None,
    ) -> Self:
        if parameter.annotation is INSPECT_EMPTY:
            raise TypeError(
                "Untyped argument %s",
                parameter.name,
            )

        match parameter.default:
            case FunctionArgument() as argument:
                return cls(
                    name=parameter.name,
                    aliased=argument.aliased,
                    description=argument.description,
                    annotation=parameter.annotation,
                    default=argument.default,
                    default_factory=argument.default_factory,
                    allows_missing=allows_missing(
                        parameter.annotation,
                        globalns=globalns,
                        localns=localns,
                    ),
                    validator=argument.validator
                    if not_missing(argument.validator)
                    else parameter_validator(
                        parameter.annotation,
                        verifier=argument.verifier,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=frozenset(),
                    ),
                    specification=argument.specification
                    if not_missing(argument.specification)
                    else parameter_specification(
                        parameter.annotation,
                        description=argument.description,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=frozenset(),
                    ),
                )

            case default:
                return cls(
                    name=parameter.name,
                    aliased=None,
                    description=None,
                    annotation=parameter.annotation,
                    default=MISSING if default is INSPECT_EMPTY else default,
                    default_factory=MISSING,
                    allows_missing=allows_missing(
                        parameter.annotation,
                        globalns=globalns,
                        localns=localns,
                    ),
                    validator=parameter_validator(
                        parameter.annotation,
                        verifier=None,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=frozenset(),
                    ),
                    specification=parameter_specification(
                        parameter.annotation,
                        description=None,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=frozenset(),
                    ),
                )
