from collections.abc import Callable
from inspect import Parameter as InspectParameter
from inspect import _empty as INSPECT_EMPTY  # pyright: ignore[reportPrivateUsage]
from inspect import signature
from types import EllipsisType
from typing import Any, ClassVar, cast, final, overload

from haiway import MISSING, Missing, mimic_function
from haiway.state import AttributeAnnotation
from haiway.state.attributes import resolve_attribute_annotation

from draive.parameters.parameter import Parameter
from draive.parameters.specification import ParameterSpecification
from draive.parameters.types import (
    ParameterDefaultFactory,
    ParameterValidation,
    ParameterVerification,
)
from draive.parameters.validation import ParameterValidationContext

__all__ = [
    "Argument",
    "ParametrizedFunction",
]


@overload
def Argument[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default: Value | Missing = MISSING,
    validator: ParameterValidation[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Argument[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default_factory: ParameterDefaultFactory[Value] | Missing = MISSING,
    validator: ParameterValidation[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Argument[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default: Value | Missing = MISSING,
    verifier: ParameterVerification[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


@overload
def Argument[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default_factory: ParameterDefaultFactory[Value] | Missing = MISSING,
    verifier: ParameterVerification[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


def Argument[Value](  # noqa: PLR0913
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default: Value | Missing = MISSING,
    default_factory: ParameterDefaultFactory[Value] | Missing = MISSING,
    validator: ParameterValidation[Value] | Missing = MISSING,
    verifier: ParameterVerification[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value:  # it is actually a FunctionArgument, but type checker has to be fooled
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
        description: str | Missing,
        default: Any | Missing,
        default_factory: ParameterDefaultFactory[Any] | Missing,
        validator: ParameterValidation[Any] | Missing,
        verifier: ParameterVerification[Any] | Missing,
        specification: ParameterSpecification | Missing,
    ) -> None:
        self.aliased: str | None = aliased
        self.description: str | Missing = description
        self.default: Any | Missing = default
        self.default_factory: Callable[[], Any] | Missing = default_factory
        self.validator: ParameterValidation[Any] | Missing = validator
        self.verifier: ParameterVerification[Any] | Missing = verifier
        self.specification: ParameterSpecification | Missing = specification


class ParametrizedFunction[**Args, Result]:
    __IMMUTABLE__: ClassVar[EllipsisType] = ...

    def __init__(
        self,
        function: Callable[Args, Result],
        /,
    ) -> None:
        assert (  # nosec: B101
            not isinstance(function, ParametrizedFunction)
        ), "Cannot parametrize the same function more than once!"

        self._call: Callable[Args, Result] = function
        self._name: str = function.__name__
        self._parameters: dict[str, Parameter[Any]] = {
            parameter.name: _resolve_argument(
                parameter,
                module=function.__module__,
            )
            for parameter in signature(function).parameters.values()
        }

        mimic_function(function, within=self)

    def validate_arguments(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # TODO: add support for positional arguments
        with ParameterValidationContext().scope(self.__class__.__qualname__) as context:
            validated: dict[str, Any] = {}
            for parameter in self._parameters.values():
                validated[parameter.name] = parameter.validated(
                    parameter.find(kwargs),
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


def _resolve_argument(
    parameter: InspectParameter,
    /,
    *,
    module: str,
) -> Parameter[Any]:
    if parameter.annotation is INSPECT_EMPTY:
        raise TypeError(
            "Untyped argument %s",
            parameter.name,
        )

    attribute: AttributeAnnotation = resolve_attribute_annotation(
        parameter.annotation,
        module=module,
        type_parameters={},
        self_annotation=None,
        recursion_guard={},
    )

    match parameter.default:
        case FunctionArgument() as argument:
            return Parameter[Any].of(
                attribute,
                name=parameter.name,
                alias=argument.aliased,
                description=argument.description,
                default_value=argument.default,
                default_factory=argument.default_factory,
                validator=argument.validator,
                verifier=argument.verifier,
                converter=MISSING,
                specification=argument.specification,
                required=attribute.required
                and argument.default is MISSING
                and argument.default_factory is MISSING,
            )
        case default:
            return Parameter[Any].of(
                attribute,
                name=parameter.name,
                alias=None,
                description=MISSING,
                default_value=MISSING if default is INSPECT_EMPTY else default,
                default_factory=MISSING,
                validator=MISSING,
                verifier=MISSING,
                converter=MISSING,
                specification=MISSING,
                required=attribute.required and default is INSPECT_EMPTY,
            )
