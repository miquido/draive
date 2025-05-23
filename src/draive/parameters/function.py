from collections.abc import Callable, Iterable, Mapping
from inspect import Parameter as InspectParameter
from inspect import _empty as INSPECT_EMPTY  # pyright: ignore[reportPrivateUsage]
from inspect import signature
from types import EllipsisType
from typing import Any, ClassVar, cast, final, get_type_hints, overload

from haiway import MISSING, DefaultValue, Missing
from haiway.state import AttributeAnnotation
from haiway.state.attributes import resolve_attribute_annotation
from haiway.utils import mimic_function

from draive.parameters.parameter import Parameter
from draive.parameters.specification import ParameterSpecification
from draive.parameters.types import ParameterValidation, ParameterVerification
from draive.parameters.validation import ParameterValidationContext

__all__ = (
    "Argument",
    "ParametrizedFunction",
)


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
    default_factory: Callable[[], Value] | Missing = MISSING,
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
    default_factory: Callable[[], Value] | Missing = MISSING,
    verifier: ParameterVerification[Value] | Missing = MISSING,
    specification: ParameterSpecification | Missing = MISSING,
) -> Value: ...


def Argument[Value](
    *,
    aliased: str | None = None,
    description: str | Missing = MISSING,
    default: Value | Missing = MISSING,
    default_factory: Callable[[], Value] | Missing = MISSING,
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
    def __init__(
        self,
        aliased: str | None,
        description: str | Missing,
        default: Any | Missing,
        default_factory: Callable[[], Any] | Missing,
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
        self._parameters: dict[str, Parameter[Any]]
        object.__setattr__(
            self,
            "_parameters",
            {},
        )
        self._variadic_keyword_parameters: Parameter[Any] | None
        object.__setattr__(
            self,
            "_variadic_keyword_parameters",
            None,
        )
        type_hints: Mapping[str, Any] = get_type_hints(function)
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
    def arguments(self) -> Iterable[Parameter[Any]]:
        return self._parameters.values()

    def validate_arguments(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # TODO: add support for positional arguments
        with ParameterValidationContext().scope(self.__class__.__qualname__) as context:
            validated: dict[str, Any] = {}
            if self._variadic_keyword_parameters is None:
                for parameter in self._parameters.values():
                    validated[parameter.name] = parameter.validated(
                        parameter.find(kwargs),
                        context=context,
                    )

            else:
                for parameter in self._parameters.values():
                    validated[parameter.name] = parameter.validated(
                        parameter.pick(kwargs),
                        context=context,
                    )

                for key, value in kwargs.items():
                    validated[key] = self._variadic_keyword_parameters.validated(
                        value,
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
) -> Parameter[Any]:
    if parameter.annotation is INSPECT_EMPTY or type_hint is None:
        raise TypeError(
            "Untyped argument %s",
            parameter.name,
        )

    attribute: AttributeAnnotation = resolve_attribute_annotation(
        type_hint,
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
                default=DefaultValue(
                    argument.default,
                    factory=argument.default_factory,
                ),
                validator=argument.validator,
                verifier=argument.verifier,
                converter=MISSING,
                specification=argument.specification,
                required=attribute.required
                and argument.default is MISSING
                and argument.default_factory is MISSING,
            )

        case DefaultValue() as default:  # pyright: ignore[reportUnknownVariableType]
            return Parameter[Any].of(
                attribute,
                name=parameter.name,
                alias=None,
                description=MISSING,
                default=default,  # pyright: ignore[reportUnknownArgumentType]
                validator=MISSING,
                verifier=MISSING,
                converter=MISSING,
                specification=MISSING,
                required=False,
            )

        case value:
            return Parameter[Any].of(
                attribute,
                name=parameter.name,
                alias=None,
                description=MISSING,
                default=DefaultValue(MISSING if value is INSPECT_EMPTY else value),
                validator=MISSING,
                verifier=MISSING,
                converter=MISSING,
                specification=MISSING,
                required=attribute.required and value is INSPECT_EMPTY,
            )
