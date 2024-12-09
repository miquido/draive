from collections.abc import Callable, Mapping
from typing import Any, Self, cast, final

from haiway import MISSING, Missing
from haiway.state import AttributeAnnotation

from draive.parameters.specification import ParameterSpecification, parameter_specification
from draive.parameters.types import (
    ParameterConverter,
    ParameterDefaultFactory,
    ParameterValidationContext,
    ParameterValidator,
    ParameterVerifier,
)
from draive.parameters.validation import parameter_validator

__all__ = [
    "Parameter",
]


def _missing_default() -> Missing:
    return MISSING


@final
class Parameter[Type]:
    @classmethod
    def of(  # noqa: PLR0913
        cls,
        annotation: AttributeAnnotation,
        /,
        *,
        name: str,
        alias: str | None,
        description: str | Missing,
        default_value: ParameterDefaultFactory[Type] | Missing,
        default_factory: ParameterDefaultFactory[Type] | Missing,
        validator: ParameterValidator[Type] | Missing,
        verifier: ParameterVerifier[Type] | Missing,
        converter: ParameterConverter[Type] | Missing,
        specification: ParameterSpecification | Missing,
    ) -> Self:
        assert validator is MISSING or verifier is MISSING  # nosec: B101
        assert description is MISSING or specification is MISSING  # nosec: B101
        assert default_value is MISSING or default_factory is MISSING  # nosec: B101
        default: ParameterDefaultFactory[Type] | Missing
        if default_value is MISSING:
            default = default_factory

        else:
            default = lambda: cast(Type, default_value)  # noqa: E731

        return cls(
            name=name,
            alias=alias,
            default=default,
            validator=parameter_validator(
                annotation,
                verifier=None if verifier is MISSING else cast(ParameterVerifier[Type], verifier),
            )
            if validator is MISSING
            else cast(ParameterValidator[Type], validator),
            converter=converter,
            specification=parameter_specification(
                annotation,
                description=None if description is MISSING else cast(str, description),
            )
            if specification is MISSING
            else cast(ParameterSpecification, specification),
        )

    def __init__(  # noqa: PLR0913
        self,
        *,
        name: str,
        alias: str | None,
        default: ParameterDefaultFactory[Type] | Missing,
        validator: ParameterValidator[Type],
        converter: ParameterConverter[Type] | Missing,
        specification: ParameterSpecification,
    ) -> None:
        self.name: str = name
        self.alias: str | None = alias
        self.default: Callable[[], Type | Missing]
        if default is MISSING:
            self.default = _missing_default

        else:
            self.default = cast(Callable[[], Type | Missing], default)

        self.validator: ParameterValidator[Any] = validator
        self.converter: ParameterConverter[Type] | None = (
            None if converter is MISSING else cast(ParameterConverter[Type], converter)
        )
        self.specification: ParameterSpecification = specification

    def find(
        self,
        mapping: Mapping[str, Any],
        /,
    ) -> Any:
        if self.alias:
            return mapping.get(
                self.alias,
                default=mapping.get(
                    self.name,
                    default=MISSING,
                ),
            )

        else:
            return mapping.get(
                self.name,
                default=MISSING,
            )

    def validated(
        self,
        value: Any,
        /,
        context: ParameterValidationContext,
    ) -> Any:
        with context.scope(f".{self.name}"):
            try:
                return self.validator(
                    self.default() if value is MISSING else value,
                    context=context,
                )

            except Exception as exc:
                if value is MISSING:
                    raise ValueError("Missing value") from None

                else:
                    raise exc
