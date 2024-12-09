from collections.abc import Callable, Mapping
from typing import Any, Self, cast, final

from haiway import MISSING, Missing
from haiway.state import AttributeAnnotation

from draive.parameters.specification import ParameterSpecification, parameter_specification
from draive.parameters.types import (
    ParameterConversion,
    ParameterDefaultFactory,
    ParameterValidation,
    ParameterValidationContext,
    ParameterVerification,
)
from draive.parameters.validation import ParameterValidator

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
        validator: ParameterValidation[Type] | Missing,
        verifier: ParameterVerification[Type] | Missing,
        converter: ParameterConversion[Type] | Missing,
        specification: ParameterSpecification | Missing,
        required: bool,
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
            validator=ParameterValidator.of(
                annotation,
                verifier=None
                if verifier is MISSING
                else cast(ParameterVerification[Type], verifier),
                recursion_guard={},  # TODO: add enclosing type Self?
            )
            if validator is MISSING
            else cast(ParameterValidation[Type], validator),
            converter=converter,
            specification=parameter_specification(
                annotation,
                description=cast(str, description) if description is not MISSING else None,
            )
            if specification is MISSING
            else cast(ParameterSpecification, specification),
            required=required,
        )

    def __init__(  # noqa: PLR0913
        self,
        *,
        name: str,
        alias: str | None,
        default: ParameterDefaultFactory[Type] | Missing,
        validator: ParameterValidation[Type],
        converter: ParameterConversion[Type] | Missing,
        specification: ParameterSpecification,
        required: bool,
    ) -> None:
        self.name: str = name
        self.alias: str | None = alias
        self.default: Callable[[], Type | Missing]
        if default is MISSING:
            self.default = _missing_default

        else:
            self.default = cast(Callable[[], Type | Missing], default)

        self.validator: ParameterValidation[Any] = validator
        self.converter: ParameterConversion[Type] | None = (
            None if converter is MISSING else cast(ParameterConversion[Type], converter)
        )
        self.specification: ParameterSpecification = specification
        self.required: bool = required

    def find(
        self,
        mapping: Mapping[str, Any],
        /,
    ) -> Any:
        if self.alias:
            return mapping.get(
                self.alias,
                mapping.get(
                    self.name,
                    MISSING,
                ),
            )

        else:
            return mapping.get(
                self.name,
                MISSING,
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
                    value if value is not MISSING else self.default(),
                    context=context,
                )

            except Exception as exc:
                if value is MISSING:
                    raise ValueError("Missing value") from None

                else:
                    raise exc
