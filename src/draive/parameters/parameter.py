from collections.abc import Mapping, MutableMapping
from typing import Any, Self, cast, final

from haiway import MISSING, DefaultValue, Missing
from haiway.state import AttributeAnnotation

from draive.parameters.specification import ParameterSpecification, parameter_specification
from draive.parameters.types import (
    ParameterConversion,
    ParameterValidation,
    ParameterValidationContext,
    ParameterVerification,
)
from draive.parameters.validation import ParameterValidator

__all__ = ("Parameter",)


@final
class Parameter[Type]:
    @classmethod
    def of(
        cls,
        annotation: AttributeAnnotation,
        /,
        *,
        name: str,
        alias: str | None,
        description: str | Missing,
        default: DefaultValue[Type],
        validator: ParameterValidation[Type] | Missing,
        verifier: ParameterVerification[Type] | Missing,
        converter: ParameterConversion[Type] | Missing,
        specification: ParameterSpecification | Missing,
        required: bool,
    ) -> Self:
        assert validator is MISSING or verifier is MISSING  # nosec: B101
        assert description is MISSING or specification is MISSING  # nosec: B101

        return cls(
            name=name,
            description=None if description is MISSING else cast(str, description),
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

    def __init__(
        self,
        *,
        name: str,
        description: str | None,
        alias: str | None,
        default: DefaultValue[Type],
        validator: ParameterValidation[Type],
        converter: ParameterConversion[Type] | Missing,
        specification: ParameterSpecification,
        required: bool,
    ) -> None:
        self.name: str = name
        self.description: str | None = description
        self.alias: str | None = alias
        self.default: DefaultValue[Type] = default
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

    def pick(
        self,
        mapping: MutableMapping[str, Any],
        /,
    ) -> Any:
        if self.alias:
            if self.alias in mapping:
                picked: Any = mapping[self.alias]
                del mapping[self.alias]
                return picked

            elif self.name in mapping:
                picked: Any = mapping[self.name]
                del mapping[self.name]
                return picked

            else:
                return MISSING

        elif self.name in mapping:
            picked: Any = mapping[self.name]
            del mapping[self.name]
            return picked

        else:
            return MISSING

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
