from typing import Any, cast, is_typeddict

from haiway import AttributeAnnotation, Validator
from haiway.attributes.annotations import resolve_attribute

from draive.parameters.types import ParameterVerification

__all__ = ("ParameterValidator",)


class ParameterValidator[Type]:
    @classmethod
    def of_typed_dict[Dict](
        cls,
        typed_dict: type[Dict],
        /,
    ) -> Validator[Dict]:
        if not is_typeddict(typed_dict):
            raise ValueError("Type has to be a TypedDict")

        return cast(
            Validator[Dict],
            cls.of(
                resolve_attribute(
                    typed_dict,
                    module=typed_dict.__module__,
                    resolved_parameters={},  # TODO: type parameters?
                    recursion_guard={},
                ),
                verifier=None,
            ),
        )

    @classmethod
    def of(
        cls,
        annotation: AttributeAnnotation,
        /,
        *,
        verifier: ParameterVerification[Any] | None,
    ) -> Validator[Any]:
        validate: Validator[Any] = cast(Validator[Any], annotation.validate)

        if verifier is None:
            return validate

        else:
            verification = cast(ParameterVerification[Type], verifier)

            def validator(value: Any) -> Type:
                validated: Type = validate(value)
                verification(validated)
                return validated

            return cls(
                annotation,
                validation=validator,
            )

    def __init__(
        self,
        annotation: AttributeAnnotation,
        *,
        validation: Validator[Type],
    ) -> None:
        self.annotation: AttributeAnnotation = annotation
        self.validation: Validator[Type] = validation

    def __call__(
        self,
        value: Any,
    ) -> Type:
        return self.validation(value)
