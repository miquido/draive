import json
from collections.abc import Callable, Collection, Coroutine, Mapping, Sequence
from typing import Any, Protocol, cast, final, runtime_checkable

from haiway import Function, Meta, MetaValues, Missing, TypeSpecification, URITemplate
from haiway.attributes import Attribute

from draive.resources.types import (
    MimeType,
    Resource,
    ResourceContent,
    ResourceCorrupted,
    ResourceReference,
    ResourceReferenceTemplate,
)

__all__ = (
    "ResourceAvailabilityCheck",
    "ResourceTemplate",
    "resource",
)


@runtime_checkable
class ResourceAvailabilityCheck(Protocol):
    def __call__(self) -> bool: ...


@final
class ResourceTemplate[**Args](
    Function[Args, Coroutine[None, None, Collection[ResourceReference] | ResourceContent]]
):
    __slots__ = (
        "_check_availability",
        "declaration",
    )

    def __init__(
        self,
        /,
        template_uri: URITemplate | str,
        *,
        name: str,
        description: str | None,
        mime_type: MimeType | None,
        availability_check: ResourceAvailabilityCheck | None,
        meta: Meta,
        function: Callable[
            Args, Coroutine[None, None, Sequence[ResourceReference] | ResourceContent]
        ],
    ) -> None:
        super().__init__(function)
        assert all(arg.name in self._keyword_arguments for arg in self._positional_arguments)  # nosec: B101
        assert self._variadic_positional_arguments is None  # nosec: B101

        uri_template: URITemplate = (
            template_uri if isinstance(template_uri, URITemplate) else URITemplate.of(template_uri)
        )
        parameters_by_alias: dict[str, Attribute] = {
            attribute.alias: attribute
            for attribute in self._keyword_arguments.values()
            if attribute.alias is not None
        }
        for template_parameter in uri_template.variables:
            # variadic keyword arguments are left for extras, template variables identify
            # the resource and always have to be declared
            parameter: Attribute | None = self._keyword_arguments.get(
                template_parameter
            ) or parameters_by_alias.get(template_parameter)
            if parameter is None:
                raise ValueError(
                    f"URI template parameter {template_parameter} not found in function arguments"
                )

        self.declaration: ResourceReferenceTemplate
        object.__setattr__(
            self,
            "declaration",
            ResourceReferenceTemplate.of(
                template_uri=uri_template,
                name=name,
                description=description,
                mime_type=mime_type,
                meta=meta,
            ),
        )
        self._check_availability: ResourceAvailabilityCheck
        object.__setattr__(
            self,
            "_check_availability",
            availability_check
            or (
                lambda: True  # available by default
            ),
        )

    @property
    def available(self) -> bool:
        try:
            return self._check_availability()

        except Exception:
            return False

    @property
    def has_args(self) -> bool:
        return bool(self.declaration.template_uri.variables)

    async def resolve(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Resource:
        assert not args  # nosec: B101
        _, v_kwargs = self.validate_arguments(*args, **kwargs)
        resolved_uri: str = self.declaration.template_uri.expand(self._template_values(v_kwargs))

        try:
            return Resource(
                uri=resolved_uri,
                resource=await super().__call__(*args, **kwargs),
            )

        except Exception as exc:
            raise ResourceCorrupted(uri=resolved_uri) from exc

    def _template_values(
        self,
        arguments: Mapping[str, Any],
        /,
    ) -> Mapping[str, Any]:
        # validated arguments are keyed by name while a template may refer to the alias
        values: dict[str, Any] = {}
        for argument in self._keyword_arguments.values():
            if argument.name not in arguments:
                continue

            values[argument.name] = arguments[argument.name]
            if argument.alias is not None:
                values[argument.alias] = arguments[argument.name]

        return values

    def _coerce_uri_arguments(
        self,
        uri_params: Mapping[str, str],
    ) -> dict[str, Any]:
        coerced: dict[str, Any] = {}
        for argument in self._keyword_arguments.values():
            raw_value: Any
            if argument.alias is not None and argument.alias in uri_params:
                raw_value = uri_params[argument.alias]

            elif argument.name in uri_params:
                raw_value = uri_params[argument.name]

            else:
                raw_value = argument.default()

            if raw_value is Missing:
                continue

            converted: Any = self._coerce_parameter_value(argument, raw_value)
            # Pass only the canonical parameter name to avoid duplicate alias/name kwargs
            coerced[argument.name] = converted

        return coerced

    def _coerce_parameter_value(
        self,
        parameter: Attribute,
        value: Any,
    ) -> Any:
        if not isinstance(value, str):
            return value

        assert parameter.specification  # nosec: B101
        converted, _ = self._coerce_from_specification(parameter.specification, value)
        return converted

    def _coerce_from_specification(  # noqa: C901, PLR0911, PLR0912
        self,
        specification: TypeSpecification,
        value: str,
    ) -> tuple[Any, bool]:
        if "anyOf" in specification:
            if any(self._accepts_any_text(option) for option in specification["anyOf"]):
                # text is already a valid value, converting it would alter what was provided
                return value, False

            for option in specification["anyOf"]:
                converted, changed = self._coerce_from_specification(option, value)
                if changed:
                    return converted, True

            return value, False

        type_name: Any | None = specification.get("type")
        if isinstance(type_name, Sequence) and not isinstance(type_name, str):
            if self._accepts_any_text(specification):
                return value, False

            # alternatives of plain types are named by a list of types instead
            # of a list of alternatives - each is tried in the declared order
            for alternative in cast(Sequence[Any], type_name):
                if not isinstance(alternative, str):
                    continue

                converted, changed = self._coerce_from_specification(
                    # the remaining keys (`enum` especially) still apply to each alternative
                    cast(TypeSpecification, {**specification, "type": alternative}),
                    value,
                )
                if changed:
                    return converted, True

            return value, False

        if type_name == "integer":
            try:
                return int(value), True
            except ValueError:
                return value, False

        if type_name == "number":
            try:
                return float(value), True
            except ValueError:
                return value, False

        if type_name == "boolean":
            parsed_bool = self._parse_boolean(value)
            if parsed_bool is not None:
                return parsed_bool, True

            return value, False

        if type_name == "array":
            parsed_array = self._parse_json_sequence(value)
            if parsed_array is not None:
                return parsed_array, True

            return value, False

        if type_name == "object":
            parsed_object = self._parse_json_mapping(value)
            if parsed_object is not None:
                return parsed_object, True

            return value, False

        if type_name == "null":
            normalised = value.strip().lower()
            if normalised in {"null", "none"}:
                return None, True

            return value, False

        if "enum" in specification:
            converted_enum, enum_changed = self._match_enum(specification["enum"], value)
            if enum_changed:
                return converted_enum, True

        return value, False

    def _accepts_any_text(
        self,
        specification: TypeSpecification,
    ) -> bool:
        if "enum" in specification or "const" in specification:
            return False

        type_name: Any | None = specification.get("type")
        if isinstance(type_name, Sequence) and not isinstance(type_name, str):
            return "string" in cast(Sequence[Any], type_name)

        return type_name == "string"

    def _parse_boolean(self, raw: str) -> bool | None:
        normalised = raw.strip().lower()
        if normalised in {"true", "1", "yes", "y", "on"}:
            return True

        if normalised in {"false", "0", "no", "n", "off"}:
            return False

        return None

    def _parse_json_sequence(
        self,
        raw: str,
    ) -> Sequence[Any] | None:
        try:
            parsed: Mapping[str, Any] | Sequence[Any] = json.loads(raw)

        except ValueError:
            return None

        if isinstance(parsed, Sequence) and not isinstance(parsed, str | bytes):
            return parsed

        return None

    def _parse_json_mapping(
        self,
        raw: str,
    ) -> Mapping[str, Any] | None:
        try:
            parsed: Mapping[str, Any] | Sequence[Any] = json.loads(raw)

        except ValueError:
            return None

        if isinstance(parsed, Mapping):
            return parsed

        return None

    def _match_enum(  # noqa: C901, PLR0911, PLR0912
        self,
        enum_values: Sequence[Any],
        value: str,
    ) -> tuple[Any, bool]:
        for enum_value in enum_values:
            if isinstance(enum_value, str) and value == enum_value:
                return enum_value, True

            if isinstance(enum_value, bool):
                parsed_bool = self._parse_boolean(value)
                if parsed_bool is not None and parsed_bool is enum_value:
                    return enum_value, True

            if isinstance(enum_value, int):
                try:
                    if int(value) == enum_value:
                        return enum_value, True
                except ValueError:
                    continue

            if isinstance(enum_value, float):
                try:
                    if float(value) == enum_value:
                        return enum_value, True
                except ValueError:
                    continue

            if enum_value is None:
                normalised = value.strip().lower()
                if normalised in {"null", "none", ""}:
                    return None, True

            if value == str(enum_value):
                return enum_value, True

        return value, False

    def matches_uri(
        self,
        uri: str,
        /,
    ) -> bool:
        return self.declaration.template_uri.matches(uri)

    async def resolve_from_uri(
        self,
        uri: str,
        /,
    ) -> Resource:
        try:
            # the template verifies the whole URI and decodes its parameters within the
            # boundaries it declared, a mismatch means the URI is not of this resource
            uri_params: Mapping[str, str] | None = self.declaration.template_uri.match(uri)
            if uri_params is None:
                raise ResourceCorrupted(uri=uri)

            call_kwargs: dict[str, Any] = (
                self._coerce_uri_arguments(uri_params) if uri_params else {}
            )

            return Resource(
                uri=uri,  # Use the actual URI provided, not the template
                resource=await self.__call__(**call_kwargs),  # pyright: ignore[reportCallIssue]
            )

        except Exception as exc:
            raise ResourceCorrupted(uri=uri) from exc


def resource[**Args](
    *,
    uri_template: URITemplate | str,
    mime_type: str | None = None,
    name: str | None = None,
    description: str | None = None,
    availability_check: ResourceAvailabilityCheck | None = None,
    meta: Meta | MetaValues | None = None,
) -> Callable[
    [Callable[Args, Coroutine[None, None, Sequence[ResourceReference] | ResourceContent]]],
    ResourceTemplate[Args],
]:
    """
    Declare a resource template from a function resolving its content.

    Parameters
    ----------
    uri_template : URITemplate | str
        RFC 6570 template of the resource URI, its variables have to match the function
        arguments by name or alias, variadic keyword arguments do not match them. A text is
        verified when the resource is declared.
    mime_type : str | None
        Mime type of the resource content.
    name : str | None
        Resource name, the function name when not provided.
    description : str | None
        Resource description.
    availability_check : ResourceAvailabilityCheck | None
        Check telling if the resource is currently available.
    meta : Meta | MetaValues | None
        Additional resource metadata.

    Returns
    -------
    Callable
        Decorator turning the function into a ``ResourceTemplate``.

    Raises
    ------
    ValidationError
        If the template text is malformed.
    ValueError
        If a template variable has no matching function argument.
    """

    def wrap(
        function: Callable[
            Args, Coroutine[None, None, Sequence[ResourceReference] | ResourceContent]
        ],
    ) -> ResourceTemplate[Args]:
        return ResourceTemplate[Args](
            uri_template,
            mime_type=mime_type,
            name=name or function.__name__,
            description=description,
            availability_check=availability_check,
            function=function,
            meta=Meta.of(meta),
        )

    return wrap
