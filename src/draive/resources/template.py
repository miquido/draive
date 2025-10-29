import json
import re
from collections.abc import Callable, Coroutine, Mapping, Sequence, Set
from typing import Any, Protocol, final
from urllib.parse import ParseResult, parse_qs, quote, urlencode, urlparse

from haiway import MISSING, Meta, MetaValues, TypeSpecification
from haiway.attributes import Attribute

from draive.parameters import ParametrizedFunction
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


class ResourceAvailabilityCheck(Protocol):
    def __call__(self) -> bool: ...


@final
class ResourceTemplate[**Args](
    ParametrizedFunction[Args, Coroutine[None, None, Sequence[ResourceReference] | ResourceContent]]
):
    __slots__ = (
        "_check_availability",
        "_match_pattern",
        "declaration",
    )

    def __init__(
        self,
        /,
        template_uri: str,
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

        # Verify URI parameters against function arguments
        template_parameters: Set[str] = self._extract_template_parameters(template_uri)
        for template_parameter in template_parameters:
            parameter: Attribute | None = self._parameters.get(template_parameter)
            if parameter is None:
                raise ValueError(
                    f"URI template parameter {template_parameter} not found in function arguments"
                )

        self.declaration: ResourceReferenceTemplate
        object.__setattr__(
            self,
            "declaration",
            ResourceReferenceTemplate(
                template_uri=template_uri,
                mime_type=mime_type,
                meta=meta.updated(
                    name=name,
                    description=description,
                ),
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
        self._match_pattern: str | None
        object.__setattr__(
            self,
            "_match_pattern",
            _prepare_match_pattern(template_uri) if template_parameters else None,
        )

    @property
    def available(self) -> bool:
        try:
            return self._check_availability()

        except Exception:
            return False

    async def resolve(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Resource:
        values: Mapping[str, Any] = self.validate_arguments(**kwargs)
        provided_names: Set[str] = {
            parameter.name
            for parameter in self._parameters.values()
            if parameter.name in kwargs
            or (parameter.alias is not None and parameter.alias in kwargs)
        }
        resolved_uri: str = self._expand_uri(values, provided_names)

        try:
            return Resource(
                uri=resolved_uri,
                content=await super().__call__(*args, **kwargs),
                meta=self.declaration.meta,
            )

        except Exception as exc:
            raise ResourceCorrupted(uri=resolved_uri) from exc

    def _extract_template_parameters(
        self,
        uri: str,
    ) -> Set[str]:
        params: set[str] = set()
        # Find all {param} patterns in the URI template
        for match in re.findall(r"\{([^}]+)\}", uri):
            # Clean up parameter names (remove RFC 6570 prefixes)
            # Remove RFC 6570 prefixes
            if match.startswith("/") or match.startswith("#"):
                params.add(match[1:])

            elif match.startswith("?"):
                # Query parameters can be comma-separated: {?var1,var2}
                for p in match[1:].split(","):
                    params.add(p.strip())

            else:
                params.add(match)

        return params

    def _extract_query_parameters(
        self,
        parsed_uri: ParseResult,
    ) -> dict[str, str]:
        params: dict[str, str] = {}
        if parsed_uri.query:
            query_params = parse_qs(parsed_uri.query)
            for key, values in query_params.items():
                if values:
                    params[key] = values[0]  # Take first value

        return params

    def _as_str(self, value: Any) -> str | None:
        if value is None:
            return None

        return str(value)

    def _encode_path_segment(self, value: Any) -> str:
        s = self._as_str(value)
        return "" if s is None else quote(s, safe="")

    def _encode_fragment(self, value: Any) -> str:
        s = self._as_str(value)
        return "" if s is None else quote(s, safe="")

    def _expand_query(
        self,
        names: Sequence[str],
        base: str,
        values: Mapping[str, Any],
        provided: Set[str],
    ) -> str:
        items: list[tuple[str, str]] = []
        for name in names:
            if name not in provided:
                continue

            v = values.get(name)
            if v is None:
                continue

            sv = self._as_str(v)
            if sv is None or sv == "":
                continue

            items.append((name, sv))

        if not items:
            return ""

        prefix = "&" if base.rfind("?") > base.rfind("#") else "?"
        return prefix + urlencode(items)

    def _expand_uri(self, values: Mapping[str, Any], provided: Set[str]) -> str:
        template: str = self.declaration.template_uri
        resolved: list[str] = []

        for part in re.split(r"(\{[^}]+\})", template):
            if not part or not part.startswith("{"):
                resolved.append(part)
                continue

            # {/var}
            m = re.fullmatch(r"\{/([^}]+)\}", part)
            if m:
                name = m.group(1)
                seg = self._encode_path_segment(values.get(name))
                resolved.append("/" + seg if seg else "")
                continue

            # {?a,b}
            m = re.fullmatch(r"\{\?([^}]+)\}", part)
            if m:
                names = [n.strip() for n in m.group(1).split(",") if n.strip()]
                resolved.append(self._expand_query(names, "".join(resolved), values, provided))
                continue

            # {#var}
            m = re.fullmatch(r"\{#([^}]+)\}", part)
            if m:
                name = m.group(1)
                if name in provided:
                    frag = self._encode_fragment(values.get(name))
                    resolved.append("#" + frag if frag else "")
                else:
                    resolved.append("")
                continue

            # {var}
            m = re.fullmatch(r"\{([^}]+)\}", part)
            if m:
                name = m.group(1)
                if name.startswith("?") or name.startswith("#"):
                    resolved.append("")
                else:
                    resolved.append(self._encode_path_segment(values.get(name)))
                continue

            resolved.append(part)

        return "".join(resolved)

    def _extract_fragment_parameters(
        self,
        parsed_uri: ParseResult,
    ) -> dict[str, str]:
        params: dict[str, str] = {}
        if parsed_uri.fragment and "{#" in self.declaration.template_uri:
            # Find fragment template pattern {#var}
            fragment_start = self.declaration.template_uri.find("{#")
            if fragment_start != -1:
                fragment_end = self.declaration.template_uri.find("}", fragment_start)
                if fragment_end != -1:
                    param_name = self.declaration.template_uri[fragment_start + 2 : fragment_end]
                    params[param_name] = parsed_uri.fragment

        return params

    def _extract_path_template(self) -> str:
        if "://" in self.declaration.template_uri:
            reminder = self.declaration.template_uri.split("://", 1)[1]
            # Find where the netloc ends (first occurrence of {, /, ?, or #)
            netloc_end = len(reminder)
            for char in ["/", "{", "?", "#"]:
                pos = reminder.find(char)
                if pos != -1 and pos < netloc_end:
                    netloc_end = pos

            return reminder[netloc_end:]

        else:
            return self.declaration.template_uri

    def _extract_path_parameters_with_slash(
        self,
        actual_path: str,
    ) -> dict[str, str]:
        params: dict[str, str] = {}
        slash_params: list[str] = re.findall(r"\{/([^}]+)\}", self.declaration.template_uri)
        path_template: str = self._extract_path_template()

        # Create regex pattern
        template_for_regex: str = path_template
        for param in slash_params:
            template_for_regex = template_for_regex.replace(f"{{/{param}}}", "/([^/]+)")

        # Handle regular {var} patterns
        regular_params: list[str] = re.findall(r"\{([^}]+)\}", template_for_regex)
        for param in regular_params:
            if not param.startswith("/"):  # Skip already handled slash patterns
                template_for_regex = template_for_regex.replace(f"{{{param}}}", "([^/]+)")

        # Match against the actual path
        match: re.Match[str] | None = re.match(f"^{template_for_regex}$", actual_path)
        if match:
            positional_params: list[str] = [
                *slash_params,
                *(param for param in regular_params if not param.startswith("/")),
            ]
            groups: tuple[str | None, ...] = match.groups()
            for param, value in zip(positional_params, groups, strict=False):
                if value is None:
                    continue
                params[param] = value

        return params

    def _extract_simple_path_parameters(
        self,
        parsed_template: ParseResult,
        actual_path: str,
    ) -> dict[str, str]:
        params: dict[str, str] = {}
        template_parts = parsed_template.path.split("/")
        actual_parts = actual_path.split("/")

        for template_part, actual_part in zip(template_parts, actual_parts, strict=False):
            if template_part.startswith("{") and template_part.endswith("}"):
                param_name = template_part[1:-1]  # Remove braces
                # Skip query and fragment patterns
                if param_name.startswith("?") or param_name.startswith("#"):
                    continue

                params[param_name] = actual_part

        return params

    def _extract_uri_parameters(
        self,
        uri: str,
        /,
    ) -> Mapping[str, str]:
        params: dict[str, str] = {}
        parsed_uri: ParseResult = urlparse(uri)
        parsed_template: ParseResult = urlparse(self.declaration.template_uri)

        # Extract different types of parameters
        params.update(self._extract_query_parameters(parsed_uri))
        params.update(self._extract_fragment_parameters(parsed_uri))

        # Handle path parameters
        if "{/" in self.declaration.template_uri:
            params.update(self._extract_path_parameters_with_slash(parsed_uri.path))

        else:
            params.update(self._extract_simple_path_parameters(parsed_template, parsed_uri.path))

        return params

    def _coerce_uri_arguments(
        self,
        uri_params: Mapping[str, str],
    ) -> dict[str, Any]:
        coerced: dict[str, Any] = dict(uri_params)
        for parameter in self._parameters.values():
            raw_value: Any
            if parameter.alias is None:
                raw_value = uri_params.get(
                    parameter.name,
                    parameter.default(),
                )

            else:
                raw_value = uri_params.get(
                    parameter.alias,
                    uri_params.get(
                        parameter.name,
                        parameter.default(),
                    ),
                )

            if raw_value is MISSING:
                continue

            converted: Any = self._coerce_parameter_value(parameter, raw_value)
            coerced[parameter.name] = converted
            if parameter.alias:
                coerced[parameter.alias] = converted

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
        if "oneOf" in specification:
            for option in specification["oneOf"]:
                converted, changed = self._coerce_from_specification(option, value)
                if changed:
                    return converted, True

            return value, False

        type_name: Any | None = specification.get("type")
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

        if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
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
        if self._match_pattern is None:
            return self.declaration.template_uri == uri

        try:
            return bool(re.match(self._match_pattern, uri))

        except re.error:
            return False

    async def resolve_from_uri(
        self,
        uri: str,
        /,
    ) -> Resource:
        try:
            # Extract parameters from the URI using RFC 6570 template matching
            uri_params: Mapping[str, str] = self._extract_uri_parameters(uri)

            # If we have parameters, try to call with them
            call_kwargs: dict[str, Any] = (
                self._coerce_uri_arguments(uri_params) if uri_params else {}
            )

            content = await self.__call__(**call_kwargs)  # pyright: ignore[reportCallIssue]

            return Resource(
                uri=uri,  # Use the actual URI provided, not the template
                content=content,
                meta=self.declaration.meta,
            )

        except Exception as exc:
            raise ResourceCorrupted(uri=uri) from exc


def _prepare_match_pattern(uri_template: str) -> str:
    # Handle RFC 6570 URI template patterns
    tokens: list[str] = []
    for part in re.split(r"(\{[^}]+\})", uri_template):
        if re.fullmatch(r"\{/([^}]+)\}", part):
            # slash-prefixed, multi-segment
            tokens.append(r"(?:/[^/?#]+)+")

        elif re.fullmatch(r"\{\?[^}]+\}", part) or re.fullmatch(r"\{#[^}]+\}", part):
            # query/fragment - drop entirely
            continue

        elif re.fullmatch(r"\{([^}]+)\}", part):
            # single-segment variable
            tokens.append(r"[^/?#]+")

        else:
            # literal text
            tokens.append(re.escape(part))

    pattern = "".join(tokens)
    return f"^{pattern}$"


def resource[**Args](
    *,
    uri_template: str,
    mime_type: str | None = None,
    name: str | None = None,
    description: str | None = None,
    availability_check: ResourceAvailabilityCheck | None = None,
    meta: Meta | MetaValues | None = None,
) -> Callable[
    [Callable[Args, Coroutine[None, None, Sequence[ResourceReference] | ResourceContent]]],
    ResourceTemplate[Args],
]:
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
