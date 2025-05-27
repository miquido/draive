import re
from collections.abc import Callable, Coroutine, Mapping, Sequence, Set
from typing import Any, Protocol, final
from urllib.parse import ParseResult, parse_qs, urlparse

from draive.commons import Meta, MetaValues
from draive.parameters import Parameter, ParametrizedFunction
from draive.resources.types import (
    Resource,
    ResourceContent,
    ResourceException,
    ResourceTemplateDeclaration,
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
    ParametrizedFunction[Args, Coroutine[None, None, Sequence[Resource] | ResourceContent]]
):
    __slots__ = (
        "_check_availability",
        "_match_pattern",
        "declaration",
    )

    def __init__(
        self,
        /,
        uri_template: str,
        *,
        mime_type: str | None,
        name: str,
        description: str | None,
        availability_check: ResourceAvailabilityCheck | None,
        meta: Meta,
        function: Callable[Args, Coroutine[None, None, Sequence[Resource] | ResourceContent]],
    ) -> None:
        super().__init__(function)

        # Verify URI parameters against function arguments
        template_parameters: Set[str] = self._extract_template_parameters(uri_template)
        for template_parameter in template_parameters:
            parameter: Parameter[Any] | None = self._parameters.get(template_parameter)
            if parameter is None:
                raise ValueError(
                    f"URI template parameter {template_parameter} not found in function arguments"
                )

        self.declaration: ResourceTemplateDeclaration
        object.__setattr__(
            self,
            "declaration",
            ResourceTemplateDeclaration(
                uri_template=uri_template,
                mime_type=mime_type,
                name=name,
                description=description,
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
        self._match_pattern: str | None
        object.__setattr__(
            self,
            "_match_pattern",
            _prepare_match_pattern(uri_template) if template_parameters else None,
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
        try:
            return Resource(
                name=self.declaration.name,
                description=self.declaration.description,
                uri=self.declaration.uri_template,  # TODO: resolve uri with args
                content=await super().__call__(*args, **kwargs),
                meta=self.declaration.meta,
            )

        except Exception as exc:
            raise ResourceException(f"Resolving resource '{self.declaration.name}' failed") from exc

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

    def _extract_fragment_parameters(
        self,
        parsed_uri: ParseResult,
    ) -> dict[str, str]:
        params: dict[str, str] = {}
        if parsed_uri.fragment and "{#" in self.declaration.uri_template:
            # Find fragment template pattern {#var}
            fragment_start = self.declaration.uri_template.find("{#")
            if fragment_start != -1:
                fragment_end = self.declaration.uri_template.find("}", fragment_start)
                if fragment_end != -1:
                    param_name = self.declaration.uri_template[fragment_start + 2 : fragment_end]
                    params[param_name] = parsed_uri.fragment

        return params

    def _extract_path_template(self) -> str:
        if "://" in self.declaration.uri_template:
            remainder = self.declaration.uri_template.split("://", 1)[1]
            # Find where the netloc ends (first occurrence of {, /, ?, or #)
            netloc_end = len(remainder)
            for char in ["/", "{", "?", "#"]:
                pos = remainder.find(char)
                if pos != -1 and pos < netloc_end:
                    netloc_end = pos

            return remainder[netloc_end:]

        else:
            return self.declaration.uri_template

    def _extract_path_parameters_with_slash(
        self,
        actual_path: str,
    ) -> dict[str, str]:
        params: dict[str, str] = {}
        slash_params: list[Any] = re.findall(r"\{/([^}]+)\}", self.declaration.uri_template)
        path_template: str = self._extract_path_template()

        # Create regex pattern
        template_for_regex: str = path_template
        for param in slash_params:
            template_for_regex = template_for_regex.replace(f"{{/{param}}}", "/([^/]+)")

        # Handle regular {var} patterns
        regular_params: list[Any] = re.findall(r"\{([^}]+)\}", template_for_regex)
        for param in regular_params:
            if not param.startswith("/"):  # Skip already handled slash patterns
                template_for_regex = template_for_regex.replace(f"{{{param}}}", "([^/]+)")

        # Match against the actual path
        match: re.Match[str] | None = re.match(f"^{template_for_regex}$", actual_path)
        if match:
            all_params: list[Any] = slash_params + [
                p for p in regular_params if not p.startswith("/")
            ]
            for param, value in zip(all_params, match.groups(), strict=False):
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
        parsed_template: ParseResult = urlparse(self.declaration.uri_template)

        # Extract different types of parameters
        params.update(self._extract_query_parameters(parsed_uri))
        params.update(self._extract_fragment_parameters(parsed_uri))

        # Handle path parameters
        if "{/" in self.declaration.uri_template:
            params.update(self._extract_path_parameters_with_slash(parsed_uri.path))

        else:
            params.update(self._extract_simple_path_parameters(parsed_template, parsed_uri.path))

        return params

    def matches_uri(
        self,
        uri: str,
        /,
    ) -> bool:
        if self._match_pattern is None:
            return self.declaration.uri_template == uri

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
            if uri_params:
                content = await self.__call__(**uri_params)  # pyright: ignore[reportCallIssue]

            else:
                content = await self.__call__()  # pyright: ignore[reportCallIssue]

            return Resource(
                name=self.declaration.name,
                description=self.declaration.description,
                uri=uri,  # Use the actual URI provided, not the template
                content=content,
                meta=self.declaration.meta,
            )

        except Exception as exc:
            raise ResourceException(f"Resolving resource '{uri}' failed") from exc


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
    [Callable[Args, Coroutine[None, None, Sequence[Resource] | ResourceContent]]],
    ResourceTemplate[Args],
]:
    def wrap(
        function: Callable[Args, Coroutine[None, None, Sequence[Resource] | ResourceContent]],
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
