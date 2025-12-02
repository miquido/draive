import re
from collections.abc import Callable, Generator, Mapping, MutableSequence
from typing import Final

from draive.multimodal.content import Multimodal, MultimodalContent

# Match `{%variable%}` where the name contains no whitespace.
_PLACEHOLDER_PATTERN: Final[re.Pattern[str]] = re.compile(r"{%([^\s%]+)%}")

__all__ = (
    "parse_template_variables",
    "resolve_multimodal_template",
    "resolve_text_template",
)


def parse_template_variables(
    template: str,
) -> Generator[str]:
    for match in _PLACEHOLDER_PATTERN.finditer(template):
        yield match.group(1)


def resolve_text_template(
    template: str,
    *,
    arguments: Mapping[str, Multimodal],
) -> str:
    parts: MutableSequence[str] = []
    append: Callable[[str], None] = parts.append
    get_argument: Callable[[str], Multimodal | None] = arguments.get
    cursor: int = 0

    for match in _PLACEHOLDER_PATTERN.finditer(template):
        start: int
        end: int
        start, end = match.span()
        if start > cursor:
            append(template[cursor:start])

        variable_name: str = match.group(1)
        variable_value: Multimodal | None = get_argument(variable_name)
        if variable_value is None:
            raise KeyError(f"Missing template argument: {variable_name}")

        if isinstance(variable_value, str):
            append(variable_value)

        else:
            append(variable_value.to_str())

        cursor = end

    if cursor < len(template):
        append(template[cursor:])

    return "".join(parts)


def resolve_multimodal_template(
    template: str,
    *,
    arguments: Mapping[str, Multimodal],
) -> MultimodalContent:
    parts: MutableSequence[Multimodal] = []
    append: Callable[[Multimodal], None] = parts.append
    get_argument: Callable[[str], Multimodal | None] = arguments.get
    cursor: int = 0

    for match in _PLACEHOLDER_PATTERN.finditer(template):
        start, end = match.span()
        if start > cursor:
            append(template[cursor:start])

        variable_name: str = match.group(1)
        variable_value: Multimodal | None = get_argument(variable_name)
        if variable_value is None:
            raise KeyError(f"Missing template argument: {variable_name}")

        append(variable_value)
        cursor = end

    if cursor < len(template):
        append(template[cursor:])

    if not parts:
        return MultimodalContent.empty

    return MultimodalContent.of(*parts)
