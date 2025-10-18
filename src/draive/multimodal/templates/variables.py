from collections.abc import Generator, Mapping, MutableSequence, Set

from haiway import Immutable

from draive.multimodal.content import Multimodal, MultimodalContent

__all__ = (
    "parse_template_variables",
    "resolve_multimodal_template",
    "resolve_multimodal_template",
)


def parse_template_variables(
    template: str,
) -> Set[str]:
    return {
        segment.name
        for segment in _split_template(template)
        if isinstance(segment, TemplateVariable)
    }


def resolve_text_template(
    template: str,
    *,
    arguments: Mapping[str, Multimodal],
) -> str:
    parts: MutableSequence[str] = []
    for segment in _split_template(template):
        if isinstance(segment, str):
            parts.append(segment)

        elif segment.name in arguments:
            argument: Multimodal = arguments[segment.name]
            if isinstance(argument, str):
                parts.append(argument)

            else:
                parts.append(argument.to_str())

        else:
            raise KeyError(f"Missing template argument: {segment.name}")

    return "".join(parts)


def resolve_multimodal_template(
    template: str,
    *,
    arguments: Mapping[str, Multimodal],
) -> MultimodalContent:
    parts: MutableSequence[Multimodal] = []
    for segment in _split_template(template):
        if isinstance(segment, str):
            parts.append(segment)

        elif segment.name in arguments:
            parts.append(arguments[segment.name])

        else:
            raise KeyError(f"Missing template argument: {segment.name}")

    if not parts:
        return MultimodalContent.empty

    return MultimodalContent.of(*parts)


class TemplateVariable(Immutable):
    name: str


type TemplateSegment = TemplateVariable | str


def _split_template(
    template: str,
) -> Generator[TemplateSegment]:
    cursor: int = 0

    while cursor < len(template):
        # find variable opening
        open_index: int = template.find("{%", cursor)

        if open_index == -1:
            if literal := template[cursor:]:
                yield literal

            break  # no chance for more variables

        # find variable closing
        end_index: int = template.find("%}", open_index + 2)
        if end_index == -1:
            if literal := template[cursor:]:
                yield literal

            break  # no chance for more variables

        variable_name: str = template[open_index + 2 : end_index]
        if not variable_name or any(char.isspace() for char in variable_name):
            if literal := template[cursor : end_index + 2]:
                yield literal

            cursor = end_index + 2
            continue  # move cursor and continue

        if prefix := template[cursor:open_index]:
            yield prefix

        yield TemplateVariable(name=variable_name)
        cursor = end_index + 2  # add variable to the list and continue
