from collections.abc import Generator

__all__ = [
    "xml_tag",
    "xml_tags",
]


def xml_tags(  # noqa: C901
    tag: str,
    /,
    source: str,
) -> Generator[str, None]:
    opening_tag_prefix: str = f"<{tag} "
    opening_tag: str = f"<{tag}>"
    closing_tag: str = f"</{tag}>"

    def check_opening(
        accumulator: str,
        /,
    ) -> bool:
        return accumulator == opening_tag or (
            accumulator.endswith(">") and accumulator.startswith(opening_tag_prefix)
        )

    accumulator: str = ""
    content: str = ""
    in_tag: bool = False
    in_content: bool = False

    for char in source:
        if in_tag:
            if char == ">":
                accumulator += char
                if in_content and accumulator == closing_tag:
                    in_content = False
                    yield content
                    content = ""  # clear to be ready for next tag

                elif check_opening(accumulator):
                    in_content = True
                    content = ""  # clear current content in case of nested tags

                elif in_content:
                    content += accumulator

                in_tag = False
                accumulator = ""

            elif char == "<":
                if in_content:
                    content += accumulator
                accumulator = char

            else:
                accumulator += char

        elif char == "<":
            in_tag = True
            accumulator = char

        elif in_content:
            content += char

        # else skip character


def xml_tag(
    tag: str,
    /,
    source: str,
) -> str | None:
    try:
        return next(
            xml_tags(
                tag,
                source=source,
            )
        )

    except StopIteration:
        return None
