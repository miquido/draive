__all__ = [
    "tag_content",
]


def tag_content(  # noqa: C901, PLR0912
    tag: str,
    /,
    source: str,
) -> str | None:
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
                    return content

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
        elif in_content:
            if char == "<":
                in_tag = True
                accumulator = char

            else:
                content += char

        elif char == "<":
            in_tag = True
            accumulator = char

    return None  # tag not found or not closed
