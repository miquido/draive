from collections.abc import Generator

__all__ = [
    "markdown_block",
    "markdown_blocks",
]


def markdown_blocks(  # noqa: C901, PLR0912
    info: str = "",
    /,
    *,
    source: str,
) -> Generator[str, None]:
    boundary_sequence: str = "```"
    opening_sequence: str = f"{boundary_sequence}{info}"

    accumulator: str = ""
    content: str = ""
    in_sequence: bool = False
    in_content: bool = False

    for char in source:
        if in_sequence:
            if accumulator.startswith(boundary_sequence):
                if char.isspace():
                    if in_content:
                        in_content = False
                        yield content.strip()
                        content = ""  # clear to be ready for next block

                    elif accumulator.startswith(opening_sequence):
                        in_content = True
                        content = ""  # clear current content in case of nested blocks

                    in_sequence = False
                    accumulator = ""

                elif char == "`":
                    if in_content:
                        content += char
                    # keep the accumulator unchanged

                elif char.isalnum() and not in_content:  # take the info
                    accumulator += char

                elif in_content:
                    accumulator += char
                    content += accumulator
                    in_sequence = False
                    accumulator = ""

                else:
                    in_sequence = False
                    accumulator = ""

            elif char == "`":
                accumulator += char

            elif in_content:
                in_sequence = False
                accumulator += char
                content += accumulator
                accumulator = ""

            else:
                in_sequence = False
                accumulator = ""

        elif char == "`":
            in_sequence = True
            accumulator = char

        elif in_content:
            content += char

        # else skip character

    # when we hit the end while closing sequence was in place yield the last part
    if in_content and in_sequence and accumulator == boundary_sequence:
        yield content.strip()


def markdown_block(
    info: str = "",
    /,
    *,
    source: str,
) -> str | None:
    try:
        return next(
            markdown_blocks(
                info,
                source=source,
            )
        )

    except StopIteration:
        return None
