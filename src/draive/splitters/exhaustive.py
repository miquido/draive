import re
from collections.abc import Callable, Sequence
from re import Match, Pattern

__all__ = [
    "exhaustive_split_text",
    "exhaustive_regex_split_text",
]


def exhaustive_split_text(
    text: str,
    part_size: int,
    count_size: Callable[[str], int],
    separators: Sequence[str] | str | None = None,
    part_overlap_size: int | None = None,
) -> list[str]:
    # if the text is already small enough just use it
    if count_size(text) <= part_size:
        return [text]

    splitters: Sequence[str]

    match separators:
        case None:
            splitters = ["\n\n", "\n", " "]

        case str() as splitter:
            splitters = [splitter, " "]

        case [*separators]:
            splitters = separators

    # split using provided separators
    parts: list[str] = _split(
        text=text,
        splitters=splitters,
    )
    # then merge
    return _merge(
        parts=parts,
        part_size=part_size,
        count_size=count_size,
        part_overlap_size=part_overlap_size,
    )


def _split(
    text: str,
    splitters: Sequence[str],
) -> list[str]:
    result: list[str] = [text]

    for splitter in splitters:
        parts: list[str] = []
        for element in result:
            splitted: list[str] = element.split(splitter)
            for part in splitted[:-1]:
                parts.append(part + splitter)
            parts.append(splitted[-1])
        result = parts

    return result


def exhaustive_regex_split_text(
    text: str,
    part_size: int,
    count_size: Callable[[str], int],
    separators: Sequence[str | Pattern[str]] | str | Pattern[str] | None = None,
    part_overlap_size: int | None = None,
) -> list[str]:
    if count_size(text) <= part_size:
        return [text]

    splitters: Sequence[str | Pattern[str]]

    match separators:
        case None:
            splitters = ["(\n\n)", "(\n)", r"(\s)"]
        case str() | Pattern() as splitter:
            splitters = [splitter, r"(\s)"]
        case [*separators]:
            splitters = separators

    parts: list[str] = _regex_split(
        text=text,
        splitters=splitters,
    )
    return _merge(
        parts=parts,
        part_size=part_size,
        count_size=count_size,
        part_overlap_size=part_overlap_size,
    )


def _regex_split(
    text: str,
    splitters: Sequence[str | Pattern[str]],
) -> list[str]:
    result: list[str] = [text]

    for splitter in splitters:
        parts: list[str] = []
        for element in result:
            splitted: list[str] = re.split(
                f"({_remove_capturing_groups(splitter)})",
                element,
            )
            # Add the first part (always a non-splitter part)
            parts.append(splitted[0])
            # Combine splitter and following part
            parts.extend(
                [
                    splitter + following_part
                    for splitter, following_part in zip(
                        splitted[1::2],
                        splitted[2::2],
                        strict=False,
                    )
                ]
            )
        result = parts

    return result


def _remove_capturing_groups(pattern: str | Pattern[str]) -> str:
    if isinstance(pattern, Pattern):
        pattern = pattern.pattern

    def process_group(match: Match[str]) -> str:
        group_type, group_content = match.groups()
        special_constructs: set[str] = {"?:", "?=", "?!", "?<=", "?<!", "?>"}

        if group_type in special_constructs:
            return match.group(0)  # Return special constructs unchanged

        return group_content  # Remove parentheses for normal capturing groups

    # Regex to match all types of parentheses groups
    group_pattern = r"""\(  # Match opening parenthesis
                        (\??[=!<>:]?)  # Optional special characters for non-capturing groups
                        (  # Main content of the group
                            (?:
                                [^()\\]  # Any character that's not a parenthesis or backslash
                                |  # OR
                                \\.  # An escaped character
                                |  # OR
                                \((?:[^()\\]|\\.)*\)  # A nested group
                            )*
                        )
                    \)  # Match closing parenthesis
                    """

    processed_pattern: str = re.sub(
        group_pattern,
        process_group,
        pattern,
        flags=re.VERBOSE,
    )

    return processed_pattern


def _merge(
    parts: list[str],
    part_size: int,
    count_size: Callable[[str], int],
    part_overlap_size: int | None,
) -> list[str]:
    result: list[str] = []
    accumulator: str = ""
    overlap_accumulator: list[str] = []

    # iterate over splitted parts
    for part in parts:
        merged_part: str = accumulator + part
        # check if can add to previous part
        if count_size(merged_part) <= part_size:
            accumulator = merged_part
            overlap_accumulator.append(part)

        # check if current part is not too big on its own
        elif count_size(part) > part_size:
            raise ValueError("Failed to split text fitting required size.")

        # if we have overlap defined do overlap between last part and current (not fitting)
        elif part_overlap_size := part_overlap_size:
            if chunk := accumulator.strip():
                result.append(chunk)

            # clear accumulator - we will fill it now
            accumulator = ""
            for element in reversed(overlap_accumulator):
                merged_accumulator: str = element + accumulator

                if (
                    count_size(merged_accumulator) < part_overlap_size
                    and count_size(accumulator + part) <= part_size
                ):
                    overlap_accumulator = [element, *overlap_accumulator]
                    accumulator = merged_accumulator

                else:
                    break

            overlap_accumulator.append(part)
            accumulator = accumulator + part

        # otherwise start a new part out of current
        else:
            if chunk := accumulator.strip():
                result.append(chunk)

            accumulator = part
            overlap_accumulator = [part]

    # add leftover if any
    if chunk := accumulator.strip():
        result.append(chunk)

    return result
