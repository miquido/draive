from collections.abc import Callable, Sequence

__all__ = [
    "exhaustive_split_text",
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
