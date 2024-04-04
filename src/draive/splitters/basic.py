from collections.abc import Callable

__all__ = [
    "split_text",
]


def split_text(
    text: str,
    part_size: int,
    count_size: Callable[[str], int],
    separators: tuple[str, str] | str | None = None,
    part_overlap_size: int | None = None,
) -> list[str]:
    # if the text is already small enough just use it
    if count_size(text) < part_size:
        return [text]

    # split using provided separators
    splitter, parts = _split(
        text=text,
        separators=separators,
    )
    # then merge
    return _merge(
        parts=parts,
        splitter=splitter,
        part_size=part_size,
        count_size=count_size,
        part_overlap_size=part_overlap_size,
    )


def _split(
    text: str,
    separators: tuple[str, str] | str | None = None,
) -> tuple[str, list[str]]:
    # prepare initial splitter - default is new paragraph
    splitter: str
    alt_splitter: str
    match separators:
        case (str() as primary, str() as secondary):
            splitter = primary
            alt_splitter = secondary
        case str() as primary:
            splitter = primary
            alt_splitter = "\n"
        case None:
            splitter = "\n\n"
            alt_splitter = "\n"

    # try splitting using provided splitter
    parts: list[str] = text.split(splitter)
    # if splitting has done nothing retry using secondary splitter
    if len(parts) == 1:
        splitter = alt_splitter
        parts = text.split(splitter)
    # if splitting has still done nothing then fail
    if len(parts) == 1:
        raise ValueError("Failed to properly split text with provided separators")

    return (splitter, parts)


def _merge(  # noqa: C901
    parts: list[str],
    part_size: int,
    count_size: Callable[[str], int],
    splitter: str,
    part_overlap_size: int | None,
) -> list[str]:
    result: list[str] = []
    accumulator: list[str] = []
    accumulator_size: int = 0
    # iterate over splitted pats
    for part in parts:
        temp_size: int = count_size(part)
        # check if can add to previous part
        if accumulator_size + temp_size < part_size:
            accumulator.append(part)
            accumulator_size += temp_size
        # check if part is not too big on its own
        elif temp_size > part_size:
            chunk: str = splitter.join(accumulator).strip()
            if chunk:
                result.append(chunk)
            # do special splitting if it is too big indeed
            # force overlap and split on newlines and spaces
            result.extend(
                split_text(
                    text=part,
                    part_size=part_size,
                    separators=("\n", " "),
                    part_overlap_size=part_overlap_size
                    or int(part_size * 0.2),  # if there was no overlap force at least 20%
                    count_size=count_size,
                ),
            )
            accumulator = []
            accumulator_size = 0
        # if we have overlap defined do overlap between last part and current (not fitting)
        elif part_overlap_size := part_overlap_size:
            chunk: str = splitter.join(accumulator).strip()
            if chunk:
                result.append(chunk)
            overlap_size: int = 0
            overlap: list[str] = []
            for element in reversed(accumulator):
                element_size: int = count_size(element)
                if overlap_size + element_size < part_overlap_size:
                    overlap.append(element)
                    overlap_size += element_size
                else:
                    break

            accumulator = [*reversed(overlap), part]
            accumulator_size = overlap_size + temp_size
        # otherwise make start a new part out of current
        else:
            chunk: str = splitter.join(accumulator).strip()
            if chunk:
                result.append(chunk)
            accumulator = [part]
            accumulator_size = temp_size

    chunk: str = splitter.join(accumulator).strip()
    if chunk:  # add leftover if any
        result.append(chunk)

    return result
