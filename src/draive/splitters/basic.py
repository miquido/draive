from collections.abc import Callable, Iterator, Sequence

__all__ = ("basic_split_text",)


def basic_split_text(
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
    used_splitter, fallback_splitters, parts = _split(
        text=text,
        splitters=splitters,
    )
    # then merge
    return _merge(
        parts=parts,
        splitter=used_splitter,
        fallback_splitters=fallback_splitters,
        part_size=part_size,
        count_size=count_size,
        part_overlap_size=part_overlap_size,
    )


def _split(
    text: str,
    splitters: Sequence[str],
) -> tuple[str, list[str], list[str]]:
    iterator: Iterator[str] = iter(splitters)
    while splitter := next(iterator, None):
        # try splitting using provided splitters
        parts: list[str] = text.split(splitter)
        if len(parts) == 1:
            continue

        else:
            # used_splitter, remaining_splitters, parts
            return (splitter, list(iterator), parts)

    # if splitting has still done nothing then fail
    raise ValueError("Text splitting failed")


def _merge(  # noqa: C901, PLR0912
    parts: list[str],
    part_size: int,
    count_size: Callable[[str], int],
    splitter: str,
    fallback_splitters: list[str],
    part_overlap_size: int | None,
) -> list[str]:
    result: list[str] = []
    accumulator: str = ""
    overlap_accumulator: list[str] = []
    last_part_idx: int = len(parts) - 1
    # iterate over splitted parts
    for idx, part in enumerate(parts):
        # Add the separator back if it is not the last part
        current_part: str
        if idx == last_part_idx:
            current_part = part

        else:
            current_part = part + splitter

        merged_part: str = accumulator + current_part
        # check if can add to previous part
        if count_size(merged_part) <= part_size:
            accumulator = merged_part
            overlap_accumulator.append(current_part)

        # check if current part is not too big on its own
        elif count_size(current_part) > part_size:
            if chunk := accumulator.strip():
                result.append(chunk)

            # clean up accumulators - we have made nested splitting
            accumulator = ""
            overlap_accumulator = []

            # do nested splitting if the part is too big
            result.extend(
                basic_split_text(
                    text=current_part,
                    part_size=part_size,
                    separators=fallback_splitters,
                    part_overlap_size=part_overlap_size,
                    count_size=count_size,
                ),
            )

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
                    and count_size(accumulator + current_part) <= part_size
                ):
                    overlap_accumulator = [element, *overlap_accumulator]
                    accumulator = merged_accumulator

                else:
                    break

            overlap_accumulator.append(current_part)
            accumulator = accumulator + current_part

        # otherwise make start a new part out of current
        else:
            if chunk := accumulator.strip():
                result.append(chunk)

            accumulator = current_part
            overlap_accumulator = [current_part]

    # add leftover if any
    if chunk := accumulator.strip():
        result.append(chunk)

    return result
