from collections.abc import Callable, Sequence
from typing import Literal

from draive.splitters.basic import basic_split_text
from draive.splitters.exhaustive import exhaustive_split_text


def split_text(  # noqa: PLR0913
    text: str,
    part_size: int,
    count_size: Callable[[str], int],
    separators: Sequence[str] | str | None = None,
    part_overlap_size: int | None = None,
    mode: Literal["basic", "exhaustive"] = "basic",
) -> list[str]:
    match mode:
        case "basic":
            return basic_split_text(
                text=text,
                part_size=part_size,
                count_size=count_size,
                separators=separators,
                part_overlap_size=part_overlap_size,
            )

        case "exhaustive":
            return exhaustive_split_text(
                text=text,
                part_size=part_size,
                count_size=count_size,
                separators=separators,
                part_overlap_size=part_overlap_size,
            )
