from collections.abc import Generator, Sequence
from typing import TypeVar

__all__ = [
    "split_sequence",
]

_T = TypeVar("_T")


def split_sequence(
    sequence: Sequence[_T],
    /,
    part_size: int,
) -> Generator[Sequence[_T], None, None]:
    parts_count, reminder = divmod(len(sequence), part_size)
    if reminder > 0:
        parts_count += 1

    return (sequence[i * part_size : (i + 1) * part_size] for i in range(parts_count))
