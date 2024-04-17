from collections.abc import Generator, Sequence

__all__ = [
    "split_sequence",
]


def split_sequence[Element](
    sequence: Sequence[Element],
    /,
    part_size: int,
) -> Generator[Sequence[Element], None, None]:
    parts_count, reminder = divmod(len(sequence), part_size)
    if reminder > 0:
        parts_count += 1

    return (sequence[i * part_size : (i + 1) * part_size] for i in range(parts_count))
