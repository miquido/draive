from collections.abc import Callable, Sequence
from re import Pattern
from typing import Literal, cast, overload

from draive.splitters.basic import basic_split_text
from draive.splitters.exhaustive import exhaustive_regex_split_text, exhaustive_split_text


@overload
def split_text(
    text: str,
    *,
    part_size: int,
    count_size: Callable[[str], int],
    separators: Sequence[str | Pattern[str]] | str | Pattern[str] | None = None,
    part_overlap_size: int | None = None,
    mode: Literal["exhaustive_regex"],
) -> list[str]: ...


@overload
def split_text(
    text: str,
    *,
    part_size: int,
    count_size: Callable[[str], int],
    separators: Sequence[str] | str | None = None,
    part_overlap_size: int | None = None,
    mode: Literal["basic", "exhaustive"] = "basic",
) -> list[str]: ...


def split_text(  # noqa: PLR0913
    text: str,
    part_size: int,
    count_size: Callable[[str], int],
    separators: Sequence[str | Pattern[str]] | str | Pattern[str] | None = None,
    part_overlap_size: int | None = None,
    mode: Literal[
        "basic",
        "exhaustive",
        "exhaustive_regex",
    ] = "basic",
) -> list[str]:
    match mode:
        case "basic":
            assert not isinstance(separators, Pattern)  # nosec: B101
            assert not any(isinstance(separator, Pattern) for separator in separators or [])  # nosec: B101
            return basic_split_text(
                text=text,
                part_size=part_size,
                count_size=count_size,
                separators=cast(Sequence[str] | str | None, separators),
                part_overlap_size=part_overlap_size,
            )

        case "exhaustive":
            assert not isinstance(separators, Pattern)  # nosec: B101
            assert not any(isinstance(separator, Pattern) for separator in separators or [])  # nosec: B101
            return exhaustive_split_text(
                text=text,
                part_size=part_size,
                count_size=count_size,
                separators=cast(Sequence[str] | str | None, separators),
                part_overlap_size=part_overlap_size,
            )
        case "exhaustive_regex":
            return exhaustive_regex_split_text(
                text=text,
                part_size=part_size,
                count_size=count_size,
                separators=separators,
                part_overlap_size=part_overlap_size,
            )
