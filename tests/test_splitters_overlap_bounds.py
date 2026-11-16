import pytest

from draive.splitters import split_text

_MODES = ("basic", "exhaustive", "exhaustive_regex")


@pytest.mark.parametrize("mode", _MODES)
def test_split_without_overlap_respects_part_size(mode: str) -> None:
    text: str = " ".join(["word"] * 40)

    parts: list[str] = split_text(text, part_size=12, count_size=len, mode=mode)  # pyright: ignore[reportCallIssue, reportArgumentType]

    assert all(len(part) <= 12 for part in parts)
    assert " ".join(parts).split() == text.split()


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("text", ("", "   \n\n  \n   " * 5))
def test_split_of_blank_text_is_stable(mode: str, text: str) -> None:
    parts: list[str] = split_text(text, part_size=5, count_size=len, mode=mode)  # pyright: ignore[reportCallIssue, reportArgumentType]

    # fits as a whole when short enough, otherwise nothing but whitespace is dropped
    assert parts in ([text], [])


@pytest.mark.parametrize("mode", _MODES)
def test_split_keeps_unicode_words_intact(mode: str) -> None:
    text: str = "héllo wörld 日本語 テキスト " * 5

    parts: list[str] = split_text(text, part_size=12, count_size=len, mode=mode)  # pyright: ignore[reportCallIssue, reportArgumentType]

    assert all(len(part) <= 12 for part in parts)
    assert " ".join(parts).split() == text.split()


@pytest.mark.parametrize("mode", _MODES)
def test_split_text_without_separators_fails_explicitly(mode: str) -> None:
    with pytest.raises(ValueError):
        split_text("a" * 50, part_size=5, count_size=len, mode=mode)  # pyright: ignore[reportCallIssue, reportArgumentType]


@pytest.mark.parametrize(
    ("mode", "text", "part_size", "overlap"),
    (
        ("basic", "xx xxxx x x xxxxx", 7, 6),
        ("exhaustive", "xx xxx xxxxxx", 11, 10),
        ("exhaustive_regex", "xxxxxx x xxxxxx xx", 11, 10),
    ),
)
def test_split_with_overlap_respects_part_size(
    mode: str,
    text: str,
    part_size: int,
    overlap: int,
) -> None:
    parts: list[str] = split_text(
        text,
        part_size=part_size,
        count_size=len,
        part_overlap_size=overlap,
        mode=mode,  # pyright: ignore[reportArgumentType]
    )

    assert all(len(part) <= part_size for part in parts)


@pytest.mark.parametrize("mode", _MODES)
def test_split_with_overlap_never_exceeds_part_size_for_random_inputs(mode: str) -> None:
    import random

    rng = random.Random(20260716)
    for _ in range(150):
        words: list[str] = ["x" * rng.randint(1, 9) for _ in range(rng.randint(1, 60))]
        text: str = " ".join(words)
        part_size: int = rng.randint(10, 40)
        overlap: int = rng.randint(0, part_size - 1)
        try:
            parts: list[str] = split_text(
                text,
                part_size=part_size,
                count_size=len,
                part_overlap_size=overlap,
                mode=mode,  # pyright: ignore[reportArgumentType]
            )
        except ValueError:
            continue  # explicit refusal is acceptable, silently oversized parts are not

        assert all(len(part) <= part_size for part in parts), (text, part_size, overlap)
        covered: set[str] = set(" ".join(parts).split())
        assert covered >= set(words)
