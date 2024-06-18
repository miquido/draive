import pytest
from draive import split_text


@pytest.fixture
def sample_text() -> str:
    with open("tests/data/sample_text.txt") as file:
        return file.read()


@pytest.fixture
def nested_splitting_expected_result() -> list[str]:
    with open("tests/data/nested_splitting.txt") as file:
        return file.read().split("\n")


@pytest.fixture
def overlap_splitting_expected_result() -> list[str]:
    with open("tests/data/overlap_splitting.txt") as file:
        return file.read().split("\n")


def test_returns_single_part_when_text_fits_part_size() -> None:
    assert split_text(
        text="0",
        part_size=1,
        count_size=len,
        separators=[],
    ) == ["0"]


def test_fails_when_no_separators_provided() -> None:
    with pytest.raises(ValueError):
        split_text(
            text="0",
            part_size=0,
            count_size=len,
            separators=[],
        )


def test_fails_when_cant_split() -> None:
    with pytest.raises(ValueError):
        split_text(
            text="01",
            part_size=1,
            count_size=len,
            separators=["."],
        )


def test_strips_whitespaces_with_leading_and_trailing_part_whitespaces() -> None:
    assert split_text(
        text=" 01  23 ",
        part_size=3,
        count_size=len,
        separators=[" "],
    ) == ["01", "23"]


def test_uses_fallback_splitter_when_required() -> None:
    assert split_text(
        text="01 23",
        part_size=3,
        count_size=len,
        separators=["\n", " "],
    ) == ["01", "23"]


def test_uses_all_splitters_when_required() -> None:
    assert split_text(
        text="01 23\n45 \n67",
        part_size=3,
        count_size=len,
        separators=["\n", " "],
    ) == ["01", "23", "45", "67"]


def test_adds_trailing_splitters_when_required() -> None:
    assert split_text(
        text="01.23.",
        part_size=3,
        count_size=len,
        separators=["."],
    ) == ["01.", "23."]


def test_uses_paragraphs_splitting_with_default_separators() -> None:
    assert split_text(
        text="01\n\n23",
        part_size=4,
        count_size=len,
    ) == ["01", "23"]


def test_uses_newline_splitting_with_default_separators() -> None:
    assert split_text(
        text="01\n23",
        part_size=4,
        count_size=len,
    ) == ["01", "23"]


def test_uses_words_splitting_with_default_separators() -> None:
    assert split_text(
        text="01 23",
        part_size=3,
        count_size=len,
    ) == ["01", "23"]


def test_uses_words_splitting_when_one_separators_provided() -> None:
    assert split_text(
        text="01 23",
        part_size=3,
        count_size=len,
        separators=".",
    ) == ["01", "23"]


def test_splitting_text_when_parts_fit_part_size_exactly() -> None:
    assert split_text(
        text="ab.cd.ef.gh.",
        part_size=3,
        count_size=len,
        separators=".",
    ) == ["ab.", "cd.", "ef.", "gh."]


def test_splitting_text_when_each_base_part_doesnt_fit_part_size() -> None:
    assert split_text(
        text="1234 5678.890 ab cd.efgh ijklm",
        part_size=6,
        count_size=len,
        separators=[".", " "],
    ) == ["1234", "5678.", "890", "ab cd.", "efgh", "ijklm"]


def test_overlapping_parts_with_primary_splitting() -> None:
    assert split_text(
        text="a.b.c.d.e.f.g.h.i.j.k.l.m",
        part_size=9,
        count_size=len,
        separators=["."],
        part_overlap_size=6,
    ) == ["a.b.c.d.", "c.d.e.f.", "e.f.g.h.", "g.h.i.j.", "i.j.k.l.m"]


def test_overlapping_parts_with_nested_splitting() -> None:
    assert split_text(
        text="1234 56.788.90.ab.cd.ef.g.h ijklm",
        part_size=6,
        count_size=len,
        separators=[" ", "."],
        part_overlap_size=6,
    ) == [
        "1234",
        "56.",
        "56.788.",
        "788.90.",
        "90.ab.",
        "ab.cd.",
        "cd.ef.",
        "ef.g.",
        "ef.g.h",
        "ijklm",
    ]


def test_uses_nested_splitting_with_long_text(
    sample_text: str,
    nested_splitting_expected_result: list[str],
) -> None:
    result: list[str] = split_text(
        text=sample_text,
        part_size=500,
        count_size=len,
        separators=["\n\n", "."],
    )
    assert result == nested_splitting_expected_result


def test_uses_overlap_splitting_with_long_text(
    sample_text: str,
    overlap_splitting_expected_result: list[str],
) -> None:
    result: list[str] = split_text(
        text=sample_text,
        part_size=500,
        count_size=len,
        separators=["\n", "."],
        part_overlap_size=100,
    )
    assert result == overlap_splitting_expected_result
