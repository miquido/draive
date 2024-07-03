import pytest
from draive.splitters.exhaustive import exhaustive_regex_split_text


@pytest.fixture
def sample_text() -> str:
    with open("tests/data/markdown_sample_text.txt") as file:
        return file.read()


@pytest.fixture
def exhaustive_splitting_expected_result() -> str:
    with open("tests/data/markdown_exhaustive_splitting.txt") as file:
        return file.read()


@pytest.fixture
def exhaustive_overlap_splitting_expected_result() -> str:
    with open("tests/data/markdown_exhaustive_overlap_splitting.txt") as file:
        return file.read()


def test_returns_single_part_when_text_fits_part_size() -> None:
    assert exhaustive_regex_split_text(
        text="0",
        part_size=1,
        count_size=len,
        separators=[],
    ) == ["0"]


def test_fails_when_no_separators_provided() -> None:
    with pytest.raises(ValueError):
        exhaustive_regex_split_text(
            text="0",
            part_size=0,
            count_size=len,
            separators=[],
        )


def test_fails_when_cant_split() -> None:
    with pytest.raises(ValueError):
        exhaustive_regex_split_text(
            text="01",
            part_size=1,
            count_size=len,
            separators=[r"\s"],
        )


def test_strips_whitespaces_with_leading_and_trailing_part_whitespaces() -> None:
    assert exhaustive_regex_split_text(
        text=" 01  23 ",
        part_size=3,
        count_size=len,
        separators=[r"\s"],
    ) == ["01", "23"]


def test_adds_leading_splitters_when_required() -> None:
    assert exhaustive_regex_split_text(
        text="#01#23",
        part_size=3,
        count_size=len,
        separators=["#"],
    ) == ["#01", "#23"]


def test_uses_paragraphs_splitting_with_default_separators() -> None:
    assert exhaustive_regex_split_text(
        text="01\n\n23",
        part_size=4,
        count_size=len,
    ) == ["01", "23"]


def test_uses_newline_splitting_with_default_separators() -> None:
    assert exhaustive_regex_split_text(
        text="01\n23",
        part_size=4,
        count_size=len,
    ) == ["01", "23"]


def test_uses_words_splitting_with_default_separators() -> None:
    assert exhaustive_regex_split_text(
        text="01 23",
        part_size=3,
        count_size=len,
    ) == ["01", "23"]


def test_uses_words_splitting_when_one_separators_provided() -> None:
    assert exhaustive_regex_split_text(
        text="01 23",
        part_size=3,
        count_size=len,
        separators="#",
    ) == ["01", "23"]


def test_splitting_text_when_parts_fit_part_size_exactly() -> None:
    assert exhaustive_regex_split_text(
        text="#ab#cd#ef#gh",
        part_size=3,
        count_size=len,
        separators="#",
    ) == ["#ab", "#cd", "#ef", "#gh"]


def test_splitting_markdown_headers() -> None:
    assert exhaustive_regex_split_text(
        text="# Header1\n\nThis is Header1\n\n# Header2\n\nThis is Header2",
        part_size=30,
        count_size=len,
        separators=[r"#\s.*"],
    ) == [
        "# Header1\n\nThis is Header1",
        "# Header2\n\nThis is Header2",
    ]


def test_splitting_markdown_headers_with_subheaders() -> None:
    assert exhaustive_regex_split_text(
        text="# Title\n\nThis is Header1\n\n# Subtitle\n\nThis is Sub-header",
        part_size=30,
        count_size=len,
        separators=[r"#{1,6}\s.*"],
    ) == [
        "# Title\n\nThis is Header1",
        "# Subtitle\n\nThis is Sub-header",
    ]


def test_splitting_html_tag() -> None:
    assert exhaustive_regex_split_text(
        text='<div class="container">This is a short text</div>',
        part_size=45,
        count_size=len,
        separators=[r"<.+?>"],
    ) == [
        '<div class="container">This is a short text',
        "</div>",
    ]


def test_overlapping_parts_with_parts_similar_sizes() -> None:
    assert exhaustive_regex_split_text(
        text="#a#b#c#d#e#f#g#h#i#j#k#l#m",
        part_size=9,
        count_size=len,
        separators=["#"],
        part_overlap_size=6,
    ) == ["#a#b#c#d", "#c#d#e#f", "#e#f#g#h", "#g#h#i#j", "#i#j#k#l", "#k#l#m"]


def test_overlapping_parts_with_different_sizes() -> None:
    assert exhaustive_regex_split_text(
        text="#1234 56#788#90#ab#cd#ef#g#h ijklm",
        part_size=6,
        count_size=len,
        separators=[r"\s", "#"],
        part_overlap_size=6,
    ) == [
        "#1234",
        "#1234 56",
        "56#788",
        "#788#90",
        "#90#ab",
        "#ab#cd",
        "#cd#ef",
        "#ef#g",
        "#ef#g#h",
        "#h ijklm",
    ]


def test_regex_text_splitting_with_long_text(
    sample_text: str,
    exhaustive_splitting_expected_result: str,
) -> None:
    result: list[str] = exhaustive_regex_split_text(
        text=sample_text,
        part_size=500,
        count_size=len,
        separators=r"#{1,6}\s.*",
    )
    assert "\n\n\n\n".join(result) == exhaustive_splitting_expected_result


def test_uses_overlap_splitting_with_long_text(
    sample_text: str,
    exhaustive_overlap_splitting_expected_result: str,
) -> None:
    result: list[str] = exhaustive_regex_split_text(
        text=sample_text,
        part_size=500,
        count_size=len,
        separators=r"#{1,6}\s.*",
        part_overlap_size=100,
    )
    assert "\n\n\n\n".join(result) == exhaustive_overlap_splitting_expected_result
