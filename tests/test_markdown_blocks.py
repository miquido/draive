from draive import markdown_block, markdown_blocks


def test_returns_none_with_empty():
    source: str = ""

    assert markdown_block(source=source) is None


def test_info_returns_none_with_empty():
    source: str = ""

    assert markdown_block("test", source=source) is None


def test_returns_none_without_block():
    source: str = "Lorem ipsum"

    assert markdown_block(source=source) is None


def test_info_returns_none_without_block():
    source: str = "Lorem ipsum"

    assert markdown_block("test", source=source) is None


def test_returns_none_with_one_line_block():
    source: str = "`Lorem ipsum`"

    assert markdown_block(source=source) is None


def test_info_returns_none_with_one_line_block():
    source: str = "`test Lorem ipsum`"

    assert markdown_block("test", source=source) is None


def test_returns_content_with_no_info_block():
    source: str = "``` Lorem ipsum```"

    assert markdown_block(source=source) == "Lorem ipsum"


def test_returns_content_with_any_info_block():
    source: str = "```other Lorem ipsum```"

    assert markdown_block(source=source) == "Lorem ipsum"


def test_info_returns_none_with_no_info_block():
    source: str = "``` Lorem ipsum```"

    assert markdown_block("test", source=source) is None


def test_info_returns_none_with_other_info_block():
    source: str = "```other Lorem ipsum```"

    assert markdown_block("test", source=source) is None


def test_returns_none_with_block_closing():
    source: str = "Lorem ipsum```"

    assert markdown_block(source=source) is None


def test_info_returns_none_with_block_closing():
    source: str = "Lorem ipsum```"

    assert markdown_block("test", source=source) is None


def test_info_returns_none_with_missing_info():
    source: str = "```Lorem ipsum```"

    assert markdown_block("test", source=source) is None


def test_returns_none_without_block_closing():
    source: str = "``` Lorem ipsum"

    assert markdown_block(source=source) is None


def test_info_returns_none_without_block_closing():
    source: str = "```test Lorem ipsum"

    assert markdown_block("test", source=source) is None


def test_returns_none_with_malformed_block_opening():
    source: str = "`` Lorem ipsum```"

    assert markdown_block("test", source=source) is None


def test_info_returns_none_with_malformed_block_opening():
    source: str = "``test Lorem ipsum```"

    assert markdown_block("test", source=source) is None


def test_returns_none_with_malformed_block_closing():
    source: str = "```test Lorem ipsum``"

    assert markdown_block(source=source) is None


def test_info_returns_none_with_malformed_block_closing():
    source: str = "```test Lorem ipsum``"

    assert markdown_block("test", source=source) is None


def test_returns_content_with_valid_block():
    source: str = "``` Lorem ipsum```"

    assert markdown_block(source=source) == "Lorem ipsum"


def test_info_returns_content_with_valid_block():
    source: str = "```test Lorem ipsum```"

    assert markdown_block("test", source=source) == "Lorem ipsum"


def test_returns_content_with_surrounded_block():
    source: str = "Lorem``` Lorem ipsum``` ipsum"

    assert markdown_block(source=source) == "Lorem ipsum"


def test_info_returns_content_with_surrounded_block():
    source: str = "Lorem```test Lorem ipsum``` ipsum"

    assert markdown_block("test", source=source) == "Lorem ipsum"


def test_returns_first_content_with_multiple_blocks():
    source: str = "``` Lorem ipsum``` ``` Other```"

    assert markdown_block(source=source) == "Lorem ipsum"


def test_info_returns_first_content_with_multiple_blocks():
    source: str = "```test Lorem ipsum``` ```test Other```"

    assert markdown_block("test", source=source) == "Lorem ipsum"


def test_returns_outer_content_with_nested_blocks():
    source: str = "```outer Other ```inner Lorem ipsum``` ```"

    assert markdown_block(source=source) == "Other ```inner Lorem ipsum"


def test_info_returns_outer_content_with_nested_blocks():
    source: str = "```test Other ```test Lorem ipsum``` ```"

    assert markdown_block("test", source=source) == "Other ```test Lorem ipsum"


def test_returns_content_with_extended_block_closing():
    source: str = "``` Lorem`````"

    assert markdown_block(source=source) == "Lorem``"


def test_info_returns_content_with_extended_block_closing():
    source: str = "```test Lorem`````"

    assert markdown_block("test", source=source) == "Lorem``"


def test_returns_content_with_multiple_blocks():
    source: str = "``` Lorem ipsum``` ```test Dolor``` ```more Sit amet```"
    contents: list[str] = list(markdown_blocks(source=source))

    assert contents == ["Lorem ipsum", "Dolor", "Sit amet"]


def test_info_returns_content_with_multiple_blocks():
    source: str = "```test Lorem ipsum``` ```test Dolor``` ```test Sit amet```"
    contents: list[str] = list(markdown_blocks("test", source=source))

    assert contents == ["Lorem ipsum", "Dolor", "Sit amet"]


def test_info_returns_content_with_multiple_blocks_skipping_other_blocks():
    source: str = "```test Lorem ipsum``` ```other Dolor``` ```test Sit amet```"
    contents: list[str] = list(markdown_blocks("test", source=source))

    assert contents == ["Lorem ipsum", "Sit amet"]
