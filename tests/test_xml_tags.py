from draive import xml_tag, xml_tags


def test_returns_none_with_empty():
    source: str = ""

    assert xml_tag("test", source=source) is None


def test_returns_none_without_tag():
    source: str = "Lorem ipsum"

    assert xml_tag("test", source=source) is None


def test_returns_none_with_other_tag():
    source: str = "<other>Lorem ipsum</other>"

    assert xml_tag("test", source=source) is None


def test_returns_none_with_closing_tag():
    source: str = "Lorem ipsum</test>"

    assert xml_tag("test", source=source) is None


def test_returns_none_with_reversed_tags():
    source: str = "</test>Lorem ipsum<test>"

    assert xml_tag("test", source=source) is None


def test_returns_none_without_closing_tag():
    source: str = "<test>Lorem ipsum"

    assert xml_tag("test", source=source) is None


def test_returns_none_with_malformed_opening_tag():
    source: str = "<testx>Lorem ipsum</test>"

    assert xml_tag("test", source=source) is None


def test_returns_none_with_malformed_closing_tag():
    source: str = "<test>Lorem ipsum</testx>"

    assert xml_tag("test", source=source) is None


def test_returns_content_with_valid_tag():
    source: str = "<test>Lorem ipsum</test>"

    assert xml_tag("test", source=source) == "Lorem ipsum"


def test_returns_content_with_surrounded_tag():
    source: str = "Lorem<test>Lorem ipsum</test>ipsum"

    assert xml_tag("test", source=source) == "Lorem ipsum"


def test_returns_content_with_opening_tag_containing_extras():
    source: str = "<test extra items=here>Lorem ipsum</test>"

    assert xml_tag("test", source=source) == "Lorem ipsum"


def test_returns_first_content_with_multiple_tags():
    source: str = "<test>Lorem ipsum</test><test>Other</test>"

    assert xml_tag("test", source=source) == "Lorem ipsum"


def test_returns_nested_content_with_multiple_nested_tags():
    source: str = "<test>Other<test>Lorem ipsum</test></test>"

    assert xml_tag("test", source=source) == "Lorem ipsum"


def test_returns_nested_content_with_fake_tags():
    source: str = "<test>Lorem<ipsum</test>"

    assert xml_tag("test", source=source) == "Lorem<ipsum"


def test_returns_nested_content_with_other_tags():
    source: str = "<other>Other<more><test>Lorem</more>ipsum</test></other>"

    assert xml_tag("test", source=source) == "Lorem</more>ipsum"


def test_returns_content_with_multiple_tags():
    source: str = "<test>Lorem ipsum</test><test>Dolor</test><test>Sit amet</test>"
    contents: list[str] = list(xml_tags("test", source=source))

    assert contents == ["Lorem ipsum", "Dolor", "Sit amet"]


def test_skips_content_from_different_tags():
    source: str = "<test>Lorem ipsum</test><other>Other</other><more><test>Dolor</test>"
    contents: list[str] = list(xml_tags("test", source=source))

    assert contents == ["Lorem ipsum", "Dolor"]
