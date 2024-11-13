from draive import MediaContent, MultimodalContent
from draive.multimodal.text import TextContent


def test_returns_none_with_empty():
    assert MultimodalContent.of().extract_first("test") is None


def test_returns_none_without_tag():
    assert MultimodalContent.of("Lorem ipsum").extract_first("test") is None

    assert (
        MultimodalContent.of(
            "Lorem",
            MediaContent.url("http://image", mime_type="image/png"),
            "ipsum",
        ).extract_first("test")
        is None
    )


def test_returns_none_with_other_tag():
    assert MultimodalContent.of("<other>Lorem ipsum</other>").extract_first("test") is None

    assert (
        MultimodalContent.of(
            "<other>Lorem",
            MediaContent.url("http://image", mime_type="image/png"),
            "ipsum</other>",
        ).extract_first("test")
        is None
    )


def test_returns_empty_with_closing_tag():
    assert (
        MultimodalContent.of("Lorem ipsum</test>").extract_first("test") == MultimodalContent.of()
    )

    assert (
        MultimodalContent.of(
            "Lorem",
            MediaContent.url("http://image", mime_type="image/png"),
            "ipsum</test>",
        ).extract_first("test")
        == MultimodalContent.of()
    )


def test_returns_empty_with_reversed_tags():
    assert (
        MultimodalContent.of("</test>Lorem ipsum<test>").extract_first("test")
        == MultimodalContent.of()
    )

    assert (
        MultimodalContent.of(
            "</test>Lorem",
            MediaContent.url("http://image", mime_type="image/png"),
            "ipsum<test>",
        ).extract_first("test")
        == MultimodalContent.of()
    )


def test_returns_none_without_closing_tag():
    assert MultimodalContent.of("<test>Lorem ipsum").extract_first("test") is None

    assert (
        MultimodalContent.of(
            "<test>Lorem",
            MediaContent.url("http://image", mime_type="image/png"),
            "ipsum",
        ).extract_first("test")
        is None
    )


def test_returns_empty_with_malformed_opening_tag():
    assert (
        MultimodalContent.of("<testx>Lorem ipsum</test>").extract_first("test")
        == MultimodalContent.of()
    )

    assert (
        MultimodalContent.of(
            "<testx>Lorem",
            MediaContent.url("http://image", mime_type="image/png"),
            "ipsum</test>",
        ).extract_first("test")
        == MultimodalContent.of()
    )


def test_returns_none_with_malformed_closing_tag():
    assert MultimodalContent.of("<test>Lorem ipsum</testx>").extract_first("test") is None

    assert (
        MultimodalContent.of(
            "<test>Lorem",
            MediaContent.url("http://image", mime_type="image/png"),
            "ipsum</testx>",
        ).extract_first("test")
        is None
    )


def test_returns_content_with_valid_tag():
    assert MultimodalContent.of("<test>Lorem ipsum</test>").extract_first(
        "test"
    ) == MultimodalContent.of("Lorem ipsum")

    assert MultimodalContent.of(
        "<test>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</test>",
    ).extract_first("test") == MultimodalContent.of(
        "Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum",
    )


def test_returns_content_with_tag_spit_into_multiple_parts():
    assert MultimodalContent.of(
        TextContent(text="<te", meta={"meta": 1}),
        TextContent(text="st>Lorem ipsum</te", meta={"meta": 2}),
        TextContent(text="st>", meta={"meta": 3}),
    ).extract_first("test") == MultimodalContent.of(
        TextContent(text="Lorem ipsum", meta={"meta": 2}),
    )

    assert MultimodalContent.of(
        TextContent(text="<", meta={"meta": 1}),
        TextContent(text="test", meta={"meta": 2}),
        TextContent(text=">Lorem", meta={"meta": 3}),
        MediaContent.url("http://image", mime_type="image/png"),
        TextContent(text="ipsum", meta={"meta": 4}),
        TextContent(text="<", meta={"meta": 5}),
        TextContent(text="/", meta={"meta": 6}),
        TextContent(text="te", meta={"meta": 7}),
        TextContent(text="st>", meta={"meta": 8}),
    ).extract_first("test") == MultimodalContent.of(
        TextContent(text="Lorem", meta={"meta": 3}),
        MediaContent.url("http://image", mime_type="image/png"),
        TextContent(text="ipsum", meta={"meta": 4}),
    )


def test_returns_content_with_surrounded_tag():
    assert MultimodalContent.of("Lorem<test>Lorem ipsum</test>ipsum").extract_first(
        "test"
    ) == MultimodalContent.of("Lorem ipsum")

    assert MultimodalContent.of(
        MediaContent.url("http://image", mime_type="image/png"),
        "Lorem<test>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</test>ipsum",
        MediaContent.url("http://image", mime_type="image/png"),
    ).extract_first("test") == MultimodalContent.of(
        "Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum",
    )


def test_returns_content_with_opening_tag_containing_extras():
    assert MultimodalContent.of("<test extra items=here>Lorem ipsum</test>").extract_first(
        "test"
    ) == MultimodalContent.of("Lorem ipsum")

    assert MultimodalContent.of(
        "<test extra items=here>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</test>",
    ).extract_first("test") == MultimodalContent.of(
        "Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum",
    )


def test_returns_first_content_with_multiple_tags():
    assert MultimodalContent.of("<test>Lorem ipsum</test><test>Other</test>").extract_first(
        "test"
    ) == MultimodalContent.of("Lorem ipsum")

    assert MultimodalContent.of(
        "<test>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</test><test>Other</test>",
    ).extract_first("test") == MultimodalContent.of(
        "Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum",
    )


def test_returns_inner_content_with_multiple_nested_tags():
    assert MultimodalContent.of("<test>Other<test>Lorem ipsum</test></test>").extract_first(
        "test"
    ) == MultimodalContent.of("Lorem ipsum")

    assert MultimodalContent.of(
        "<test>Other<test>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</test></test>",
    ).extract_first("test") == MultimodalContent.of(
        "Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum",
    )


def test_returns_nested_content_with_fake_tags():
    assert MultimodalContent.of("<test>Lorem<ipsum</test>").extract_first(
        "test"
    ) == MultimodalContent.of("Lorem<ipsum")

    assert MultimodalContent.of(
        "<test>Lorem<",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</test>",
    ).extract_first("test") == MultimodalContent.of(
        "Lorem<",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum",
    )


def test_returns_nested_content_with_other_tags():
    assert MultimodalContent.of(
        "<other>Other<more><test>Lorem</more>ipsum</test></other>"
    ).extract_first("test") == MultimodalContent.of("Lorem</more>ipsum")

    assert MultimodalContent.of(
        "<other>Other<more><test>Lorem</more>",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</test></other>",
    ).extract_first("test") == MultimodalContent.of(
        "Lorem</more>",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum",
    )


def test_returns_content_with_multiple_tags():
    assert list(
        MultimodalContent.of(
            "<test>Lorem ipsum</test><test>Dolor</test><test>Sit amet</test>"
        ).extract("test")
    ) == [
        MultimodalContent.of("Lorem ipsum"),
        MultimodalContent.of("Dolor"),
        MultimodalContent.of("Sit amet"),
    ]

    assert list(
        MultimodalContent.of(
            "<test>Lorem ",
            MediaContent.url("http://image", mime_type="image/png"),
            "ipsum</test><test>Dolor</test><test>Sit ",
            MediaContent.url("http://image", mime_type="image/png"),
            "amet</test>",
        ).extract("test")
    ) == [
        MultimodalContent.of(
            "Lorem ",
            MediaContent.url("http://image", mime_type="image/png"),
            "ipsum",
        ),
        MultimodalContent.of(
            "Dolor",
        ),
        MultimodalContent.of(
            "Sit ",
            MediaContent.url("http://image", mime_type="image/png"),
            "amet",
        ),
    ]


def test_skips_content_from_different_tags():
    assert list(
        MultimodalContent.of(
            "<test>Lorem ipsum</test><other>Other</other><more><test>Dolor</test>"
        ).extract("test")
    ) == [
        MultimodalContent.of("Lorem ipsum"),
        MultimodalContent.of("Dolor"),
    ]

    assert list(
        MultimodalContent.of(
            "<test>Lorem ",
            MediaContent.url("http://image", mime_type="image/png"),
            "ipsum</test><other>Other</other><more><test>Sit ",
            MediaContent.url("http://image", mime_type="image/png"),
            "amet</test>",
        ).extract("test")
    ) == [
        MultimodalContent.of(
            "Lorem ",
            MediaContent.url("http://image", mime_type="image/png"),
            "ipsum",
        ),
        MultimodalContent.of(
            "Sit ",
            MediaContent.url("http://image", mime_type="image/png"),
            "amet",
        ),
    ]
