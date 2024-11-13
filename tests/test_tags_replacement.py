from draive import MediaContent, MultimodalContent
from draive.multimodal.text import TextContent


def test_returns_unchanged_with_empty():
    assert (
        MultimodalContent.of().replacing(
            "test",
            replacement="replaced",
        )
        == MultimodalContent.of()
    )

    assert (
        MultimodalContent.of().replacing(
            "test",
            replacement="replaced",
            remove_tags=True,
        )
        == MultimodalContent.of()
    )


def test_returns_unchanged_without_tag():
    assert MultimodalContent.of("Lorem ipsum").replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("Lorem ipsum")

    assert MultimodalContent.of("Lorem ipsum").replacing(
        "test",
        replacement="replaced",
        remove_tags=True,
    ) == MultimodalContent.of("Lorem ipsum")

    assert MultimodalContent.of(
        "Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum",
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum",
    )


def test_returns_unchanged_with_other_tag():
    assert MultimodalContent.of("<other>Lorem ipsum</other>").replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<other>Lorem ipsum</other>")

    assert MultimodalContent.of("<other>Lorem ipsum</other>").replacing(
        "test",
        replacement="replaced",
        remove_tags=True,
    ) == MultimodalContent.of("<other>Lorem ipsum</other>")

    assert MultimodalContent.of(
        "<other>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</other>",
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<other>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</other>",
    )


def test_returns_unchanged_without_closing_tag():
    assert MultimodalContent.of("<test>Lorem ipsum").replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>Lorem ipsum")

    assert MultimodalContent.of("<test>Lorem ipsum").replacing(
        "test",
        replacement="replaced",
        remove_tags=True,
    ) == MultimodalContent.of("<test>Lorem ipsum")

    assert MultimodalContent.of(
        "<test>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum",
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum",
    )


def test_returns_replaced_expanded_with_only_closing_tag():
    assert MultimodalContent.of("Lorem </test> ipsum").replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("Lorem <test>replaced</test> ipsum")

    assert MultimodalContent.of("Lorem </test> ipsum").replacing(
        "test",
        replacement="replaced",
        remove_tags=True,
    ) == MultimodalContent.of("Lorem replaced ipsum")

    assert MultimodalContent.of(
        "Lorem </test> ",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum",
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "Lorem <test>replaced</test> ",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum",
    )


def test_returns_replaced_expanded_with_reversed_tags():
    assert MultimodalContent.of("</test>Lorem <test> ipsum").replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test>Lorem <test> ipsum")

    assert MultimodalContent.of("</test>Lorem <test> ipsum").replacing(
        "test",
        replacement="replaced",
        remove_tags=True,
    ) == MultimodalContent.of("replacedLorem <test> ipsum")

    assert MultimodalContent.of(
        "</test>Lorem <test> ",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum",
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>replaced</test>Lorem <test> ",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum",
    )


def test_returns_replaced_expanded_with_malformed_opening_tag():
    assert MultimodalContent.of("<testx>Lorem ipsum</test>").replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<testx>Lorem ipsum<test>replaced</test>")

    assert MultimodalContent.of("<testx>Lorem ipsum</test>").replacing(
        "test",
        replacement="replaced",
        remove_tags=True,
    ) == MultimodalContent.of("<testx>Lorem ipsumreplaced")

    assert MultimodalContent.of(
        "<testx>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</test>",
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<testx>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum<test>replaced</test>",
    )


def test_returns_unchanged_with_malformed_closing_tag():
    assert MultimodalContent.of("<test>Lorem ipsum</testx>").replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>Lorem ipsum</testx>")

    assert MultimodalContent.of("<test>Lorem ipsum</testx>").replacing(
        "test",
        replacement="replaced",
        remove_tags=True,
    ) == MultimodalContent.of("<test>Lorem ipsum</testx>")

    assert MultimodalContent.of(
        "<test>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</testx>",
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</testx>",
    )


def test_returns_replaced_with_valid_tag():
    assert MultimodalContent.of("<test>Lorem ipsum</test>").replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test>")

    assert MultimodalContent.of("<test>Lorem ipsum</test>").replacing(
        "test",
        replacement="replaced",
        remove_tags=True,
    ) == MultimodalContent.of("replaced")

    assert MultimodalContent.of(
        "<test>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</test>",
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>replaced</test>",
    )


def test_returns_replaced_inner_with_multiple_nested_tags():
    assert MultimodalContent.of("<test>Other<test>Lorem ipsum</test></test>").replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>Other<test>replaced</test><test>replaced</test>")

    assert MultimodalContent.of("<test>Other<test>Lorem ipsum</test></test>").replacing(
        "test",
        replacement="replaced",
        remove_tags=True,
    ) == MultimodalContent.of("<test>Otherreplacedreplaced")

    assert MultimodalContent.of(
        "<test>Other<test>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</test></test>",
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>Other<test>replaced</test><test>replaced</test>",
    )


def test_returns_replaced_with_tag_spit_into_multiple_parts():
    assert MultimodalContent.of(
        TextContent(text="<te", meta={"meta": 1}),
        TextContent(text="st>Lorem ipsum</te", meta={"meta": 2}),
        TextContent(text="st>", meta={"meta": 3}),
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        TextContent(text="<te", meta={"meta": 1}),
        TextContent(text="st>", meta={"meta": 2}),
        "replaced",
        TextContent(text="</te", meta={"meta": 2}),
        TextContent(text="st>", meta={"meta": 3}),
    )

    assert MultimodalContent.of(
        TextContent(text="<te", meta={"meta": 1}),
        TextContent(text="st>Lorem ipsum</te", meta={"meta": 2}),
        TextContent(text="st>", meta={"meta": 3}),
    ).replacing(
        "test",
        replacement="replaced",
        remove_tags=True,
    ) == MultimodalContent.of(
        "replaced",
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
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        TextContent(text="<", meta={"meta": 1}),
        TextContent(text="test", meta={"meta": 2}),
        TextContent(text=">", meta={"meta": 3}),
        "replaced",
        TextContent(text="<", meta={"meta": 5}),
        TextContent(text="/", meta={"meta": 6}),
        TextContent(text="te", meta={"meta": 7}),
        TextContent(text="st>", meta={"meta": 8}),
    )


def test_returns_replaced_with_surrounded_tag():
    assert MultimodalContent.of("Lorem<test>Lorem ipsum</test>ipsum").replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("Lorem<test>replaced</test>ipsum")

    assert MultimodalContent.of("Lorem<test>Lorem ipsum</test>ipsum").replacing(
        "test",
        replacement="replaced",
        remove_tags=True,
    ) == MultimodalContent.of("Loremreplacedipsum")

    assert MultimodalContent.of(
        MediaContent.url("http://image", mime_type="image/png"),
        "Lorem<test>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</test>ipsum",
        MediaContent.url("http://image", mime_type="image/png"),
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        MediaContent.url("http://image", mime_type="image/png"),
        "Lorem<test>replaced</test>ipsum",
        MediaContent.url("http://image", mime_type="image/png"),
    )


def test_returns_all_replaced_with_multiple_tags():
    assert MultimodalContent.of("<test>Lorem ipsum</test><test>Other</test>").replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test><test>replaced</test>")

    assert MultimodalContent.of("<test>Lorem ipsum</test><test>Other</test>").replacing(
        "test",
        replacement="replaced",
        remove_tags=True,
    ) == MultimodalContent.of("replacedreplaced")

    assert MultimodalContent.of(
        "<test>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</test><test>Other</test>",
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>replaced</test><test>replaced</test>",
    )


def test_returns_replaced_nested_with_nested_tags():
    assert MultimodalContent.of("<other>Other<test>Lorem ipsum</test></other>").replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<other>Other<test>replaced</test></other>")

    assert MultimodalContent.of("<other>Other<test>Lorem ipsum</test></other>").replacing(
        "test",
        replacement="replaced",
        remove_tags=True,
    ) == MultimodalContent.of("<other>Otherreplaced</other>")

    assert MultimodalContent.of(
        "<other>Other<test>Lorem",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</test></other>",
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<other>Other<test>replaced</test></other>",
    )


def test_returns_replaced_content_with_fake_tags():
    assert MultimodalContent.of("<test>Lorem<ipsum</test>").replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test>")

    assert MultimodalContent.of("<te<test>Lorem<ipsum</test>").replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<te<test>replaced</test>")

    assert MultimodalContent.of("<test>Lorem<ipsum</test>").replacing(
        "test",
        replacement="replaced",
        remove_tags=True,
    ) == MultimodalContent.of("replaced")

    assert MultimodalContent.of("<te<test>Lorem<ipsum</test>").replacing(
        "test",
        replacement="replaced",
        remove_tags=True,
    ) == MultimodalContent.of("<tereplaced")

    assert MultimodalContent.of(
        "<test>Lorem<",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</test>",
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>replaced</test>",
    )

    assert MultimodalContent.of(
        "<te<test>Lorem<",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</test>",
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<te<test>replaced</test>",
    )


def test_returns_replaced_nested_with_other_tags():
    assert MultimodalContent.of(
        "<other>Other<more><test>Lorem</more>ipsum</test></other>"
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<other>Other<more><test>replaced</test></other>")

    assert MultimodalContent.of(
        "<other>Other<more><test>Lorem</more>ipsum</test></other>"
    ).replacing(
        "test",
        replacement="replaced",
        remove_tags=True,
    ) == MultimodalContent.of("<other>Other<more>replaced</other>")

    assert MultimodalContent.of(
        "<other>Other<more><test>Lorem</more>",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</test></other>",
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<other>Other<more><test>replaced</test></other>",
    )


def test_returns_replaced_with_multipart_replacement():
    assert MultimodalContent.of(
        "<other>Other<more><test>Lorem</more>ipsum</test></other>"
    ).replacing(
        "test",
        replacement=MultimodalContent.of(
            "replaced-start",
            MediaContent.url("http://replacement", mime_type="image/png"),
            "replaced-end",
        ),
    ) == MultimodalContent.of(
        "<other>Other<more><test>replaced-start",
        MediaContent.url("http://replacement", mime_type="image/png"),
        "replaced-end</test></other>",
    )

    assert MultimodalContent.of(
        "<other>Other<more><test>Lorem</more>ipsum</test></other>"
    ).replacing(
        "test",
        replacement=MultimodalContent.of(
            "replaced-start",
            MediaContent.url("http://replacement", mime_type="image/png"),
            "replaced-end",
        ),
        remove_tags=True,
    ) == MultimodalContent.of(
        "<other>Other<more>replaced-start",
        MediaContent.url("http://replacement", mime_type="image/png"),
        "replaced-end</other>",
    )

    assert MultimodalContent.of(
        "<other>Other<more><test>Lorem</more>",
        MediaContent.url("http://image", mime_type="image/png"),
        "ipsum</test></other>",
    ).replacing(
        "test",
        replacement=MultimodalContent.of(
            "replaced-start",
            MediaContent.url("http://replacement", mime_type="image/png"),
            "replaced-end",
        ),
    ) == MultimodalContent.of(
        "<other>Other<more><test>replaced-start",
        MediaContent.url("http://replacement", mime_type="image/png"),
        "replaced-end</test></other>",
    )
