from draive import MultimodalContent, MultimodalTag, TextContent
from draive.resources import ResourceReference


def test_returns_none_with_empty():
    assert MultimodalContent.empty.tag("test") is None


def test_returns_none_without_tag():
    assert MultimodalContent.of("Lorem ipsum").tag("test") is None

    assert (
        MultimodalContent.of(
            "Lorem",
            ResourceReference.of(uri="http://image", mime_type="image/png"),
            "ipsum",
        ).tag("test")
        is None
    )


def test_returns_none_with_other_tag():
    assert MultimodalContent.of("<other>Lorem ipsum</other>").tag("test") is None

    assert (
        MultimodalContent.of(
            "<other>Lorem",
            ResourceReference.of(uri="http://image", mime_type="image/png"),
            "ipsum</other>",
        ).tag("test")
        is None
    )


def test_returns_none_with_closing_tag():
    assert MultimodalContent.of("Lorem ipsum</test>").tag("test") is None

    assert (
        MultimodalContent.of(
            "Lorem",
            ResourceReference.of(uri="http://image", mime_type="image/png"),
            "ipsum</test>",
        ).tag("test")
        is None
    )


def test_returns_empty_with_closed_tag():
    tag = MultimodalContent.of("Lorem ipsum<test/>").tag("test")
    assert tag is not None
    assert tag.name == "test"
    assert tag.content == MultimodalContent.empty

    tag = MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "ipsum<test/>",
    ).tag("test")
    assert tag is not None
    assert tag.name == "test"
    assert tag.content == MultimodalContent.empty


def test_returns_none_with_unclosed_tag():
    assert MultimodalContent.of("</test>Lorem ipsum<test>").tag("test") is None

    assert (
        MultimodalContent.of(
            "Lorem",
            ResourceReference.of(uri="http://image", mime_type="image/png"),
            "</test>ipsum<test>",
        ).tag("test")
        is None
    )


def test_unclosed_tag_does_not_block_other_tags():
    tags = list(MultimodalContent.of("<broken>value", "<strategy>plan</strategy>").tags("strategy"))

    assert len(tags) == 1
    assert tags[0].content == MultimodalContent.of("plan")

    tags = list(
        MultimodalContent.of(
            "<broken>value",
            ResourceReference.of(uri="http://image", mime_type="image/png"),
            "<strategy>plan</strategy>",
        ).tags("strategy")
    )

    assert len(tags) == 1
    assert tags[0].content == MultimodalContent.of("plan")


def test_tag_to_str_escapes_attributes():
    tag = MultimodalTag.of(
        "inner",
        name="sample",
        meta={
            "attr": '<>&"',
        },
    )

    expected = '<sample attr="&lt;&gt;&amp;{}">inner</sample>'.format('\\"')
    assert tag.to_str() == expected


def test_parses_tag_spanning_multiple_text_parts():
    content = MultimodalContent(
        parts=(
            TextContent.of("<test", meta={"segment": "a"}),
            TextContent.of(' attr="value">payload', meta={"segment": "a"}),
            TextContent.of("</test>", meta={"segment": "a"}),
        ),
    )

    tag = content.tag("test")
    assert tag is not None
    assert tag.content.to_str() == "payload"


def test_returns_none_with_incomplete_opening_tag():
    assert MultimodalContent.of("<test>Lorem ipsum").tag("test") is None

    assert (
        MultimodalContent.of(
            "Lorem",
            ResourceReference.of(uri="http://image", mime_type="image/png"),
            "<test>ipsum",
        ).tag("test")
        is None
    )


def test_returns_none_with_mismatched_opening_tag():
    assert MultimodalContent.of("<testx>Lorem ipsum</test>").tag("test") is None

    assert (
        MultimodalContent.of(
            "Lorem",
            ResourceReference.of(uri="http://image", mime_type="image/png"),
            "<testx>ipsum</test>",
        ).tag("test")
        is None
    )


def test_returns_none_with_mismatched_closing_tag():
    assert MultimodalContent.of("<test>Lorem ipsum</testx>").tag("test") is None

    assert (
        MultimodalContent.of(
            "Lorem",
            ResourceReference.of(uri="http://image", mime_type="image/png"),
            "<test>ipsum</testx>",
        ).tag("test")
        is None
    )


def test_returns_tag_with_valid_content():
    tag = MultimodalContent.of("<test>Lorem ipsum</test>").tag("test")
    assert tag is not None
    assert tag.name == "test"
    assert tag.content == MultimodalContent.of("Lorem ipsum")

    tag = MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>ipsum</test>",
    ).tag("test")
    assert tag is not None
    assert tag.name == "test"
    assert tag.content == MultimodalContent.of("ipsum")


def test_returns_first_tag_with_multiple():
    tag = MultimodalContent.of("Lorem<test>Lorem ipsum</test>ipsum").tag("test")
    assert tag is not None
    assert tag.name == "test"
    assert tag.content == MultimodalContent.of("Lorem ipsum")

    tag = MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>ipsum</test>",
        "More",
    ).tag("test")
    assert tag is not None
    assert tag.name == "test"
    assert tag.content == MultimodalContent.of("ipsum")


def test_returns_tag_with_attributes():
    tag = MultimodalContent.of('<test key="value">Lorem ipsum</test>').tag("test")
    assert tag is not None
    assert tag.name == "test"
    assert tag.content == MultimodalContent.of("Lorem ipsum")
    assert tag.meta.get("key") == "value"

    tag = MultimodalContent.of('<test key="value" next="more">Lorem ipsum</test>').tag("test")
    assert tag is not None
    assert tag.name == "test"
    assert tag.content == MultimodalContent.of("Lorem ipsum")
    assert tag.meta.get("key") == "value"
    assert tag.meta.get("next") == "more"

    tag = MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        '<test key="value">ipsum</test>',
    ).tag("test")
    assert tag is not None
    assert tag.name == "test"
    assert tag.content == MultimodalContent.of("ipsum")
    assert tag.meta.get("key") == "value"

    tag = MultimodalContent.of(
        '<test key="value">Lorem',
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "ipsum</test>",
    ).tag("test")
    assert tag is not None
    assert tag.name == "test"
    assert tag.content == MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "ipsum",
    )
    assert tag.meta.get("key") == "value"


def test_handles_invalid_attribute_formats():
    assert MultimodalContent.of('<test key=value">Lorem ipsum</test>').tag("test") is None

    assert MultimodalContent.of('<testkey="value">Lorem ipsum</test>').tag("test") is None

    assert MultimodalContent.of('<test key="va\nlue">Lorem ipsum</test>').tag("test") is None

    assert MultimodalContent.of("<testkey=>Lorem ipsum</test>").tag("test") is None


def test_returns_all_tags():
    tags = list(MultimodalContent.of("<test>Lorem ipsum</test><test>Other</test>").tags("test"))

    assert len(tags) == 2
    assert tags[0].name == "test"
    assert tags[0].content == MultimodalContent.of("Lorem ipsum")
    assert tags[1].name == "test"
    assert tags[1].content == MultimodalContent.of("Other")

    tags = list(
        MultimodalContent.of(
            "<test>Lorem",
            ResourceReference.of(uri="http://image1", mime_type="image/png"),
            "ipsum</test><test>Other</test>",
        ).tags("test")
    )

    assert len(tags) == 2
    assert tags[0].name == "test"
    assert tags[0].content == MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image1", mime_type="image/png"),
        "ipsum",
    )
    assert tags[1].name == "test"
    assert tags[1].content == MultimodalContent.of("Other")


def test_returns_nested_tags_with_same_name():
    tags = list(MultimodalContent.of("<test>Other<test>Lorem ipsum</test></test>").tags("test"))

    assert [tag.name for tag in tags] == ["test", "test"]
    assert tags[0].content == MultimodalContent.of("Other", "<test>Lorem ipsum</test>")
    assert tags[1].content == MultimodalContent.of("Lorem ipsum")

    tags = list(
        MultimodalContent.of(
            "<test>Other",
            ResourceReference.of(uri="http://image", mime_type="image/png"),
            "<test>ipsum</test></test>",
        ).tags("test")
    )

    assert [tag.name for tag in tags] == ["test", "test"]
    assert tags[0].content == MultimodalContent.of(
        "Other",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>ipsum</test>",
    )
    assert tags[1].content == MultimodalContent.of("ipsum")


def test_returns_tag_with_fake_content():
    tag = MultimodalContent.of("<test>Lorem<ipsum</test>").tag("test")
    assert tag is not None
    assert tag.name == "test"
    assert tag.content == MultimodalContent.of("Lorem<ipsum")

    tag = MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>ipsum<other</test>",
    ).tag("test")
    assert tag is not None
    assert tag.name == "test"
    assert tag.content == MultimodalContent.of("ipsum<other")


def test_returns_complex_with_multimodal_content():
    tags = list(
        MultimodalContent.of(
            "Lorem",
            ResourceReference.of(uri="http://image1", mime_type="image/png"),
            "<test>Lorem ipsum</test>",
            ResourceReference.of(uri="http://image2", mime_type="image/png"),
            "Dolor",
            "<test>Sit amet</test>",
        ).tags("test")
    )

    assert len(tags) == 2
    assert tags[0].name == "test"
    assert tags[0].content == MultimodalContent.of("Lorem ipsum")
    assert tags[1].name == "test"
    assert tags[1].content == MultimodalContent.of("Sit amet")

    tags = list(
        MultimodalContent.of(
            "Lorem",
            ResourceReference.of(uri="http://image1", mime_type="image/png"),
            "<test>Lorem",
            ResourceReference.of(uri="http://image2", mime_type="image/png"),
            "ipsum</test>",
            "Dolor",
            "<test>Sit amet</test>",
        ).tags("test")
    )

    assert len(tags) == 2
    assert tags[0].name == "test"
    assert tags[0].content == MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image2", mime_type="image/png"),
        "ipsum",
    )
    assert tags[1].name == "test"
    assert tags[1].content == MultimodalContent.of("Sit amet")

    tags = list(
        MultimodalContent.of(
            "Lorem",
            ResourceReference.of(uri="http://image1", mime_type="image/png"),
            "<test>Lorem",
            ResourceReference.of(uri="http://image2", mime_type="image/png"),
            "ipsum</test>Dolor<test>Sit",
            ResourceReference.of(uri="http://image3", mime_type="image/png"),
            "amet</test>",
        ).tags("test")
    )

    assert len(tags) == 2
    assert tags[0].name == "test"
    assert tags[0].content == MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image2", mime_type="image/png"),
        "ipsum",
    )
    assert tags[1].name == "test"
    assert tags[1].content == MultimodalContent.of(
        "Sit",
        ResourceReference.of(uri="http://image3", mime_type="image/png"),
        "amet",
    )


def test_handles_empty_tags():
    tag = MultimodalContent.of("<test></test>").tag("test")
    assert tag is not None
    assert tag.name == "test"
    assert tag.content == MultimodalContent.empty


def test_handles_whitespace_only_tags():
    tag = MultimodalContent.of("<test>   \n\t  </test>").tag("test")
    assert tag is not None
    assert tag.name == "test"
    assert tag.content == MultimodalContent.of("   \n\t  ")


def test_handles_special_characters_in_attributes():
    tag = MultimodalContent.of('<test special="value">\ntest\n</test>').tag("test")
    assert tag is not None
    assert tag.name == "test"
    assert tag.content == MultimodalContent.of("\ntest\n")
    assert tag.meta.get("special") == "value"


def test_handles_escaped_quotes_in_attributes():
    tag = MultimodalContent.of('<test quote="\\"quoted\\"">content</test>').tag("test")
    assert tag is not None
    assert tag.name == "test"
    assert tag.content == MultimodalContent.of("content")
    assert tag.meta.get("quote") == '"quoted"'


def test_handles_self_closing_tags():
    tags = list(MultimodalContent.of("<test/><test/><test/>").tags("test"))

    assert len(tags) == 3
    for tag in tags:
        assert tag.content == MultimodalContent.empty

    tags = list(MultimodalContent.of("<test/>text<test>content</test><test/>").tags("test"))

    assert len(tags) == 3
    assert tags[0].content == MultimodalContent.empty
    assert tags[1].content == MultimodalContent.of("content")
    assert tags[2].content == MultimodalContent.empty


def test_handles_self_closing_tags_with_attributes():
    content = MultimodalContent.of('<img src="path/to.png" alt="logo"/>')

    tags = list(content.tags("img"))

    assert len(tags) == 1
    assert tags[0].content == MultimodalContent.empty
    assert tags[0].meta.get("src") == "path/to.png"
    assert tags[0].meta.get("alt") == "logo"


def test_self_closing_tag_with_attributes_mixed_text():
    content = MultimodalContent.of(
        "Intro ",
        '<img src="a/b.svg"/>',
        " middle ",
        "<test>keep</test>",
        " end",
    )

    img_tags = list(content.tags("img"))
    test_tags = list(content.tags("test"))

    assert len(img_tags) == 1
    assert img_tags[0].content == MultimodalContent.empty
    assert img_tags[0].meta.get("src") == "a/b.svg"

    assert len(test_tags) == 1
    assert test_tags[0].content == MultimodalContent.of("keep")


def test_self_closing_tag_with_attributes_and_space_before_slash():
    content = MultimodalContent.of('<img src="hero.png" alt="hero" />')

    tags = list(content.tags("img"))

    assert len(tags) == 1
    assert tags[0].content == MultimodalContent.empty
    assert tags[0].meta.get("src") == "hero.png"
    assert tags[0].meta.get("alt") == "hero"


def test_invalid_self_closing_tag_with_malformed_attributes_is_skipped():
    content = MultimodalContent.of("<img src=hero/>")

    assert content.tag("img") is None

    # Surrounding valid tag should still be found
    content = MultimodalContent.of("<img src=hero/>", '<img src="ok.png"/>')

    tags = list(content.tags("img"))

    assert len(tags) == 1
    assert tags[0].meta.get("src") == "ok.png"


def test_handles_tags_with_numbers():
    tag = MultimodalContent.of("<test123>content</test123>").tag("test123")
    assert tag is not None
    assert tag.name == "test123"
    assert tag.content == MultimodalContent.of("content")


def test_handles_malformed_attribute_syntax():
    assert MultimodalContent.of('<test a="1" b= c d="2">content</test>').tag("test") is None


def test_handles_unclosed_attribute_quotes():
    assert MultimodalContent.of('<test attr="value>content</test>').tag("test") is None


def test_handles_nested_angle_brackets():
    tag = MultimodalContent.of("<test>a < b > c</test>").tag("test")
    assert tag is not None
    assert tag.name == "test"
    assert tag.content == MultimodalContent.of("a < b > c")


def test_handles_multiple_spaces_in_tags():
    assert MultimodalContent.of("<test   key='value'   >content</test>").tag("test") is not None


def test_handles_malformed_closing_tags():
    assert MultimodalContent.of("<test>content</test attr='value'>").tag("test") is None


def test_handles_case_sensitivity():
    assert MultimodalContent.of("<TEST>content</TEST>").tag("test") is None


def test_handles_mixed_media_and_empty_tags():
    tags = list(
        MultimodalContent.of(
            "<test/>",
            ResourceReference.of(uri="http://image1", mime_type="image/png"),
            "<test></test>",
            ResourceReference.of(uri="http://image2", mime_type="image/png"),
            "<test/>",
        ).tags("test")
    )

    assert len(tags) == 3
    assert tags[0].name == "test"
    assert tags[0].content == MultimodalContent.empty
    assert tags[1].name == "test"
    assert tags[1].content == MultimodalContent.empty
    assert tags[2].name == "test"
    assert tags[2].content == MultimodalContent.empty


def test_handles_attribute_without_value():
    assert MultimodalContent.of("<test solo>content</test>").tag("test") is None
