from draive import MultimodalContent, TextContent
from draive.resources import ResourceReference


def test_returns_unchanged_with_empty():
    assert (
        MultimodalContent.empty.replacing_tag(
            "test",
            replacement="replaced",
        )
        == MultimodalContent.empty
    )

    assert (
        MultimodalContent.empty.replacing_tag(
            "test",
            replacement="replaced",
            strip_tags=True,
        )
        == MultimodalContent.empty
    )


def test_returns_unchanged_without_tag():
    assert MultimodalContent.of("Lorem ipsum").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("Lorem ipsum")

    assert MultimodalContent.of("Lorem ipsum").replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("Lorem ipsum")

    assert MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "ipsum",
    ).replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "ipsum",
    )


def test_returns_unchanged_with_other_tag():
    assert MultimodalContent.of("<other>Lorem ipsum</other>").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<other>Lorem ipsum</other>")

    assert MultimodalContent.of('<other key="value">Lorem ipsum</other>').replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of('<other key="value">Lorem ipsum</other>')

    assert MultimodalContent.of("<other>Lorem ipsum</other>").replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<other>Lorem ipsum</other>")

    assert MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<other>ipsum</other>",
    ).replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<other>ipsum</other>",
    )


def test_returns_unchanged_without_closing_tag():
    assert MultimodalContent.of("<test>Lorem ipsum").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>Lorem ipsum")

    assert MultimodalContent.of("<test>Lorem ipsum").replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<test>Lorem ipsum")

    assert MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>ipsum",
    ).replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>ipsum",
    )


def test_returns_unchanged_with_only_closing_tag():
    assert MultimodalContent.of("Lorem </test> ipsum").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("Lorem </test> ipsum")

    assert MultimodalContent.of("Lorem </test> ipsum").replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("Lorem </test> ipsum")

    assert MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "</test> ipsum",
    ).replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "</test> ipsum",
    )


def test_returns_unchanged_with_reversed_tags():
    assert MultimodalContent.of("</test>Lorem <test> ipsum").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("</test>Lorem <test> ipsum")

    assert MultimodalContent.of("</test>Lorem <test> ipsum").replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("</test>Lorem <test> ipsum")

    assert MultimodalContent.of(
        "</test>Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test> ipsum",
    ).replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "</test>Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test> ipsum",
    )


def test_returns_unchanged_with_malformed_opening_tag():
    assert MultimodalContent.of("<testx>Lorem ipsum</test>").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<testx>Lorem ipsum</test>")

    assert MultimodalContent.of("<test key=invalid>Lorem ipsum</test>").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<test key=invalid>Lorem ipsum</test>")

    assert MultimodalContent.of("<testx>Lorem ipsum</test>").replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<testx>Lorem ipsum</test>")

    assert MultimodalContent.of(
        "<testx>Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "ipsum</test>",
    ).replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<testx>Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "ipsum</test>",
    )


def test_returns_unchanged_with_malformed_closing_tag():
    assert MultimodalContent.of("<test>Lorem ipsum</testx>").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>Lorem ipsum</testx>")

    assert MultimodalContent.of('<test key="value">Lorem ipsum</testx>').replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of('<test key="value">Lorem ipsum</testx>')

    assert MultimodalContent.of("<test>Lorem ipsum</testx>").replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<test>Lorem ipsum</testx>")

    assert MultimodalContent.of(
        "<test>Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "ipsum</testx>",
    ).replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "ipsum</testx>",
    )


def test_returns_replaced_with_valid_tag():
    assert MultimodalContent.of("<test>Lorem ipsum</test>").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test>")

    assert MultimodalContent.of("<test>Lorem ipsum</test>").replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced")

    assert MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>ipsum</test>",
    ).replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>replaced</test>",
    )


def test_replaces_first_completed_tag_when_nested():
    assert MultimodalContent.of("<test>Other<test>Lorem ipsum</test></test>").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>Other<test>replaced</test></test>")

    assert MultimodalContent.of("<test>Other<test>Lorem ipsum</test></test>").replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<test>Otherreplaced</test>")

    assert MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>Other<test>ipsum</test></test>",
    ).replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>Other<test>replaced</test></test>",
    )


def test_returns_replaced_with_surrounded_tag():
    assert MultimodalContent.of("Lorem<test>Lorem ipsum</test>ipsum").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("Lorem<test>replaced</test>ipsum")

    assert MultimodalContent.of("Lorem<test>Lorem ipsum</test>ipsum").replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("Loremreplacedipsum")

    assert MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>ipsum</test>",
        "More",
    ).replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>replaced</test>",
        "More",
    )


def test_replacing_nested_inside_other_tag():
    assert MultimodalContent.of("<other>Other<test>Lorem ipsum</test></other>").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<other>Other<test>replaced</test></other>")

    assert MultimodalContent.of("<other>Other<test>Lorem ipsum</test></other>").replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<other>Otherreplaced</other>")

    assert MultimodalContent.of(
        "<other>Other",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>ipsum</test></other>",
    ).replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<other>Other",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>replaced</test></other>",
    )


def test_replacing_skips_unclosed_tag():
    assert MultimodalContent.of("<broken>value<test>ok</test>").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<broken>value<test>replaced</test>")

    assert MultimodalContent.of(
        "<broken>value",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>ok</test>",
    ).replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<broken>value",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>replaced</test>",
    )


def test_replacing_tag_spanning_multiple_text_parts():
    content = MultimodalContent(
        parts=(
            TextContent.of("<test", meta={"segment": "a"}),
            TextContent.of(' attr="value">payload', meta={"segment": "a"}),
            TextContent.of("</test>", meta={"segment": "a"}),
        ),
    )

    replaced = content.replacing_tag("test", replacement="done")
    assert replaced.to_str() == '<test attr="value">done</test>'

    stripped = content.replacing_tag(
        "test",
        replacement="done",
        strip_tags=True,
    )
    assert stripped.to_str() == "done"


def test_returns_replaced_content_with_fake_tags():
    assert MultimodalContent.of("<test>Lorem<ipsum</test>").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test>")

    assert MultimodalContent.of("<te<test>Lorem<ipsum</test>").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<te<test>replaced</test>")

    assert MultimodalContent.of("<test>Lorem<ipsum</test>").replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced")

    assert MultimodalContent.of("<te<test>Lorem<ipsum</test>").replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<tereplaced")

    assert MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>ipsum<other</test>",
    ).replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>replaced</test>",
    )


def test_returns_replaced_all_with_exhaustive():
    assert MultimodalContent.of("<test>Lorem ipsum</test><test>Other</test>").replacing_tag(
        "test",
        replacement="replaced",
        exhaustive=True,
    ) == MultimodalContent.of("<test>replaced</test><test>replaced</test>")

    assert MultimodalContent.of("<test>Lorem ipsum</test><test>Other</test>").replacing_tag(
        "test",
        replacement="replaced",
        exhaustive=True,
        strip_tags=True,
    ) == MultimodalContent.of("replacedreplaced")

    assert MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>ipsum</test><test>More</test>",
    ).replacing_tag(
        "test",
        replacement="replaced",
        exhaustive=True,
    ) == MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>replaced</test><test>replaced</test>",
    )


def test_returns_replaced_first_without_exhaustive():
    assert MultimodalContent.of("<test>Lorem ipsum</test><test>Other</test>").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test><test>Other</test>")

    assert MultimodalContent.of("<test>Lorem ipsum</test><test>Other</test>").replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced<test>Other</test>")

    assert MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>ipsum</test><test>More</test>",
    ).replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image", mime_type="image/png"),
        "<test>replaced</test><test>More</test>",
    )


def test_returns_replaced_complex_with_multimodal_content():
    assert MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image1", mime_type="image/png"),
        "<test>Lorem ipsum</test>",
        ResourceReference.of(uri="http://image2", mime_type="image/png"),
        "Dolor",
        "<test>Sit amet</test>",
    ).replacing_tag(
        "test",
        replacement=MultimodalContent.of(
            "replaced",
            ResourceReference.of(uri="http://replaced", mime_type="image/png"),
        ),
    ) == MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image1", mime_type="image/png"),
        "<test>replaced",
        ResourceReference.of(uri="http://replaced", mime_type="image/png"),
        "</test>",
        ResourceReference.of(uri="http://image2", mime_type="image/png"),
        "Dolor",
        "<test>Sit amet</test>",
    )

    assert MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image1", mime_type="image/png"),
        "<test>Lorem ipsum</test>",
        ResourceReference.of(uri="http://image2", mime_type="image/png"),
        "Dolor",
        "<test>Sit amet</test>",
    ).replacing_tag(
        "test",
        replacement=MultimodalContent.of(
            "replaced",
            ResourceReference.of(uri="http://replaced", mime_type="image/png"),
        ),
        exhaustive=True,
    ) == MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image1", mime_type="image/png"),
        "<test>replaced",
        ResourceReference.of(uri="http://replaced", mime_type="image/png"),
        "</test>",
        ResourceReference.of(uri="http://image2", mime_type="image/png"),
        "Dolor",
        "<test>replaced",
        ResourceReference.of(uri="http://replaced", mime_type="image/png"),
        "</test>",
    )

    assert MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image1", mime_type="image/png"),
        "<test>Lorem ipsum</test>",
        ResourceReference.of(uri="http://image2", mime_type="image/png"),
        "Dolor",
        "<test>Sit amet</test>",
    ).replacing_tag(
        "test",
        replacement=MultimodalContent.of(
            "replaced",
            ResourceReference.of(uri="http://replaced", mime_type="image/png"),
        ),
        strip_tags=True,
    ) == MultimodalContent.of(
        "Lorem",
        ResourceReference.of(uri="http://image1", mime_type="image/png"),
        "replaced",
        ResourceReference.of(uri="http://replaced", mime_type="image/png"),
        ResourceReference.of(uri="http://image2", mime_type="image/png"),
        "Dolor",
        "<test>Sit amet</test>",
    )


def test_handles_empty_tags():
    assert MultimodalContent.of("<test></test>").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test>")

    assert MultimodalContent.of("<test></test>").replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced")


def test_handles_whitespace_only_tags():
    assert MultimodalContent.of("<test>   \n\t  </test>").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test>")

    assert MultimodalContent.of("<test>   \n\t  </test>").replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced")


def test_handles_special_characters_in_attributes():
    assert MultimodalContent.of('<test special="&quot;\'<>/">\ntest\n</test>').replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of('<test special="&quot;\'<>/">replaced</test>')

    assert MultimodalContent.of('<test special="&quot;\'<>/">\ntest\n</test>').replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced")


def test_handles_escaped_quotes_in_attributes():
    assert MultimodalContent.of('<test quote="\\"quoted\\"">content</test>').replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of('<test quote="\\"quoted\\"">replaced</test>')

    assert MultimodalContent.of('<test quote="\\"quoted\\"">content</test>').replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced")


def test_handles_self_closing_tags():
    results = list(MultimodalContent.of("<test/><test/><test/>").tags("test"))

    assert len(results) == 3
    for result in results:
        assert result.content == MultimodalContent.empty

    results = list(MultimodalContent.of("<test/>text<test>content</test><test/>").tags("test"))

    assert len(results) == 3
    assert results[0].content == MultimodalContent.empty
    assert results[1].content == MultimodalContent.of("content")
    assert results[2].content == MultimodalContent.empty


def test_handles_tags_with_numbers():
    assert MultimodalContent.of("<test123>content</test123>").replacing_tag(
        "test123",
        replacement="replaced",
    ) == MultimodalContent.of("<test123>replaced</test123>")

    assert MultimodalContent.of("<test123>content</test123>").replacing_tag(
        "test123",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced")


def test_ignores_malformed_attribute_syntax():
    results = list(MultimodalContent.of('<test a="1" b= c d="2">content</test>').tags("test"))
    # Should not parse due to malformed attributes
    assert len(results) == 0

    results = list(MultimodalContent.of('<test attr="value>content</test>').tags("test"))
    # Should not parse due to unclosed quote
    assert len(results) == 0


def test_handles_tags_with_angle_brackets_in_content():
    assert MultimodalContent.of("<test>a < b > c</test>").replacing_tag(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test>")

    assert MultimodalContent.of("<test>a < b > c</test>").replacing_tag(
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced")


def test_ignores_whitespace_in_tag_names():
    results = list(MultimodalContent.of("<test   key='value'   >content</test>").tags("test"))
    # Should parse despite whitespace in tag
    assert len(results) == 1
    assert results[0].meta.get("key") == "value"

    results = list(MultimodalContent.of("<test>content</test attr='value'>").tags("test"))
    # Should not parse due to attributes in closing tag
    assert len(results) == 0


def test_case_sensitive_tag_names():
    results = list(MultimodalContent.of("<TEST>content</TEST>").tags("test"))
    # Should not match due to case difference
    assert len(results) == 0


def test_handles_self_closing_with_exhaustive():
    results = list(
        MultimodalContent.of("<test>inner1</test><test/><test>inner2</test><test/>").tags("test")
    )

    assert len(results) == 4
    assert results[0].content == MultimodalContent.of("inner1")
    assert results[1].content == MultimodalContent.empty
    assert results[2].content == MultimodalContent.of("inner2")
    assert results[3].content == MultimodalContent.empty


def test_ignores_attributes_without_values():
    results = list(MultimodalContent.of("<test solo>content</test>").tags("test"))
    # Should not parse due to attribute without value
    assert len(results) == 0
