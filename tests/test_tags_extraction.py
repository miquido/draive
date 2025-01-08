from draive import MediaContent, MultimodalContent, MultimodalTagElement


def test_returns_none_with_empty():
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of(),
            tag="test",
        )
        is None
    )


def test_returns_none_without_tag():
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of("Lorem ipsum"),
            tag="test",
        )
        is None
    )

    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of(
                "Lorem",
                MediaContent.url("http://image", media="image/png"),
                "ipsum",
            ),
            tag="test",
        )
        is None
    )


def test_returns_none_with_other_tag():
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of("<other>Lorem ipsum</other>"),
            tag="test",
        )
        is None
    )

    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of(
                "<other>Lorem",
                MediaContent.url("http://image", media="image/png"),
                "ipsum</other>",
            ),
            tag="test",
        )
        is None
    )


def test_returns_none_with_closing_tag():
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of("Lorem ipsum</test>"),
            tag="test",
        )
        is None
    )

    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of(
                "Lorem",
                MediaContent.url("http://image", media="image/png"),
                "ipsum</test>",
            ),
            tag="test",
        )
        is None
    )


def test_returns_empty_with_closed_tag():
    assert MultimodalTagElement.parse_first(
        MultimodalContent.of("Lorem ipsum<test/>"),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of(),
    )

    assert MultimodalTagElement.parse_first(
        MultimodalContent.of(
            "Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum<test/>",
        ),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of(),
    )


def test_returns_none_with_reversed_tags():
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of("</test>Lorem ipsum<test>"),
            tag="test",
        )
        is None
    )

    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of(
                "</test>Lorem",
                MediaContent.url("http://image", media="image/png"),
                "ipsum<test>",
            ),
            tag="test",
        )
        is None
    )


def test_returns_none_without_closing_tag():
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of("<test>Lorem ipsum"),
            tag="test",
        )
        is None
    )

    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of(
                "<test>Lorem",
                MediaContent.url("http://image", media="image/png"),
                "ipsum",
            ),
            tag="test",
        )
        is None
    )


def test_returns_none_with_malformed_opening_tag():
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of("<testx>Lorem ipsum</test>"),
            tag="test",
        )
        is None
    )

    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of(
                "<testx>Lorem",
                MediaContent.url("http://image", media="image/png"),
                "ipsum</test>",
            ),
            tag="test",
        )
        is None
    )


def test_returns_none_with_malformed_closing_tag():
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of("<test>Lorem ipsum</testx>"),
            tag="test",
        )
        is None
    )

    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of(
                "<test>Lorem",
                MediaContent.url("http://image", media="image/png"),
                "ipsum</testx>",
            ),
            tag="test",
        )
        is None
    )


def test_returns_content_with_valid_tag():
    assert MultimodalTagElement.parse_first(
        MultimodalContent.of("<test>Lorem ipsum</test>"),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of("Lorem ipsum"),
    )

    assert MultimodalTagElement.parse_first(
        MultimodalContent.of(
            "<test>Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test>",
        ),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of(
            "Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum",
        ),
    )


def test_returns_content_with_surrounded_tag():
    assert MultimodalTagElement.parse_first(
        MultimodalContent.of("Lorem<test>Lorem ipsum</test>ipsum"),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of("Lorem ipsum"),
    )

    assert MultimodalTagElement.parse_first(
        MultimodalContent.of(
            MediaContent.url("http://image", media="image/png"),
            "Lorem<test>Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test>ipsum",
            MediaContent.url("http://image", media="image/png"),
        ),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of(
            "Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum",
        ),
    )


def test_returns_content_with_opening_tag_containing_attributes():
    assert MultimodalTagElement.parse_first(
        MultimodalContent.of('Lorem<test key="value">Lorem ipsum</test>ipsum'),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of("Lorem ipsum"),
        attributes={"key": "value"},
    )

    assert MultimodalTagElement.parse_first(
        MultimodalContent.of('Lorem<test key="value" next="more">Lorem ipsum</test>ipsum'),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of("Lorem ipsum"),
        attributes={"key": "value", "next": "more"},
    )

    assert MultimodalTagElement.parse_first(
        MultimodalContent.of(
            'Lorem<test key="value" next="more" spaces="s p a ce" >Lorem ipsum</test>ipsum'
        ),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of("Lorem ipsum"),
        attributes={"key": "value", "next": "more", "spaces": "s p a ce"},
    )

    assert MultimodalTagElement.parse_first(
        MultimodalContent.of(
            'Lorem<test key="value" escaping="\\"escape" spaces="s p a ce" >Lorem ipsum</test>ipsum'
        ),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of("Lorem ipsum"),
        attributes={"key": "value", "escaping": '"escape', "spaces": "s p a ce"},
    )
    assert MultimodalTagElement.parse_first(
        MultimodalContent.of(
            MediaContent.url("http://image", media="image/png"),
            'Lorem<test key="value">Lorem',
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test>ipsum",
            MediaContent.url("http://image", media="image/png"),
        ),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of(
            "Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum",
        ),
        attributes={"key": "value"},
    )


def test_returns_none_with_opening_tag_containing_malformed_attributes():
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of('Lorem<test key=value">Lorem ipsum</test>ipsum'),
            tag="test",
        )
        is None
    )
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of('Lorem<testkey="value">Lorem ipsum</test>ipsum'),
            tag="test",
        )
        is None
    )
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of('Lorem<test key="va\nlue">Lorem ipsum</test>ipsum'),
            tag="test",
        )
        is None
    )
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of("Lorem<testkey=>Lorem ipsum</test>ipsum"),
            tag="test",
        )
        is None
    )


def test_returns_first_content_with_multiple_tags():
    assert MultimodalTagElement.parse_first(
        MultimodalContent.of("<test>Lorem ipsum</test><test>Other</test>"),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of("Lorem ipsum"),
    )

    assert MultimodalTagElement.parse_first(
        MultimodalContent.of(
            "<test>Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test><test>Other</test>",
        ),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of(
            "Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum",
        ),
    )


def test_returns_outer_content_with_multiple_nested_tags():
    assert MultimodalTagElement.parse_first(
        MultimodalContent.of("<test>Other<test>Lorem ipsum</test></test>"),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of("Other<test>Lorem ipsum"),
    )

    assert MultimodalTagElement.parse_first(
        MultimodalContent.of(
            "<test>Other<test>Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test></test>",
        ),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of(
            "Other<test>Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum",
        ),
    )


def test_returns_nested_content_with_fake_tags():
    assert MultimodalTagElement.parse_first(
        MultimodalContent.of("<test>Lorem<ipsum</test>"),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of("Lorem<ipsum"),
    )

    assert MultimodalTagElement.parse_first(
        MultimodalContent.of(
            "<test>Lorem<",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test>",
        ),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of(
            "Lorem<",
            MediaContent.url("http://image", media="image/png"),
            "ipsum",
        ),
    )


def test_returns_none_with_nested_in_other_tags():
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of("<other>Other<more><test>Lorem</more>ipsum</test></other>"),
            tag="test",
        )
        is None
    )

    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of(
                "<other>Other<more><test>Lorem</more>",
                MediaContent.url("http://image", media="image/png"),
                "ipsum</test></other>",
            ),
            tag="test",
        )
        is None
    )


def test_returns_content_with_multiple_tags():
    assert list(
        MultimodalTagElement.parse(
            MultimodalContent.of(
                "<test>Lorem ipsum</test><other>Dolor</other><another>Sit amet</another>"
            ),
        )
    ) == [
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of("Lorem ipsum"),
        ),
        MultimodalTagElement(
            name="other",
            content=MultimodalContent.of("Dolor"),
        ),
        MultimodalTagElement(
            name="another",
            content=MultimodalContent.of("Sit amet"),
        ),
    ]

    assert list(
        MultimodalTagElement.parse(
            MultimodalContent.of(
                "<test>Lorem ",
                MediaContent.url("http://image", media="image/png"),
                "ipsum</test><other>Dolor</other><another>Sit ",
                MediaContent.url("http://image", media="image/png"),
                "amet</another>",
            ),
        )
    ) == [
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of(
                "Lorem ",
                MediaContent.url("http://image", media="image/png"),
                "ipsum",
            ),
        ),
        MultimodalTagElement(
            name="other",
            content=MultimodalContent.of(
                "Dolor",
            ),
        ),
        MultimodalTagElement(
            name="another",
            content=MultimodalContent.of(
                "Sit ",
                MediaContent.url("http://image", media="image/png"),
                "amet",
            ),
        ),
    ]


def test_returns_content_with_multiple_filtered_tags():
    assert list(
        MultimodalTagElement.parse(
            MultimodalContent.of("<test>Lorem ipsum</test><test>Dolor</test><test>Sit amet</test>"),
            tag="test",
        )
    ) == [
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of("Lorem ipsum"),
        ),
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of("Dolor"),
        ),
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of("Sit amet"),
        ),
    ]

    assert list(
        MultimodalTagElement.parse(
            MultimodalContent.of(
                "<test>Lorem ",
                MediaContent.url("http://image", media="image/png"),
                "ipsum</test><test>Dolor</test><test>Sit ",
                MediaContent.url("http://image", media="image/png"),
                "amet</test>",
            ),
            tag="test",
        )
    ) == [
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of(
                "Lorem ",
                MediaContent.url("http://image", media="image/png"),
                "ipsum",
            ),
        ),
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of(
                "Dolor",
            ),
        ),
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of(
                "Sit ",
                MediaContent.url("http://image", media="image/png"),
                "amet",
            ),
        ),
    ]


def test_skips_content_from_different_filtered_tags():
    assert list(
        MultimodalTagElement.parse(
            MultimodalContent.of("<test>Lorem ipsum</test><other>Other</other><test>Dolor</test>"),
            tag="test",
        )
    ) == [
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of("Lorem ipsum"),
        ),
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of("Dolor"),
        ),
    ]

    assert list(
        MultimodalTagElement.parse(
            MultimodalContent.of(
                "<test>Lorem ",
                MediaContent.url("http://image", media="image/png"),
                "ipsum</test><other>Other</other><test>Sit ",
                MediaContent.url("http://image", media="image/png"),
                "amet</test>",
            ),
            tag="test",
        )
    ) == [
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of(
                "Lorem ",
                MediaContent.url("http://image", media="image/png"),
                "ipsum",
            ),
        ),
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of(
                "Sit ",
                MediaContent.url("http://image", media="image/png"),
                "amet",
            ),
        ),
    ]


### generated ###


def test_handles_empty_tag_content():
    assert MultimodalTagElement.parse_first(
        MultimodalContent.of("<test></test>"),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of(),
    )


def test_handles_whitespace_only_tag_content():
    assert MultimodalTagElement.parse_first(
        MultimodalContent.of("<test>   \n\t  </test>"),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of("   \n\t  "),
    )


def test_handles_special_characters_in_attributes():
    assert MultimodalTagElement.parse_first(
        MultimodalContent.of('<test special="&quot;\'<>/">\ntest\n</test>'),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of("\ntest\n"),
        attributes={"special": "&quot;'<>/"},
    )


def test_handles_escaped_quotes_in_attributes():
    assert MultimodalTagElement.parse_first(
        MultimodalContent.of('<test quote="\\"quoted\\"">content</test>'),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of("content"),
        attributes={"quote": '"quoted"'},
    )


def test_handles_multiple_consecutive_tags():
    assert list(
        MultimodalTagElement.parse(
            MultimodalContent.of("<test/><test/><test/>"),
            tag="test",
        )
    ) == [
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of(),
        ),
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of(),
        ),
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of(),
        ),
    ]


def test_handles_mixed_self_closing_and_normal_tags():
    assert list(
        MultimodalTagElement.parse(
            MultimodalContent.of("<test/>text<test>content</test><test/>"),
            tag="test",
        )
    ) == [
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of(),
        ),
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of("content"),
        ),
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of(),
        ),
    ]


def test_handles_tag_names_with_numbers():
    assert MultimodalTagElement.parse_first(
        MultimodalContent.of("<test123>content</test123>"),
        tag="test123",
    ) == MultimodalTagElement(
        name="test123",
        content=MultimodalContent.of("content"),
    )


def test_handles_invalid_attribute_formats():
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of('<test a="1" b= c d="2">content</test>'),
            tag="test",
        )
        is None
    )


def test_handles_unclosed_attribute_quotes():
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of('<test attr="value>content</test>'),
            tag="test",
        )
        is None
    )


def test_handles_nested_angle_brackets():
    assert MultimodalTagElement.parse_first(
        MultimodalContent.of("<test>a < b > c</test>"),
        tag="test",
    ) == MultimodalTagElement(
        name="test",
        content=MultimodalContent.of("a < b > c"),
    )


def test_handles_multiple_spaces_in_tags():
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of("<test   key='value'   >content</test>"),
            tag="test",
        )
        is None  # This case should be handled in future
    )


def test_handles_malformed_closing_tags():
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of("<test>content</test attr='value'>"),
            tag="test",
        )
        is None
    )


def test_handles_case_sensitivity():
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of("<TEST>content</TEST>"),
            tag="test",
        )
        is None
    )


def test_handles_mixed_media_and_empty_tags():
    assert list(
        MultimodalTagElement.parse(
            MultimodalContent.of(
                "<test/>",
                MediaContent.url("http://image1", media="image/png"),
                "<test></test>",
                MediaContent.url("http://image2", media="image/png"),
                "<test/>",
            ),
        )
    ) == [
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of(),
        ),
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of(),
        ),
        MultimodalTagElement(
            name="test",
            content=MultimodalContent.of(),
        ),
    ]


def test_handles_attribute_without_value():
    assert (
        MultimodalTagElement.parse_first(
            MultimodalContent.of("<test solo>content</test>"),
            tag="test",
        )
        is None
    )
