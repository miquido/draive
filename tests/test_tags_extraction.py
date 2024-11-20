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
    return  # TODO: support for attributes to be added


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
