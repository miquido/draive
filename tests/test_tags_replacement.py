from draive import MediaContent, MultimodalContent, MultimodalTagElement


def test_returns_unchanged_with_empty():
    assert (
        MultimodalTagElement.replace(
            MultimodalContent.of(),
            tag="test",
            replacement="replaced",
        )
        == MultimodalContent.of()
    )

    assert (
        MultimodalTagElement.replace(
            MultimodalContent.of(),
            tag="test",
            replacement="replaced",
            strip_tags=True,
        )
        == MultimodalContent.of()
    )


def test_returns_unchanged_without_tag():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("Lorem ipsum"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("Lorem ipsum")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("Lorem ipsum"),
        tag="test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("Lorem ipsum")

    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum",
        ),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "Lorem",
        MediaContent.url("http://image", media="image/png"),
        "ipsum",
    )


def test_returns_unchanged_with_other_tag():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<other>Lorem ipsum</other>"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("<other>Lorem ipsum</other>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("<other>Lorem ipsum</other>"),
        "test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<other>Lorem ipsum</other>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "<other>Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</other>",
        ),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<other>Lorem",
        MediaContent.url("http://image", media="image/png"),
        "ipsum</other>",
    )


def test_returns_unchanged_without_closing_tag():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>Lorem ipsum"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>Lorem ipsum")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>Lorem ipsum"),
        tag="test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<test>Lorem ipsum")

    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "<test>Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum",
        ),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>Lorem",
        MediaContent.url("http://image", media="image/png"),
        "ipsum",
    )


def test_returns_unchanged_with_only_closing_tag():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("Lorem </test> ipsum"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("Lorem </test> ipsum")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("Lorem </test> ipsum"),
        tag="test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("Lorem </test> ipsum")

    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "Lorem </test> ",
            MediaContent.url("http://image", media="image/png"),
            "ipsum",
        ),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "Lorem </test> ",
        MediaContent.url("http://image", media="image/png"),
        "ipsum",
    )


def test_returns_unchanged_with_reversed_tags():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("</test>Lorem <test> ipsum"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("</test>Lorem <test> ipsum")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("</test>Lorem <test> ipsum"),
        tag="test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("</test>Lorem <test> ipsum")

    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "</test>Lorem <test> ",
            MediaContent.url("http://image", media="image/png"),
            "ipsum",
        ),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "</test>Lorem <test> ",
        MediaContent.url("http://image", media="image/png"),
        "ipsum",
    )


def test_returns_unchanged_with_malformed_opening_tag():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<testx>Lorem ipsum</test>"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("<testx>Lorem ipsum</test>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("<testx>Lorem ipsum</test>"),
        tag="test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<testx>Lorem ipsum</test>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "<testx>Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test>",
        ),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<testx>Lorem",
        MediaContent.url("http://image", media="image/png"),
        "ipsum</test>",
    )


def test_returns_unchanged_with_malformed_closing_tag():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>Lorem ipsum</testx>"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>Lorem ipsum</testx>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>Lorem ipsum</testx>"),
        tag="test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<test>Lorem ipsum</testx>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "<test>Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</testx>",
        ),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>Lorem",
        MediaContent.url("http://image", media="image/png"),
        "ipsum</testx>",
    )


def test_returns_replaced_with_valid_tag():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>Lorem ipsum</test>"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>Lorem ipsum</test>"),
        tag="test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced")

    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "<test>Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test>",
        ),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>replaced</test>",
    )


def test_returns_replaced_outer_with_multiple_nested_tags():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>Other<test>Lorem ipsum</test></test>"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test></test>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>Other<test>Lorem ipsum</test></test>"),
        tag="test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced</test>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "<test>Other<test>Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test></test>",
        ),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>replaced</test></test>",
    )


def test_returns_replaced_with_surrounded_tag():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("Lorem<test>Lorem ipsum</test>ipsum"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("Lorem<test>replaced</test>ipsum")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("Lorem<test>Lorem ipsum</test>ipsum"),
        tag="test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("Loremreplacedipsum")

    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            MediaContent.url("http://image", media="image/png"),
            "Lorem<test>Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test>ipsum",
            MediaContent.url("http://image", media="image/png"),
        ),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of(
        MediaContent.url("http://image", media="image/png"),
        "Lorem<test>replaced</test>ipsum",
        MediaContent.url("http://image", media="image/png"),
    )


def test_returns_unchanged_with_nested_in_tags():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<other>Other<test>Lorem ipsum</test></other>"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("<other>Other<test>Lorem ipsum</test></other>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("<other>Other<test>Lorem ipsum</test></other>"),
        tag="test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<other>Other<test>Lorem ipsum</test></other>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "<other>Other<test>Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test></other>",
        ),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<other>Other<test>Lorem",
        MediaContent.url("http://image", media="image/png"),
        "ipsum</test></other>",
    )


def test_returns_replaced_content_with_fake_tags():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>Lorem<ipsum</test>"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("<te<test>Lorem<ipsum</test>"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("<te<test>replaced</test>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>Lorem<ipsum</test>"),
        tag="test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("<te<test>Lorem<ipsum</test>"),
        tag="test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<tereplaced")

    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "<test>Lorem<",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test>",
        ),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>replaced</test>",
    )

    assert MultimodalContent.of(
        "<te<test>Lorem<",
        MediaContent.url("http://image", media="image/png"),
        "ipsum</test>",
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<te<test>replaced</test>",
    )


def test_returns_replaced_with_other_tags():
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
        MediaContent.url("http://image", media="image/png"),
        "ipsum</test></other>",
    ).replacing(
        "test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<other>Other<more><test>replaced</test></other>",
    )


def test_returns_first_replaced_with_multiple_valid_tags():
    print(
        MultimodalTagElement.replace(
            MultimodalContent.of("<test>Lorem ipsum</test><test>Lorem ipsum</test>"),
            tag="test",
            replacement="replaced",
        )
    )
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>Lorem ipsum</test><test>Lorem ipsum</test>"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test><test>Lorem ipsum</test>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>Lorem ipsum</test><test>Lorem ipsum</test>"),
        tag="test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced<test>Lorem ipsum</test>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "<test>Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test>",
            "<test>Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test>",
        ),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>replaced</test>",
        "<test>Lorem",
        MediaContent.url("http://image", media="image/png"),
        "ipsum</test>",
    )


def test_returns_all_replaced_with_multiple_valid_tags():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>Lorem ipsum</test><test>Lorem ipsum</test>"),
        tag="test",
        replacement="replaced",
        replace_all=True,
    ) == MultimodalContent.of("<test>replaced</test><test>replaced</test>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>Lorem ipsum</test><test>Lorem ipsum</test>"),
        tag="test",
        replacement="replaced",
        strip_tags=True,
        replace_all=True,
    ) == MultimodalContent.of("replacedreplaced")

    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "<test>Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test>",
            "<test>Lorem",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test>",
        ),
        tag="test",
        replacement="replaced",
        replace_all=True,
    ) == MultimodalContent.of(
        "<test>replaced</test><test>replaced</test>",
    )
