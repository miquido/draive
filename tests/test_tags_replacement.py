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
        MultimodalContent.of('<other key="value">Lorem ipsum</other>'),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of('<other key="value">Lorem ipsum</other>')

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
        MultimodalContent.of("<test key=invalid>Lorem ipsum</test>"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("<test key=invalid>Lorem ipsum</test>")

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
        MultimodalContent.of('<test key="value">Lorem ipsum</testx>'),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of('<test key="value">Lorem ipsum</testx>')

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

    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "<te<test>Lorem<",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test>",
        ),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<te<test>replaced</test>",
    )


def test_returns_replaced_with_other_tags():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<other>Other</other><test>Lorem</more>ipsum</test><more>"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("<other>Other</other><test>replaced</test><more>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("<other>Other</other><test>Lorem</more>ipsum</test><more>"),
        tag="test",
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<other>Other</other>replaced<more>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "<other>Other</other><test>Lorem</more>",
            MediaContent.url("http://image", media="image/png"),
            "ipsum</test><more>",
        ),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of(
        "<other>Other</other><test>replaced</test><more>",
    )


def test_returns_first_replaced_with_multiple_valid_tags():
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


### generated ###


def test_handles_empty_replacement():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>content</test>"),
        tag="test",
        replacement="",
    ) == MultimodalContent.of("<test></test>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>content</test>"),
        tag="test",
        replacement="",
        strip_tags=True,
    ) == MultimodalContent.of("")


def test_handles_whitespace_replacement():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>content</test>"),
        tag="test",
        replacement="   \n\t  ",
    ) == MultimodalContent.of("<test>   \n\t  </test>")


def test_handles_self_closing_tag_replacement():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test/>"),
        tag="test",
        replacement="content",
    ) == MultimodalContent.of("<test>content</test>")

    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test/>"),
        tag="test",
        replacement="content",
        strip_tags=True,
    ) == MultimodalContent.of("content")


def test_handles_multiple_self_closing_tags():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test/><other/><test/>"),
        tag="test",
        replacement="content",
        replace_all=True,
    ) == MultimodalContent.of("<test>content</test><other/><test>content</test>")


def test_preserves_tag_attributes_on_replacement():
    assert MultimodalTagElement.replace(
        MultimodalContent.of('<test id="1" class="example">old</test>'),
        tag="test",
        replacement="new",
    ) == MultimodalContent.of('<test id="1" class="example">new</test>')


def test_handles_nested_tags_with_same_name():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test><test><test>content</test></test></test>"),
        tag="test",
        replacement="replaced",
        replace_all=True,
    ) == MultimodalContent.of("<test>replaced</test></test></test>")


def test_handles_mixed_content_replacement():
    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "<test>",
            MediaContent.url("http://image1", media="image/png"),
            "text",
            MediaContent.url("http://image2", media="image/png"),
            "</test>",
        ),
        tag="test",
        replacement=MediaContent.url("http://replacement", media="image/jpeg"),
    ) == MultimodalContent.of(
        "<test>", MediaContent.url("http://replacement", media="image/jpeg"), "</test>"
    )


def test_handles_escaped_characters_in_tags():
    assert MultimodalTagElement.replace(
        MultimodalContent.of('<test escaped="\\n\\t\\"">content</test>'),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of('<test escaped="\\n\\t\\"">replaced</test>')


def test_handles_adjacent_tags():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>one</test><test>two</test><test>three</test>"),
        tag="test",
        replacement="replaced",
        replace_all=True,
    ) == MultimodalContent.of("<test>replaced</test><test>replaced</test><test>replaced</test>")


def test_handles_tags_with_numeric_suffixes():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test1>content</test1>"),
        tag="test1",
        replacement="replaced",
    ) == MultimodalContent.of("<test1>replaced</test1>")


def test_handles_tags_with_special_characters():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>< > & \" '</test>"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test>")


def test_handles_replacement_with_tags():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>content</test>"),
        tag="test",
        replacement="<other>nested</other>",
    ) == MultimodalContent.of("<test><other>nested</other></test>")


def test_handles_multiline_content():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>\nline1\nline2\n</test>"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test>")


def test_handles_mixed_case_sensitivity():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<TEST>content</TEST>"),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of("<TEST>content</TEST>")


def test_handles_multiple_attributes_with_quotes():
    assert MultimodalTagElement.replace(
        MultimodalContent.of('<test a="1" b="2" c="3">content</test>'),
        tag="test",
        replacement="replaced",
    ) == MultimodalContent.of('<test a="1" b="2" c="3">replaced</test>')


def test_handles_complex_nested_structure():
    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "<outer><test>one</test><middle><test>two</test></middle><test>three</test></outer>"
        ),
        tag="test",
        replacement="replaced",
        replace_all=True,
    ) == MultimodalContent.of(
        "<outer><test>one</test><middle><test>two</test></middle><test>three</test></outer>"
    )

    assert MultimodalTagElement.replace(
        MultimodalContent.of(
            "<outer><test>one</test><middle></outer><test>two</test></middle><test>three</test>"
        ),
        tag="test",
        replacement="replaced",
        replace_all=True,
    ) == MultimodalContent.of(
        "<outer><test>one</test><middle></outer><test>replaced</test></middle><test>replaced</test>"
    )


def test_handles_replacement_with_multiple_media():
    assert MultimodalTagElement.replace(
        MultimodalContent.of("<test>old</test>"),
        tag="test",
        replacement=MultimodalContent.of(
            MediaContent.url("http://image1", media="image/png"),
            "text",
            MediaContent.url("http://image2", media="image/jpeg"),
        ),
    ) == MultimodalContent.of(
        "<test>",
        MediaContent.url("http://image1", media="image/png"),
        "text",
        MediaContent.url("http://image2", media="image/jpeg"),
        "</test>",
    )
