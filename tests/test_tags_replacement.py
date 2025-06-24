from draive import MediaReference, MultimodalContent, MultimodalTagElement


def test_returns_unchanged_with_empty():
    assert (
        MultimodalTagElement.replace(
            "test",
            content=MultimodalContent.empty,
            replacement="replaced",
        )
        == MultimodalContent.empty
    )

    assert (
        MultimodalTagElement.replace(
            "test",
            content=MultimodalContent.empty,
            replacement="replaced",
            strip_tags=True,
        )
        == MultimodalContent.empty
    )


def test_returns_unchanged_without_tag():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("Lorem ipsum"),
        replacement="replaced",
    ) == MultimodalContent.of("Lorem ipsum")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("Lorem ipsum"),
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("Lorem ipsum")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "Lorem",
            MediaReference.of("http://image", media="image/png"),
            "ipsum",
        ),
        replacement="replaced",
    ) == MultimodalContent.of(
        "Lorem",
        MediaReference.of("http://image", media="image/png"),
        "ipsum",
    )


def test_returns_unchanged_with_other_tag():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<other>Lorem ipsum</other>"),
        replacement="replaced",
    ) == MultimodalContent.of("<other>Lorem ipsum</other>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of('<other key="value">Lorem ipsum</other>'),
        replacement="replaced",
    ) == MultimodalContent.of('<other key="value">Lorem ipsum</other>')

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<other>Lorem ipsum</other>"),
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<other>Lorem ipsum</other>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "<other>Lorem",
            MediaReference.of("http://image", media="image/png"),
            "ipsum</other>",
        ),
        replacement="replaced",
    ) == MultimodalContent.of(
        "<other>Lorem",
        MediaReference.of("http://image", media="image/png"),
        "ipsum</other>",
    )


def test_returns_unchanged_without_closing_tag():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>Lorem ipsum"),
        replacement="replaced",
    ) == MultimodalContent.of("<test>Lorem ipsum")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>Lorem ipsum"),
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<test>Lorem ipsum")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "<test>Lorem",
            MediaReference.of("http://image", media="image/png"),
            "ipsum",
        ),
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>Lorem",
        MediaReference.of("http://image", media="image/png"),
        "ipsum",
    )


def test_returns_unchanged_with_only_closing_tag():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("Lorem </test> ipsum"),
        replacement="replaced",
    ) == MultimodalContent.of("Lorem </test> ipsum")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("Lorem </test> ipsum"),
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("Lorem </test> ipsum")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "Lorem </test> ",
            MediaReference.of("http://image", media="image/png"),
            "ipsum",
        ),
        replacement="replaced",
    ) == MultimodalContent.of(
        "Lorem </test> ",
        MediaReference.of("http://image", media="image/png"),
        "ipsum",
    )


def test_returns_unchanged_with_reversed_tags():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("</test>Lorem <test> ipsum"),
        replacement="replaced",
    ) == MultimodalContent.of("</test>Lorem <test> ipsum")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("</test>Lorem <test> ipsum"),
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("</test>Lorem <test> ipsum")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "</test>Lorem <test> ",
            MediaReference.of("http://image", media="image/png"),
            "ipsum",
        ),
        replacement="replaced",
    ) == MultimodalContent.of(
        "</test>Lorem <test> ",
        MediaReference.of("http://image", media="image/png"),
        "ipsum",
    )


def test_returns_unchanged_with_malformed_opening_tag():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<testx>Lorem ipsum</test>"),
        replacement="replaced",
    ) == MultimodalContent.of("<testx>Lorem ipsum</test>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test key=invalid>Lorem ipsum</test>"),
        replacement="replaced",
    ) == MultimodalContent.of("<test key=invalid>Lorem ipsum</test>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<testx>Lorem ipsum</test>"),
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<testx>Lorem ipsum</test>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "<testx>Lorem",
            MediaReference.of("http://image", media="image/png"),
            "ipsum</test>",
        ),
        replacement="replaced",
    ) == MultimodalContent.of(
        "<testx>Lorem",
        MediaReference.of("http://image", media="image/png"),
        "ipsum</test>",
    )


def test_returns_unchanged_with_malformed_closing_tag():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>Lorem ipsum</testx>"),
        replacement="replaced",
    ) == MultimodalContent.of("<test>Lorem ipsum</testx>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of('<test key="value">Lorem ipsum</testx>'),
        replacement="replaced",
    ) == MultimodalContent.of('<test key="value">Lorem ipsum</testx>')

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>Lorem ipsum</testx>"),
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<test>Lorem ipsum</testx>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "<test>Lorem",
            MediaReference.of("http://image", media="image/png"),
            "ipsum</testx>",
        ),
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>Lorem",
        MediaReference.of("http://image", media="image/png"),
        "ipsum</testx>",
    )


def test_returns_replaced_with_valid_tag():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>Lorem ipsum</test>"),
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>Lorem ipsum</test>"),
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "<test>Lorem",
            MediaReference.of("http://image", media="image/png"),
            "ipsum</test>",
        ),
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>replaced</test>",
    )


def test_returns_replaced_outer_with_multiple_nested_tags():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>Other<test>Lorem ipsum</test></test>"),
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test></test>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>Other<test>Lorem ipsum</test></test>"),
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced</test>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "<test>Other<test>Lorem",
            MediaReference.of("http://image", media="image/png"),
            "ipsum</test></test>",
        ),
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>replaced</test></test>",
    )


def test_returns_replaced_with_surrounded_tag():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("Lorem<test>Lorem ipsum</test>ipsum"),
        replacement="replaced",
    ) == MultimodalContent.of("Lorem<test>replaced</test>ipsum")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("Lorem<test>Lorem ipsum</test>ipsum"),
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("Loremreplacedipsum")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            MediaReference.of("http://image", media="image/png"),
            "Lorem<test>Lorem",
            MediaReference.of("http://image", media="image/png"),
            "ipsum</test>ipsum",
            MediaReference.of("http://image", media="image/png"),
        ),
        replacement="replaced",
    ) == MultimodalContent.of(
        MediaReference.of("http://image", media="image/png"),
        "Lorem<test>replaced</test>ipsum",
        MediaReference.of("http://image", media="image/png"),
    )


def test_returns_unchanged_with_nested_in_tags():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<other>Other<test>Lorem ipsum</test></other>"),
        replacement="replaced",
    ) == MultimodalContent.of("<other>Other<test>Lorem ipsum</test></other>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<other>Other<test>Lorem ipsum</test></other>"),
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<other>Other<test>Lorem ipsum</test></other>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "<other>Other<test>Lorem",
            MediaReference.of("http://image", media="image/png"),
            "ipsum</test></other>",
        ),
        replacement="replaced",
    ) == MultimodalContent.of(
        "<other>Other<test>Lorem",
        MediaReference.of("http://image", media="image/png"),
        "ipsum</test></other>",
    )


def test_returns_replaced_content_with_fake_tags():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>Lorem<ipsum</test>"),
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<te<test>Lorem<ipsum</test>"),
        replacement="replaced",
    ) == MultimodalContent.of("<te<test>replaced</test>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>Lorem<ipsum</test>"),
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<te<test>Lorem<ipsum</test>"),
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<tereplaced")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "<test>Lorem<",
            MediaReference.of("http://image", media="image/png"),
            "ipsum</test>",
        ),
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>replaced</test>",
    )

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "<te<test>Lorem<",
            MediaReference.of("http://image", media="image/png"),
            "ipsum</test>",
        ),
        replacement="replaced",
    ) == MultimodalContent.of(
        "<te<test>replaced</test>",
    )


def test_returns_replaced_with_other_tags():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<other>Other</other><test>Lorem</more>ipsum</test><more>"),
        replacement="replaced",
    ) == MultimodalContent.of("<other>Other</other><test>replaced</test><more>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<other>Other</other><test>Lorem</more>ipsum</test><more>"),
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("<other>Other</other>replaced<more>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "<other>Other</other><test>Lorem</more>",
            MediaReference.of("http://image", media="image/png"),
            "ipsum</test><more>",
        ),
        replacement="replaced",
    ) == MultimodalContent.of(
        "<other>Other</other><test>replaced</test><more>",
    )


def test_returns_first_replaced_with_multiple_valid_tags():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>Lorem ipsum</test><test>Lorem ipsum</test>"),
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test><test>Lorem ipsum</test>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>Lorem ipsum</test><test>Lorem ipsum</test>"),
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced<test>Lorem ipsum</test>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "<test>Lorem",
            MediaReference.of("http://image", media="image/png"),
            "ipsum</test>",
            "<test>Lorem",
            MediaReference.of("http://image", media="image/png"),
            "ipsum</test>",
        ),
        replacement="replaced",
    ) == MultimodalContent.of(
        "<test>replaced</test>",
        "<test>Lorem",
        MediaReference.of("http://image", media="image/png"),
        "ipsum</test>",
    )


def test_returns_all_replaced_with_multiple_valid_tags():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>Lorem ipsum</test><test>Lorem ipsum</test>"),
        replacement="replaced",
        replace_all=True,
    ) == MultimodalContent.of("<test>replaced</test><test>replaced</test>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>Lorem ipsum</test><test>Lorem ipsum</test>"),
        replacement="replaced",
        strip_tags=True,
        replace_all=True,
    ) == MultimodalContent.of("replacedreplaced")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "<test>Lorem",
            MediaReference.of("http://image", media="image/png"),
            "ipsum</test>",
            "<test>Lorem",
            MediaReference.of("http://image", media="image/png"),
            "ipsum</test>",
        ),
        replacement="replaced",
        replace_all=True,
    ) == MultimodalContent.of(
        "<test>replaced</test><test>replaced</test>",
    )


### generated ###


def test_handles_empty_replacement():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>content</test>"),
        replacement="",
    ) == MultimodalContent.of("<test></test>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>content</test>"),
        replacement="",
        strip_tags=True,
    ) == MultimodalContent.of("")


def test_handles_whitespace_replacement():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>content</test>"),
        replacement="   \n\t  ",
    ) == MultimodalContent.of("<test>   \n\t  </test>")


def test_handles_self_closing_tag_replacement():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test/>"),
        replacement="content",
    ) == MultimodalContent.of("<test>content</test>")

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test/>"),
        replacement="content",
        strip_tags=True,
    ) == MultimodalContent.of("content")


def test_handles_multiple_self_closing_tags():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test/><other/><test/>"),
        replacement="content",
        replace_all=True,
    ) == MultimodalContent.of("<test>content</test><other/><test>content</test>")


def test_preserves_tag_attributes_on_replacement():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of('<test id="1" class="example">old</test>'),
        replacement="new",
    ) == MultimodalContent.of('<test id="1" class="example">new</test>')


def test_handles_nested_tags_with_same_name():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test><test><test>content</test></test></test>"),
        replacement="replaced",
        replace_all=True,
    ) == MultimodalContent.of("<test>replaced</test></test></test>")


def test_handles_mixed_content_replacement():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "<test>",
            MediaReference.of("http://image1", media="image/png"),
            "text",
            MediaReference.of("http://image2", media="image/png"),
            "</test>",
        ),
        replacement=MediaReference.of("http://replacement", media="image/jpeg"),
    ) == MultimodalContent.of(
        "<test>", MediaReference.of("http://replacement", media="image/jpeg"), "</test>"
    )


def test_handles_escaped_characters_in_tags():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of('<test escaped="\\n\\t\\"">content</test>'),
        replacement="replaced",
    ) == MultimodalContent.of('<test escaped="\\n\\t\\"">replaced</test>')


def test_handles_adjacent_tags():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>one</test><test>two</test><test>three</test>"),
        replacement="replaced",
        replace_all=True,
    ) == MultimodalContent.of("<test>replaced</test><test>replaced</test><test>replaced</test>")


def test_handles_tags_with_numeric_suffixes():
    assert MultimodalTagElement.replace(
        "test1",
        content=MultimodalContent.of("<test1>content</test1>"),
        replacement="replaced",
    ) == MultimodalContent.of("<test1>replaced</test1>")


def test_handles_tags_with_special_characters():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>< > & \" '</test>"),
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test>")


def test_handles_replacement_with_tags():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>content</test>"),
        replacement="<other>nested</other>",
    ) == MultimodalContent.of("<test><other>nested</other></test>")


def test_handles_multiline_content():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>\nline1\nline2\n</test>"),
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test>")


def test_handles_case_insensitive_replacement():
    # Test that uppercase tags get replaced by lowercase search
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<TEST>content</TEST>"),
        replacement="replaced",
    ) == MultimodalContent.of("<TEST>replaced</TEST>")

    # Test mixed case tag replacement
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<Test>content</Test>"),
        replacement="replaced",
    ) == MultimodalContent.of("<Test>replaced</Test>")

    # Test uppercase search replacing lowercase tag
    assert MultimodalTagElement.replace(
        "TEST",
        content=MultimodalContent.of("<test>content</test>"),
        replacement="replaced",
    ) == MultimodalContent.of("<test>replaced</test>")

    # Test mixed cache search replacing mixed tag
    assert MultimodalTagElement.replace(
        "TeSt",
        content=MultimodalContent.of("<tEsT>content</tEsT>"),
        replacement="replaced",
    ) == MultimodalContent.of("<tEsT>replaced</tEsT>")

    # Test with strip_tags
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<TEST>content</TEST>"),
        replacement="replaced",
        strip_tags=True,
    ) == MultimodalContent.of("replaced")


def test_handles_multiple_attributes_with_quotes():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of('<test a="1" b="2" c="3">content</test>'),
        replacement="replaced",
    ) == MultimodalContent.of('<test a="1" b="2" c="3">replaced</test>')


def test_handles_complex_nested_structure():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "<outer><test>one</test><middle><test>two</test></middle><test>three</test></outer>"
        ),
        replacement="replaced",
        replace_all=True,
    ) == MultimodalContent.of(
        "<outer><test>one</test><middle><test>two</test></middle><test>three</test></outer>"
    )

    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "<outer><test>one</test><middle></outer><test>two</test></middle><test>three</test>"
        ),
        replacement="replaced",
        replace_all=True,
    ) == MultimodalContent.of(
        "<outer><test>one</test><middle></outer><test>replaced</test></middle><test>replaced</test>"
    )


def test_handles_replacement_with_multiple_media():
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<test>old</test>"),
        replacement=MultimodalContent.of(
            MediaReference.of("http://image1", media="image/png"),
            "text",
            MediaReference.of("http://image2", media="image/jpeg"),
        ),
    ) == MultimodalContent.of(
        "<test>",
        MediaReference.of("http://image1", media="image/png"),
        "text",
        MediaReference.of("http://image2", media="image/jpeg"),
        "</test>",
    )


def test_case_insensitive_self_closing_replacement():
    # Test case insensitive self-closing tag replacement
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<TEST/>"),
        replacement="content",
    ) == MultimodalContent.of("<TEST>content</TEST>")

    assert MultimodalTagElement.replace(
        "TEST",
        content=MultimodalContent.of("<test/>"),
        replacement="content",
        strip_tags=True,
    ) == MultimodalContent.of("content")


def test_case_insensitive_multiple_replacements():
    # Test case insensitive replacement with multiple mixed case tags
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of("<TEST>first</TEST><test>second</test><Test>third</Test>"),
        replacement="replaced",
        replace_all=True,
    ) == MultimodalContent.of("<TEST>replaced</TEST><test>replaced</test><Test>replaced</Test>")


def test_case_insensitive_with_attributes_replacement():
    # Test case insensitive replacement preserves attributes
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of('<TEST id="1" class="example">old</TEST>'),
        replacement="new",
    ) == MultimodalContent.of('<TEST id="1" class="example">new</TEST>')


def test_case_insensitive_mixed_with_media():
    # Test case insensitive replacement with media content
    assert MultimodalTagElement.replace(
        "test",
        content=MultimodalContent.of(
            "<TEST>old",
            MediaReference.of("http://image", media="image/png"),
            "content</TEST>",
        ),
        replacement="replaced",
    ) == MultimodalContent.of("<TEST>replaced</TEST>")
