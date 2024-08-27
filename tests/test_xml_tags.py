from draive import ImageURLContent, MultimodalContent, xml_tag, xml_tags


def test_returns_none_with_empty():
    text_source: str = ""

    assert xml_tag("test", source=text_source) is None

    multimodal_source: MultimodalContent = MultimodalContent.of()

    assert xml_tag("test", source=multimodal_source) is None


def test_returns_none_without_tag():
    text_source: str = "Lorem ipsum"

    assert xml_tag("test", source=text_source) is None

    multimodal_source: MultimodalContent = MultimodalContent.of(
        "Lorem",
        ImageURLContent(image_url="image"),
        "ipsum",
    )

    assert xml_tag("test", source=multimodal_source) is None


def test_returns_none_with_other_tag():
    text_source: str = "<other>Lorem ipsum</other>"

    assert xml_tag("test", source=text_source) is None

    multimodal_source: MultimodalContent = MultimodalContent.of(
        "<other>Lorem",
        ImageURLContent(image_url="image"),
        "ipsum</other>",
    )

    assert xml_tag("test", source=multimodal_source) is None


def test_returns_none_with_closing_tag():
    text_source: str = "Lorem ipsum</test>"

    assert xml_tag("test", source=text_source) is None

    multimodal_source: MultimodalContent = MultimodalContent.of(
        "Lorem",
        ImageURLContent(image_url="image"),
        "ipsum</test>",
    )

    assert xml_tag("test", source=multimodal_source) is None


def test_returns_none_with_reversed_tags():
    text_source: str = "</test>Lorem ipsum<test>"

    assert xml_tag("test", source=text_source) is None

    multimodal_source: MultimodalContent = MultimodalContent.of(
        "</test>Lorem",
        ImageURLContent(image_url="image"),
        "ipsum<test>",
    )

    assert xml_tag("test", source=multimodal_source) is None


def test_returns_none_without_closing_tag():
    text_source: str = "<test>Lorem ipsum"

    assert xml_tag("test", source=text_source) is None

    multimodal_source: MultimodalContent = MultimodalContent.of(
        "<test>Lorem",
        ImageURLContent(image_url="image"),
        "ipsum",
    )

    assert xml_tag("test", source=multimodal_source) is None


def test_returns_none_with_malformed_opening_tag():
    text_source: str = "<testx>Lorem ipsum</test>"

    assert xml_tag("test", source=text_source) is None

    multimodal_source: MultimodalContent = MultimodalContent.of(
        "<testx>Lorem",
        ImageURLContent(image_url="image"),
        "ipsum</test>",
    )

    assert xml_tag("test", source=multimodal_source) is None


def test_returns_none_with_malformed_closing_tag():
    text_source: str = "<test>Lorem ipsum</testx>"

    assert xml_tag("test", source=text_source) is None

    multimodal_source: MultimodalContent = MultimodalContent.of(
        "<test>Lorem",
        ImageURLContent(image_url="image"),
        "ipsum</testx>",
    )

    assert xml_tag("test", source=multimodal_source) is None


def test_returns_content_with_valid_tag():
    text_source: str = "<test>Lorem ipsum</test>"

    assert xml_tag("test", source=text_source, conversion=str) == "Lorem ipsum"

    multimodal_source: MultimodalContent = MultimodalContent.of(
        "<test>Lorem",
        ImageURLContent(image_url="image"),
        "ipsum</test>",
    )

    assert xml_tag("test", source=multimodal_source) == MultimodalContent.of(
        "Lorem",
        ImageURLContent(image_url="image"),
        "ipsum",
    )


def test_returns_content_with_surrounded_tag():
    text_source: str = "Lorem<test>Lorem ipsum</test>ipsum"

    assert xml_tag("test", source=text_source, conversion=str) == "Lorem ipsum"

    multimodal_source: MultimodalContent = MultimodalContent.of(
        ImageURLContent(image_url="image"),
        "Lorem<test>Lorem",
        ImageURLContent(image_url="image"),
        "ipsum</test>ipsum",
        ImageURLContent(image_url="image"),
    )

    assert xml_tag("test", source=multimodal_source) == MultimodalContent.of(
        "Lorem",
        ImageURLContent(image_url="image"),
        "ipsum",
    )


def test_returns_content_with_opening_tag_containing_extras():
    text_source: str = "<test extra items=here>Lorem ipsum</test>"

    assert xml_tag("test", source=text_source, conversion=str) == "Lorem ipsum"

    multimodal_source: MultimodalContent = MultimodalContent.of(
        "<test extra items=here>Lorem",
        ImageURLContent(image_url="image"),
        "ipsum</test>",
    )

    assert xml_tag("test", source=multimodal_source) == MultimodalContent.of(
        "Lorem",
        ImageURLContent(image_url="image"),
        "ipsum",
    )


def test_returns_first_content_with_multiple_tags():
    text_source: str = "<test>Lorem ipsum</test><test>Other</test>"

    assert xml_tag("test", source=text_source, conversion=str) == "Lorem ipsum"

    multimodal_source: MultimodalContent = MultimodalContent.of(
        "<test>Lorem",
        ImageURLContent(image_url="image"),
        "ipsum</test><test>Other</test>",
    )

    assert xml_tag("test", source=multimodal_source) == MultimodalContent.of(
        "Lorem",
        ImageURLContent(image_url="image"),
        "ipsum",
    )


def test_returns_nested_content_with_multiple_nested_tags():
    text_source: str = "<test>Other<test>Lorem ipsum</test></test>"

    assert xml_tag("test", source=text_source, conversion=str) == "Lorem ipsum"

    multimodal_source: MultimodalContent = MultimodalContent.of(
        "<test>Other<test>Lorem",
        ImageURLContent(image_url="image"),
        "ipsum</test></test>",
    )

    assert xml_tag("test", source=multimodal_source) == MultimodalContent.of(
        "Lorem",
        ImageURLContent(image_url="image"),
        "ipsum",
    )


def test_returns_nested_content_with_fake_tags():
    text_source: str = "<test>Lorem<ipsum</test>"

    assert xml_tag("test", source=text_source, conversion=str) == "Lorem<ipsum"

    multimodal_source: MultimodalContent = MultimodalContent.of(
        "<test>Lorem<",
        ImageURLContent(image_url="image"),
        "ipsum</test>",
    )

    assert xml_tag("test", source=multimodal_source) == MultimodalContent.of(
        "Lorem<",
        ImageURLContent(image_url="image"),
        "ipsum",
    )


def test_returns_nested_content_with_other_tags():
    text_source: str = "<other>Other<more><test>Lorem</more>ipsum</test></other>"

    assert xml_tag("test", source=text_source, conversion=str) == "Lorem</more>ipsum"

    multimodal_source: MultimodalContent = MultimodalContent.of(
        "<other>Other<more><test>Lorem</more>",
        ImageURLContent(image_url="image"),
        "ipsum</test></other>",
    )

    assert xml_tag("test", source=multimodal_source) == MultimodalContent.of(
        "Lorem</more>",
        ImageURLContent(image_url="image"),
        "ipsum",
    )


def test_returns_content_with_multiple_tags():
    text_source: str = "<test>Lorem ipsum</test><test>Dolor</test><test>Sit amet</test>"

    assert list(xml_tags("test", source=text_source, conversion=str)) == [
        "Lorem ipsum",
        "Dolor",
        "Sit amet",
    ]

    multimodal_source: MultimodalContent = MultimodalContent.of(
        "<test>Lorem ",
        ImageURLContent(image_url="image"),
        "ipsum</test><test>Dolor</test><test>Sit ",
        ImageURLContent(image_url="image"),
        "amet</test>",
    )

    assert list(xml_tags("test", source=multimodal_source)) == [
        MultimodalContent.of(
            "Lorem ",
            ImageURLContent(image_url="image"),
            "ipsum",
        ),
        MultimodalContent.of(
            "Dolor",
        ),
        MultimodalContent.of(
            "Sit ",
            ImageURLContent(image_url="image"),
            "amet",
        ),
    ]


def test_skips_content_from_different_tags():
    text_source: str = "<test>Lorem ipsum</test><other>Other</other><more><test>Dolor</test>"

    assert list(xml_tags("test", source=text_source, conversion=str)) == ["Lorem ipsum", "Dolor"]

    multimodal_source: MultimodalContent = MultimodalContent.of(
        "<test>Lorem ",
        ImageURLContent(image_url="image"),
        "ipsum</test><other>Other</other><more><test>Sit ",
        ImageURLContent(image_url="image"),
        "amet</test>",
    )

    assert list(xml_tags("test", source=multimodal_source)) == [
        MultimodalContent.of(
            "Lorem ",
            ImageURLContent(image_url="image"),
            "ipsum",
        ),
        MultimodalContent.of(
            "Sit ",
            ImageURLContent(image_url="image"),
            "amet",
        ),
    ]
