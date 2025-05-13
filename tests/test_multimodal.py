from draive import MediaContent, MediaReference, MultimodalContent, TextContent

input_string: str = "Lorem ipsum,\ndolor sit amet"
input_text: TextContent = TextContent(text=input_string)
input_text_merged: TextContent = TextContent(text=input_string + input_string)
input_image: MediaContent = MediaReference.of(
    "http://image_url",
    media="image/png",
)
input_multimodal: MultimodalContent = MultimodalContent.of(
    input_text,
    input_image,
    input_text,
)


def test_empty_is_falsy():
    assert not MultimodalContent.empty


def test_as_string_is_equal_input_text():
    assert MultimodalContent.of(input_string).to_str() == input_string


def test_merged_texts_are_concatenated():
    assert MultimodalContent.of(input_string, input_string).parts == (input_text_merged,)


def test_merged_texts_with_media_are_concatenated():
    assert MultimodalContent.of(
        input_string,
        input_image,
        input_text,
        input_string,
    ).parts == (input_text, input_image, input_text_merged)


def test_empty_texts_are_skipped():
    assert MultimodalContent.of("", "", "").parts == ()


def test_merged_contents_with_same_meta_are_concatenated():
    assert MultimodalContent.of(
        input_multimodal,
        input_multimodal,
    ).parts == (input_text, input_image, input_text_merged, input_image, input_text)


def test_merged_contents_with_different_meta_are_concatenated_where_able():
    assert MultimodalContent.of(
        MultimodalContent.of(
            "",
            input_text.updated(meta={"test": True}),
            input_image,
            MultimodalContent.of(
                input_text.updated(meta={"test": True}),
                input_text.updated(meta={"test": True}),
            ),
        ),
        MultimodalContent.of(
            input_text.updated(meta={"test": False}),
            input_image,
            MultimodalContent.of(
                input_text.updated(meta={"test": False}),
                input_text,
            ),
        ),
    ).parts == (
        input_text.updated(meta={"test": True}),
        input_image,
        input_text_merged.updated(meta={"test": True}),
        input_text.updated(meta={"test": False}),
        input_image,
        input_text.updated(meta={"test": False}),
        input_text,
    )
