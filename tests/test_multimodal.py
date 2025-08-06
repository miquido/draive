from draive import MediaContent, MediaReference, MetaContent, MultimodalContent, TextContent

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


def test_matching_meta_returns_self_when_no_values():
    content = MultimodalContent.of(input_text, input_image)
    assert content.matching_meta() is content


def test_matching_meta_filters_by_single_value():
    text_with_meta = input_text.updated(meta={"category": "important"})
    image_with_meta = input_image.updated(meta={"category": "important"})
    text_without_meta = input_text.updated(meta={"category": "normal"})

    content = MultimodalContent.of(text_with_meta, image_with_meta, text_without_meta)
    result = content.matching_meta(category="important")

    assert result.parts == (text_with_meta, image_with_meta)


def test_matching_meta_filters_by_multiple_values():
    text_match = input_text.updated(meta={"category": "important", "priority": "high"})
    image_partial = input_image.updated(meta={"category": "important", "priority": "low"})
    text_no_match = input_text.updated(meta={"category": "normal", "priority": "high"})

    content = MultimodalContent.of(text_match, image_partial, text_no_match)
    result = content.matching_meta(category="important", priority="high")

    assert result.parts == (text_match,)


def test_matching_meta_handles_missing_metadata():
    text_with_meta = input_text.updated(meta={"category": "important"})
    text_without_meta = input_text  # no meta

    content = MultimodalContent.of(text_with_meta, text_without_meta)
    result = content.matching_meta(category="important")

    assert result.parts == (text_with_meta,)


def test_matching_meta_excludes_datamodel_artifacts():
    from draive.parameters import DataModel

    class TestArtifact(DataModel):
        value: str

    text_with_meta = input_text.updated(meta={"test": "value"})
    artifact = TestArtifact(value="test")

    content = MultimodalContent.of(text_with_meta, artifact)
    result = content.matching_meta(test="value")

    assert result.parts == (text_with_meta,)


def test_matching_meta_works_with_meta_content():
    meta_content = MetaContent(category="test", content=input_text, meta={"tag": "special"})
    text_with_meta = input_text.updated(meta={"tag": "special"})
    text_without_meta = input_text.updated(meta={"tag": "normal"})

    content = MultimodalContent.of(meta_content, text_with_meta, text_without_meta)
    result = content.matching_meta(tag="special")

    assert result.parts == (meta_content, text_with_meta)


def test_matching_meta_returns_empty_when_no_matches():
    text_with_meta = input_text.updated(meta={"category": "normal"})
    content = MultimodalContent.of(text_with_meta)
    result = content.matching_meta(category="important")

    assert result.parts == ()


def test_split_by_meta_returns_empty_for_empty_content():
    result = MultimodalContent.empty.split_by_meta(key="category")
    assert result == ()


def test_split_by_meta_single_group():
    text1 = input_text.updated(meta={"category": "important"})
    image1 = input_image.updated(meta={"category": "important"})
    text2 = input_text.updated(meta={"category": "important"})
    content = MultimodalContent.of(text1, image1, text2)

    result = content.split_by_meta(key="category")

    assert len(result) == 1
    assert result[0].parts == (text1, image1, text2)


def test_split_by_meta_multiple_groups():
    text_important = input_text.updated(meta={"category": "important"})
    image_important = input_image.updated(meta={"category": "important"})
    text_normal = input_text.updated(meta={"category": "normal"})
    text_important2 = input_text.updated(meta={"category": "important"})

    content = MultimodalContent.of(text_important, image_important, text_normal, text_important2)
    result = content.split_by_meta(key="category")

    assert len(result) == 3
    assert result[0].parts == (text_important, image_important)
    assert result[1].parts == (text_normal,)
    assert result[2].parts == (text_important2,)


def test_split_by_meta_handles_none_values():
    text_with_meta = input_text.updated(meta={"category": "important"})
    text_without_meta = input_text  # no meta
    text_with_different_meta = input_text.updated(meta={"other": "value"})  # no "category" key

    content = MultimodalContent.of(text_with_meta, text_without_meta, text_with_different_meta)
    result = content.split_by_meta(key="category")

    assert len(result) == 2
    assert result[0].parts == (text_with_meta,)
    assert result[1].parts == (text_without_meta, text_with_different_meta)


def test_split_by_meta_includes_datamodel_artifacts():
    from draive.parameters import DataModel

    class TestArtifact(DataModel):
        value: str

    text_with_meta = input_text.updated(meta={"category": "important"})
    artifact = TestArtifact(value="test")
    text_with_same_meta = input_text.updated(meta={"category": "important"})

    content = MultimodalContent.of(text_with_meta, artifact, text_with_same_meta)
    result = content.split_by_meta(key="category")

    # Artifact separates the text parts, so we get 3 groups
    assert len(result) == 3
    assert result[0].parts == (text_with_meta,)
    assert result[1].parts == (artifact,)
    assert result[2].parts == (text_with_same_meta,)


def test_split_by_meta_preserves_order():
    text1 = input_text.updated(meta={"priority": "high"})
    text2 = input_text.updated(meta={"priority": "low"})
    text3 = input_text.updated(meta={"priority": "high"})
    text4 = input_text.updated(meta={"priority": "low"})
    text5 = input_text.updated(meta={"priority": "high"})

    content = MultimodalContent.of(text1, text2, text3, text4, text5)
    result = content.split_by_meta(key="priority")

    assert len(result) == 5
    assert result[0].parts == (text1,)
    assert result[1].parts == (text2,)
    assert result[2].parts == (text3,)
    assert result[3].parts == (text4,)
    assert result[4].parts == (text5,)


def test_split_by_meta_groups_consecutive_same_values():
    text1 = input_text.updated(meta={"priority": "high"})
    image1 = input_image.updated(meta={"priority": "high"})
    text2 = input_text.updated(meta={"priority": "high"})
    text3 = input_text.updated(meta={"priority": "low"})
    image2 = input_image.updated(meta={"priority": "low"})
    text4 = input_text.updated(meta={"priority": "low"})
    text5 = input_text.updated(meta={"priority": "high"})

    content = MultimodalContent.of(text1, image1, text2, text3, image2, text4, text5)
    result = content.split_by_meta(key="priority")

    assert len(result) == 3
    assert result[0].parts == (text1, image1, text2)
    assert result[1].parts == (text3, image2, text4)
    assert result[2].parts == (text5,)
