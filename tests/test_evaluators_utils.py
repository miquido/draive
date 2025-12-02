import pytest

from draive.evaluators.utils import extract_evaluation_result
from draive.multimodal import MultimodalContent, MultimodalTag


def _content_with_tags(*tags: MultimodalTag) -> MultimodalContent:
    return MultimodalContent.of(*tags)


def test_extract_evaluation_result_parses_single_rating_and_comment() -> None:
    content = _content_with_tags(
        MultimodalTag.of("Reasoned view", name="comment"),
        MultimodalTag.of("good", name="rating"),
    )

    result = extract_evaluation_result(content)

    assert result.value == 0.5
    assert result.meta["comment"] == "Reasoned view"


def test_extract_evaluation_result_requires_rating() -> None:
    content = _content_with_tags(MultimodalTag.of("Because", name="comment"))

    with pytest.raises(ValueError, match="missing rating"):
        extract_evaluation_result(content)


def test_extract_evaluation_result_rejects_multiple_ratings() -> None:
    content = _content_with_tags(
        MultimodalTag.of("Because", name="comment"),
        MultimodalTag.of("good", name="rating"),
        MultimodalTag.of("excellent", name="rating"),
    )

    with pytest.raises(ValueError):
        extract_evaluation_result(content)


def test_extract_evaluation_result_does_not_require_comment() -> None:
    content = _content_with_tags(MultimodalTag.of("good", name="rating"))

    assert extract_evaluation_result(content).value == 0.5
    assert extract_evaluation_result(content).meta["comment"] is None


def test_extract_evaluation_result_rejects_unknown_rating_value() -> None:
    content = _content_with_tags(
        MultimodalTag.of("Because", name="comment"),
        MultimodalTag.of("not-a-rating", name="rating"),
    )

    with pytest.raises(ValueError):
        extract_evaluation_result(content)
