from datetime import UTC, datetime
from uuid import uuid4

import pytest
from haiway import State

from draive.multimodal import ArtifactContent, MultimodalContent, TextContent


class Payload(State, serializable=True):
    number: int
    text: str


def test_artifact_of_state_uses_class_name_and_round_trips() -> None:
    artifact = ArtifactContent.of(Payload(number=1, text="é日本"), meta={"k": 1})

    assert artifact.category == "Payload"
    assert artifact.to_state(Payload) == Payload(number=1, text="é日本")
    assert ArtifactContent.from_json(artifact.to_json()) == artifact


def test_artifact_hidden_renders_empty_string() -> None:
    assert ArtifactContent.of({"a": 1}, category="c", hidden=True).to_str() == ""


def test_artifact_to_str_keeps_unicode_readable() -> None:
    rendered: str = ArtifactContent.of({"text": "日本"}, category="c").to_str()

    assert "日本" in rendered


def test_artifact_of_artifact_merges_meta_and_keeps_hidden() -> None:
    source = ArtifactContent.of({"a": 1}, category="c", hidden=True, meta={"x": 1})

    derived = ArtifactContent.of(source, meta={"y": 2})

    assert derived.hidden
    assert derived.meta.get("x") == 1
    assert derived.meta.get("y") == 2


def test_artifact_uuid_and_datetime_accessors_round_trip() -> None:
    identifier = uuid4()
    moment = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    artifact = (
        ArtifactContent.of({}, category="c")
        .with_uuid("id", value=identifier)
        .with_datetime("at", value=moment)
    )

    assert artifact.get_uuid("id") == identifier
    assert artifact.get_datetime("at") == moment
    assert artifact.get_uuid("missing") is None
    assert artifact.get_uuid("missing", default=identifier) == identifier


@pytest.mark.parametrize(
    ("accessor", "value"),
    (
        ("get_str", 1),
        ("get_int", "1"),
        ("get_float", "1.0"),
        ("get_bool", 1),
        ("get_uuid", 1),
        ("get_datetime", 1),
    ),
)
def test_artifact_accessors_reject_wrong_types(accessor: str, value: object) -> None:
    artifact = ArtifactContent.of({"key": value}, category="c")  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError):
        getattr(artifact, accessor)("key")


def test_artifact_get_int_rejects_booleans() -> None:
    with pytest.raises(TypeError):
        ArtifactContent.of({"key": True}, category="c").get_int("key")


def test_artifact_truthiness_follows_payload() -> None:
    assert not ArtifactContent.of({}, category="c")
    assert ArtifactContent.of({"a": 1}, category="c")


def test_content_artifacts_are_filtered_by_category_and_decoded() -> None:
    content = MultimodalContent.of(
        "text",
        ArtifactContent.of(Payload(number=1, text="a")),
        ArtifactContent.of({"number": 2, "text": "b"}, category="Other"),
    )

    assert [artifact.category for artifact in content.artifacts()] == ["Payload", "Other"]
    assert content.artifacts(Payload, category="Payload") == (Payload(number=1, text="a"),)
    assert content.artifacts(category="Missing") == ()


def test_content_json_round_trip_with_mixed_parts() -> None:
    content = MultimodalContent.of(
        "héllo ",
        TextContent.of("x", meta={"m": 1}),
        ArtifactContent.of(Payload(number=1, text="日本")),
    )

    assert MultimodalContent.from_json(content.to_json()) == content
    assert MultimodalContent.from_mapping(content.to_mapping()) == content


def test_content_rejects_unsupported_elements() -> None:
    with pytest.raises(TypeError):
        MultimodalContent.of("a", None)  # pyright: ignore[reportArgumentType]


def test_empty_text_content_is_falsy_and_renders_empty() -> None:
    content = MultimodalContent.of("")

    assert not content
    assert content.to_str() == ""


def test_artifact_of_artifact_can_unhide_explicitly() -> None:
    source = ArtifactContent.of({"a": 1}, category="c", hidden=True)

    assert not ArtifactContent.of(source, hidden=False).hidden
    assert ArtifactContent.of(source).hidden
    assert ArtifactContent.of(source, hidden=True).hidden
    assert not ArtifactContent.of(ArtifactContent.of({"a": 1}, category="c")).hidden


def test_artifact_get_float_accepts_int_but_not_bool() -> None:
    artifact = ArtifactContent.of({"i": 2, "f": 1.5, "b": True, "s": "1"}, category="c")

    assert artifact.get_float("i") == 2.0
    assert isinstance(artifact.get_float("i"), float)
    assert artifact.get_float("f") == 1.5
    assert artifact.get_float("missing", default=0.25) == 0.25
    for key in ("b", "s"):
        with pytest.raises(TypeError):
            artifact.get_float(key)
