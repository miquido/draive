from datetime import UTC, datetime

import pytest
from haiway import Meta, MetaValues

from draive import MultimodalContent, ProcessingEvent, TextContent
from draive.utils import ProcessingEvent as ExportedProcessingEvent


def test_processing_event_is_exported() -> None:
    assert ExportedProcessingEvent is ProcessingEvent


def test_of_defaults_to_empty_content_and_meta() -> None:
    event = ProcessingEvent.of("started")

    assert event.event == "started"
    assert event.content == MultimodalContent.empty
    assert event.meta == Meta.empty


def test_of_sets_utc_timestamp() -> None:
    before = datetime.now(UTC)
    event = ProcessingEvent.of("started")
    after = datetime.now(UTC)

    assert event.created.tzinfo is UTC
    assert before <= event.created <= after


def test_of_converts_string_content() -> None:
    event = ProcessingEvent.of("progress", "half way")

    assert event.content == MultimodalContent.of("half way")
    assert event.content.to_str() == "half way"


def test_of_keeps_multimodal_content() -> None:
    content = MultimodalContent.of(TextContent.of("a"), TextContent.of("b"))

    assert ProcessingEvent.of("progress", content).content == content


@pytest.mark.parametrize(
    "meta",
    (
        {"key": "value"},
        Meta.of({"key": "value"}),
    ),
    ids=("values", "meta"),
)
def test_of_accepts_meta_values_and_meta(meta: Meta | MetaValues) -> None:
    event = ProcessingEvent.of("progress", meta=meta)

    assert event.meta == Meta.of({"key": "value"})


def test_event_is_immutable() -> None:
    event = ProcessingEvent.of("started")

    with pytest.raises((AttributeError, TypeError)):
        event.event = "other"  # pyright: ignore[reportAttributeAccessIssue]


def test_updating_returns_new_instance() -> None:
    event = ProcessingEvent.of("started")
    updated = event.updating(event="finished")

    assert updated.event == "finished"
    assert event.event == "started"
    assert updated.created == event.created


def test_serialization_round_trip() -> None:
    event = ProcessingEvent.of("progress", "payload", meta={"key": "value"})

    assert ProcessingEvent.from_mapping(event.to_mapping()) == event


def test_created_defaults_when_constructed_directly() -> None:
    before = datetime.now(UTC)
    event = ProcessingEvent(event="direct", content=MultimodalContent.empty)

    assert event.created >= before
    assert event.meta == Meta.empty
