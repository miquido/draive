from collections.abc import Mapping
from datetime import datetime, time, timedelta

import pytest
from haiway import State

from draive.ollama.chat import _response_format, _schema_for_ollama


class _SimpleRecipe(State):
    name: str
    cuisine: str
    ingredients: list[str]
    steps: list[str]


class _UnsupportedSchema(State):
    metadata: dict[str, int]


class _UnionSchema(State):
    value: str | int


class _OptionalString(State):
    detail: str | None


class _UnsupportedUnion(State):
    value: str | bool


class _Written(State):
    start: datetime
    clock: time
    duration: timedelta
    late: datetime | None


def test_schema_for_ollama_preserves_supported_constructs() -> None:
    schema = _schema_for_ollama(_SimpleRecipe)

    assert schema == {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "cuisine": {"type": "string"},
            "ingredients": {"type": "array", "items": {"type": "string"}},
            "steps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "cuisine", "ingredients", "steps"],
        "additionalProperties": False,
    }


def test_schema_for_ollama_softens_additional_properties_only_when_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs: list[str] = []

    def fake_log_debug(message: str, *_, **__) -> None:
        logs.append(message)

    monkeypatch.setattr("draive.ollama.chat.ctx.log_debug", fake_log_debug, raising=False)

    schema = _schema_for_ollama(_UnsupportedSchema)

    assert schema == {
        "type": "object",
        "properties": {
            "metadata": {"type": "object"},
        },
        "required": ["metadata"],
        "additionalProperties": False,
    }
    assert logs, "Expected debug log when schema normalization removes keywords"


def test_schema_for_ollama_collapses_supported_unions(monkeypatch: pytest.MonkeyPatch) -> None:
    logs: list[str] = []

    def fake_log_debug(message: str, *_, **__) -> None:
        logs.append(message)

    monkeypatch.setattr("draive.ollama.chat.ctx.log_debug", fake_log_debug, raising=False)

    schema = _schema_for_ollama(_UnionSchema)

    assert schema == {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
        },
        "required": ["value"],
        "additionalProperties": False,
    }
    assert logs, "Expected debug log when schema normalization adjusts union types"


def test_schema_for_ollama_supports_nullable_scalars() -> None:
    schema = _schema_for_ollama(_OptionalString)

    assert schema == {
        "type": "object",
        "properties": {
            "detail": {"type": "string"},
        },
        "required": ["detail"],
        "additionalProperties": False,
    }


def test_schema_for_ollama_rejects_unsupported_constructs() -> None:
    assert _schema_for_ollama(_UnsupportedUnion) is None


def test_response_format_falls_back_to_generic_json(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []

    def fake_log_warning(message: str, *_, **__) -> None:
        warnings.append(message)

    monkeypatch.setattr("draive.ollama.chat.ctx.log_warning", fake_log_warning, raising=False)

    assert _response_format(_UnsupportedUnion) == "json"
    assert warnings, "Expected a warning when Ollama schema is unsupported"


def test_response_format_returns_plain_schema() -> None:
    schema = _response_format(_SimpleRecipe)

    assert isinstance(schema, Mapping)
    assert schema["type"] == "object"


def test_schema_for_ollama_names_written_formats_instead_of_patterns() -> None:
    schema = _schema_for_ollama(_Written)

    assert schema is not None
    properties: Mapping[str, Mapping[str, object]] = schema["properties"]
    assert properties["start"] == {"type": "string", "format": "date-time"}
    assert properties["clock"] == {"type": "string", "format": "time"}
    assert properties["duration"] == {
        "type": "string",
        "description": "ISO 8601 duration, e.g. PT45M",
    }
    assert properties["late"]["format"] == "date-time"
    assert all("pattern" not in value for value in properties.values())
