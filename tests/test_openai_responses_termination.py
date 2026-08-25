from collections.abc import AsyncIterator, Mapping, MutableSequence, Sequence
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from haiway import Meta, State, as_dict, ctx
from openai import omit

from draive.models import ModelOutputLimit, ModelTools, ModelToolSpecification
from draive.openai.config import OpenAIResponsesConfig
from draive.openai.responses import OpenAIResponses


class _FakeResponseStream:
    def __init__(self, events: Sequence[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> _FakeResponseStream:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[Any]:
        async def _iterator() -> AsyncIterator[Any]:
            for event in self._events:
                yield event

        return _iterator()


def _usage() -> Any:
    return SimpleNamespace(
        input_tokens=5,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
        output_tokens=7,
        output_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )


def _terminal_event(
    *,
    kind: str,
    incomplete_reason: str | None = None,
) -> Any:
    return SimpleNamespace(
        type=kind,
        response=SimpleNamespace(
            usage=_usage(),
            error=None,
            incomplete_details=(
                SimpleNamespace(reason=incomplete_reason) if incomplete_reason is not None else None
            ),
        ),
    )


def _model(
    events: Sequence[Any],
    *,
    requests: MutableSequence[dict[str, Any]] | None = None,
) -> OpenAIResponses:
    def _stream(**kwargs: Any) -> _FakeResponseStream:
        if requests is not None:
            requests.append(kwargs)

        return _FakeResponseStream(events)

    model = object.__new__(OpenAIResponses)
    model._base_url = None  # pyright: ignore[reportAttributeAccessIssue]
    model._client = SimpleNamespace(responses=SimpleNamespace(stream=_stream))  # pyright: ignore[reportAttributeAccessIssue]

    return model


async def _consume(model: OpenAIResponses) -> list[Any]:
    async with ctx.scope("test"):
        stream = model.completion(
            instructions="system",
            tools=ModelTools.none,
            context=(),
            output="text",
            config=OpenAIResponsesConfig(model="gpt-5", max_output_tokens=16),
        )
        return [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_truncated_response_raises_output_limit() -> None:
    # a response truncated by max_output_tokens terminates with `response.incomplete`,
    # which must not be mistaken for a successful completion
    model = _model(
        [
            _terminal_event(
                kind="response.incomplete",
                incomplete_reason="max_output_tokens",
            )
        ]
    )

    with pytest.raises(ModelOutputLimit):
        await _consume(model)


@pytest.mark.asyncio
async def test_completed_response_does_not_raise() -> None:
    model = _model([_terminal_event(kind="response.completed")])

    assert await _consume(model) == []


@pytest.mark.asyncio
async def test_requests_are_never_stored_and_carry_encrypted_reasoning() -> None:
    requests: list[dict[str, Any]] = []

    def _stream(**kwargs: Any) -> Any:
        requests.append(kwargs)
        return _FakeResponseStream(
            [
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(usage=None, error=None, incomplete_details=None),
                )
            ]
        )

    model = object.__new__(OpenAIResponses)
    model._base_url = None  # pyright: ignore[reportAttributeAccessIssue]
    model._client = SimpleNamespace(responses=SimpleNamespace(stream=_stream))  # pyright: ignore[reportAttributeAccessIssue]

    async with ctx.scope("test"):
        async for _ in model.completion(
            instructions="",
            tools=ModelTools.none,
            context=(),
            output="text",
            config=OpenAIResponsesConfig(model="gpt-5"),
        ):
            pass

    assert requests[0]["store"] is False
    assert requests[0]["include"] == ["reasoning.encrypted_content"]


@pytest.mark.asyncio
async def test_custom_base_url_omits_unsupported_include() -> None:
    requests: list[dict[str, Any]] = []

    def _stream(**kwargs: Any) -> Any:
        requests.append(kwargs)
        return _FakeResponseStream(
            [
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(usage=None, error=None, incomplete_details=None),
                )
            ]
        )

    model = object.__new__(OpenAIResponses)
    model._base_url = "https://compatible.example/v1"  # pyright: ignore[reportAttributeAccessIssue]
    model._client = SimpleNamespace(responses=SimpleNamespace(stream=_stream))  # pyright: ignore[reportAttributeAccessIssue]

    async with ctx.scope("test"):
        async for _ in model.completion(
            instructions="",
            tools=ModelTools.none,
            context=(),
            output="text",
            config=OpenAIResponsesConfig(model="gpt-5"),
        ):
            pass

    assert requests[0]["store"] is False
    assert requests[0]["include"] is omit


@pytest.mark.asyncio
async def test_schema_output_requests_strict_format() -> None:
    class Extracted(State, serializable=True):
        name: str
        note: str | None = None

    requests: list[dict[str, Any]] = []
    model = _model([_terminal_event(kind="response.completed")], requests=requests)

    async with ctx.scope("test"):
        async for _ in model.completion(
            instructions="",
            tools=ModelTools.none,
            context=(),
            output=Extracted,
            config=OpenAIResponsesConfig(model="gpt-5.6-terra"),
        ):
            pass

    text_format = requests[0]["text"]["format"]

    assert text_format["type"] == "json_schema"
    assert text_format["name"] == "Extracted"
    assert text_format["strict"] is True
    # strict mode demands a value for every field instead of leaving it to a default
    assert text_format["schema"]["required"] == ["name", "note"]


@pytest.mark.asyncio
async def test_schema_output_falls_back_when_strict_is_impossible() -> None:
    class Annotated(State, serializable=True):
        name: str
        meta: Meta = Meta.empty

    requests: list[dict[str, Any]] = []
    model = _model([_terminal_event(kind="response.completed")], requests=requests)

    async with ctx.scope("test"):
        async for _ in model.completion(
            instructions="",
            tools=ModelTools.none,
            context=(),
            output=Annotated,
            config=OpenAIResponsesConfig(model="gpt-5.6-terra"),
        ):
            pass

    text_format = requests[0]["text"]["format"]

    assert text_format["strict"] is False
    assert text_format["schema"] == as_dict(Annotated.__SPECIFICATION__)


@pytest.mark.asyncio
async def test_unsupported_output_selection_is_reported_as_itself() -> None:
    model = _model([_terminal_event(kind="response.completed")])

    async with ctx.scope("test"):
        with pytest.raises(NotImplementedError):
            async for _ in model.completion(
                instructions="",
                tools=ModelTools.none,
                context=(),
                output="audio",
                config=OpenAIResponsesConfig(model="gpt-5.6-terra"),
            ):
                pass


@pytest.mark.asyncio
async def test_reasoning_parameters_are_forwarded() -> None:
    requests: list[dict[str, Any]] = []
    model = _model([_terminal_event(kind="response.completed")], requests=requests)

    async with ctx.scope("test"):
        async for _ in model.completion(
            instructions="",
            tools=ModelTools.none,
            context=(),
            output="text",
            config=OpenAIResponsesConfig(
                model="gpt-5.6-terra",
                reasoning="high",
                reasoning_summary="detailed",
                reasoning_context="current_turn",
                reasoning_mode="pro",
                prompt_cache_retention="24h",
            ),
        ):
            pass

    assert requests[0]["reasoning"] == {
        "effort": "high",
        "summary": "detailed",
        "context": "current_turn",
        "mode": "pro",
    }
    assert requests[0]["prompt_cache_retention"] == "24h"


@pytest.mark.asyncio
async def test_reasoning_is_omitted_when_unconfigured() -> None:
    requests: list[dict[str, Any]] = []
    model = _model([_terminal_event(kind="response.completed")], requests=requests)

    async with ctx.scope("test"):
        async for _ in model.completion(
            instructions="",
            tools=ModelTools.none,
            context=(),
            output="text",
            config=OpenAIResponsesConfig(model="gpt-5.6-terra"),
        ):
            pass

    assert requests[0]["reasoning"] is omit
    assert requests[0]["prompt_cache_retention"] is omit


@pytest.mark.asyncio
async def test_strict_tool_parameters_demand_every_argument() -> None:
    requests: list[dict[str, Any]] = []
    model = _model([_terminal_event(kind="response.completed")], requests=requests)

    async with ctx.scope("test"):
        async for _ in model.completion(
            instructions="",
            tools=ModelTools.of(
                ModelToolSpecification.of(
                    name="search",
                    description="Search records",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    meta={"strict_parameters": True},
                )
            ),
            context=(),
            output="text",
            config=OpenAIResponsesConfig(model="gpt-5.6-terra"),
        ):
            pass

    tool_param = requests[0]["tools"][0]

    assert tool_param["strict"] is True
    assert tool_param["parameters"]["required"] == ["query", "limit"]


@pytest.mark.asyncio
async def test_strict_tool_parameters_fall_back_when_impossible() -> None:
    requests: list[dict[str, Any]] = []
    model = _model([_terminal_event(kind="response.completed")], requests=requests)
    specification = ModelToolSpecification.of(
        name="annotate",
        description="Annotate",
        parameters={
            "type": "object",
            "properties": {"meta": {"type": "object", "additionalProperties": True}},
            "required": [],
            "additionalProperties": False,
        },
        meta={"strict_parameters": True},
    )

    async with ctx.scope("test"):
        async for _ in model.completion(
            instructions="",
            tools=ModelTools.of(specification),
            context=(),
            output="text",
            config=OpenAIResponsesConfig(model="gpt-5.6-terra"),
        ):
            pass

    tool_param = requests[0]["tools"][0]

    assert tool_param["strict"] is False
    assert tool_param["parameters"] == specification.parameters


@pytest.mark.asyncio
async def test_tool_parameters_stay_untouched_without_strict() -> None:
    requests: list[dict[str, Any]] = []
    model = _model([_terminal_event(kind="response.completed")], requests=requests)
    specification = ModelToolSpecification.of(
        name="search",
        description="Search records",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )

    async with ctx.scope("test"):
        async for _ in model.completion(
            instructions="",
            tools=ModelTools.of(specification),
            context=(),
            output="text",
            config=OpenAIResponsesConfig(model="gpt-5.6-terra"),
        ):
            pass

    tool_param = requests[0]["tools"][0]

    assert tool_param["strict"] is False
    assert tool_param["parameters"] == specification.parameters


@pytest.mark.asyncio
async def test_completion_leaves_the_output_model_untouched() -> None:
    class Extracted(State, serializable=True):
        name: str
        note: str | None = None

    before = deepcopy(as_dict(Extracted.__SPECIFICATION__))
    schema_before = Extracted.json_schema(indent=2)
    requests: list[dict[str, Any]] = []
    model = _model([_terminal_event(kind="response.completed")], requests=requests)

    async with ctx.scope("test"):
        async for _ in model.completion(
            instructions="",
            tools=ModelTools.none,
            context=(),
            output=Extracted,
            config=OpenAIResponsesConfig(model="gpt-5.6-terra"),
        ):
            pass

    # the strict conversion builds its own mapping, the declaration stays as written
    assert requests[0]["text"]["format"]["strict"] is True
    assert as_dict(Extracted.__SPECIFICATION__) == before
    assert Extracted.json_schema(indent=2) == schema_before


@pytest.mark.asyncio
async def test_completion_leaves_the_tool_specification_untouched() -> None:
    specification = ModelToolSpecification.of(
        name="search",
        description="Search records",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        meta={"strict_parameters": True},
    )
    before = deepcopy(dict(specification.parameters or {}))
    requests: list[dict[str, Any]] = []
    model = _model([_terminal_event(kind="response.completed")], requests=requests)

    async with ctx.scope("test"):
        async for _ in model.completion(
            instructions="",
            tools=ModelTools.of(specification),
            context=(),
            output="text",
            config=OpenAIResponsesConfig(model="gpt-5.6-terra"),
        ):
            pass

    assert requests[0]["tools"][0]["strict"] is True
    assert dict(specification.parameters or {}) == before


@pytest.mark.asyncio
async def test_plain_output_selections_request_no_schema() -> None:
    for output, expected in (
        ("auto", omit),
        ("text", {"format": {"type": "text"}}),
        ("json", omit),
    ):
        requests: list[dict[str, Any]] = []
        model = _model([_terminal_event(kind="response.completed")], requests=requests)

        async with ctx.scope("test"):
            async for _ in model.completion(
                instructions="",
                tools=ModelTools.none,
                context=(),
                output=output,  # pyright: ignore[reportArgumentType]
                config=OpenAIResponsesConfig(model="gpt-5.6-terra"),
            ):
                pass

        assert requests[0]["text"] == expected


@pytest.mark.asyncio
async def test_strict_tool_parameters_fall_back_for_an_open_mapping() -> None:
    specification = ModelToolSpecification.of(
        name="record",
        description="Record labels",
        parameters={
            "type": "object",
            "properties": {
                "reference": {"type": "string"},
                "labels": {"type": "object", "additionalProperties": {"type": "string"}},
            },
            "required": ["reference"],
            "additionalProperties": False,
        },
        meta={"strict_parameters": True},
    )
    requests: list[dict[str, Any]] = []
    model = _model([_terminal_event(kind="response.completed")], requests=requests)

    async with ctx.scope("test"):
        async for _ in model.completion(
            instructions="",
            tools=ModelTools.of(specification),
            context=(),
            output="text",
            config=OpenAIResponsesConfig(model="gpt-5.6-terra"),
        ):
            pass

    tool_param = requests[0]["tools"][0]

    # enforcing it would strip `labels` out of the argument object entirely
    assert tool_param["strict"] is False
    assert tool_param["parameters"] == specification.parameters


@pytest.mark.asyncio
async def test_schema_output_keeps_strict_for_an_open_mapping() -> None:
    class Extracted(State, serializable=True):
        name: str
        labels: Mapping[str, str] = {}

    requests: list[dict[str, Any]] = []
    model = _model([_terminal_event(kind="response.completed")], requests=requests)

    async with ctx.scope("test"):
        async for _ in model.completion(
            instructions="",
            tools=ModelTools.none,
            context=(),
            output=Extracted,
            config=OpenAIResponsesConfig(model="gpt-5.6-terra"),
        ):
            pass

    text_format = requests[0]["text"]["format"]

    # a requested output keeps the mapping fillable, unlike a tool argument object
    assert text_format["strict"] is True
    assert text_format["schema"]["required"] == ["name"]
    assert "labels" in text_format["schema"]["properties"]
