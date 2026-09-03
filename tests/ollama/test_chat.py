import json
import warnings
from types import SimpleNamespace
from typing import Any

import pytest
from haiway import ctx
from ollama import Message
from ollama._client import _copy_tools
from ollama._types import ChatRequest

from draive.models import (
    ModelOutputFailed,
    ModelOutputLimit,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelTools,
    ModelToolSpecification,
)
from draive.multimodal import MultimodalContent, TextContent
from draive.ollama.chat import (
    OllamaChat,
    _content_images,
    _context_messages,
    _tool_specification_as_tool,
)
from draive.ollama.config import OllamaChatConfig
from draive.resources import ResourceContent, ResourceReference


def test_context_messages_includes_system_instructions() -> None:
    messages = list(_context_messages(instructions="Stay concise.", context=()))
    assert len(messages) == 1
    assert messages[0].role == "system"
    assert messages[0].content == "Stay concise."


def _stream_of(*chunks: Any) -> Any:
    """Build a fake ollama streaming response - `chat(stream=True)` awaits to an async iterator."""

    async def _chat(**kwargs: Any) -> Any:
        async def _iterator() -> Any:
            for chunk in chunks:
                yield chunk

        _chat.recorded = kwargs  # pyright: ignore[reportFunctionMemberAccess]
        return _iterator()

    return _chat


def _chunk(
    *,
    content: str | None = None,
    thinking: str | None = None,
    tool_calls: Any = None,
    done: bool = False,
    done_reason: str | None = None,
    prompt_eval_count: int | None = None,
    eval_count: int | None = None,
) -> Any:
    return SimpleNamespace(
        message=SimpleNamespace(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
        ),
        done=done,
        done_reason=done_reason,
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
    )


async def _collect(model: Any, **overrides: Any) -> list[Any]:
    arguments: dict[str, Any] = {
        "instructions": "system",
        "tools": ModelTools.none,
        "context": (),
        "output": "text",
        "config": OllamaChatConfig(model="ollama-test"),
        "prefill": None,
    }
    arguments.update(overrides)
    return [chunk async for chunk in model.completion(**arguments)]


@pytest.mark.asyncio
async def test_completion_translates_provider_errors_to_model_output_failed() -> None:
    async def _chat(**_: Any) -> Any:
        raise RuntimeError("connection lost")

    model = object.__new__(OllamaChat)
    model._client = SimpleNamespace(chat=_chat)

    async with ctx.scope("test"):
        with pytest.raises(ModelOutputFailed, match="connection lost"):
            _ = await _collect(model)


@pytest.mark.asyncio
async def test_completion_streams_text_fragments_incrementally() -> None:
    """Streaming chunks carry fragments, each has to be delivered as it arrives."""
    chat = _stream_of(
        _chunk(content="Hel"),
        _chunk(content="lo, "),
        _chunk(content="world!", done=True, done_reason="stop", prompt_eval_count=7, eval_count=3),
    )
    model = object.__new__(OllamaChat)
    model._client = SimpleNamespace(chat=chat)

    async with ctx.scope("test"):
        chunks = await _collect(model)

    texts = [chunk for chunk in chunks if isinstance(chunk, TextContent)]
    assert [text.text for text in texts] == ["Hel", "lo, ", "world!"]
    # a real stream is requested, not a single blocking response
    assert chat.recorded["stream"] is True  # pyright: ignore[reportFunctionMemberAccess]


@pytest.mark.asyncio
async def test_completion_reports_tool_requests_with_decoded_arguments() -> None:
    # ollama decodes tool call arguments into a mapping and delivers each call whole
    chat = _stream_of(
        _chunk(
            tool_calls=[
                SimpleNamespace(function=SimpleNamespace(name="lookup", arguments={"value": 42}))
            ],
            done=True,
            done_reason="stop",
            prompt_eval_count=1,
            eval_count=1,
        )
    )
    model = object.__new__(OllamaChat)
    model._client = SimpleNamespace(chat=chat)

    async with ctx.scope("test"):
        chunks = await _collect(model)

    requests = [chunk for chunk in chunks if isinstance(chunk, ModelToolRequest)]
    assert len(requests) == 1
    assert requests[0].tool == "lookup"
    assert requests[0].arguments == {"value": 42}


@pytest.mark.asyncio
async def test_completion_requests_and_reports_thinking() -> None:
    chat = _stream_of(
        _chunk(thinking="delib"),
        _chunk(thinking="erating"),
        _chunk(content="answer", done=True, done_reason="stop", prompt_eval_count=5, eval_count=2),
    )
    model = object.__new__(OllamaChat)
    model._client = SimpleNamespace(chat=chat)

    async with ctx.scope("test"):
        chunks = await _collect(
            model,
            config=OllamaChatConfig(model="ollama-test", thinking="high"),
        )

    # reasoning models only report thinking when it is explicitly requested
    assert chat.recorded["think"] == "high"  # pyright: ignore[reportFunctionMemberAccess]
    reasoning = [chunk for chunk in chunks if isinstance(chunk, ModelReasoningChunk)]
    assert [block.reasoning_chunk.to_str() for block in reasoning] == ["delib", "erating"]


@pytest.mark.asyncio
async def test_completion_reports_output_limit_on_length_stop() -> None:
    chat = _stream_of(
        _chunk(content="truncated"),
        _chunk(done=True, done_reason="length", prompt_eval_count=4, eval_count=64),
    )
    model = object.__new__(OllamaChat)
    model._client = SimpleNamespace(chat=chat)

    async with ctx.scope("test"):
        with pytest.raises(ModelOutputLimit):
            _ = await _collect(
                model,
                config=OllamaChatConfig(model="ollama-test", max_output_tokens=64),
            )


def test_inline_image_serializes_as_raw_base64_data() -> None:
    # `ollama.Image` accepts neither urls nor data uris, only raw base64 payloads
    resource = ResourceContent.of(b"\x89PNG\r\n\x1a\nimage-bytes", mime_type="image/png")
    images = _content_images(MultimodalContent.of(resource))
    assert images is not None
    assert len(images) == 1

    request = ChatRequest(
        model="ollama-test",
        messages=[Message(role="user", content="describe", images=images)],
    )
    serialized = request.model_dump(exclude_none=True)
    assert serialized["messages"][0]["images"] == [resource.data]


def test_image_reference_is_rejected_with_explicit_error() -> None:
    content = MultimodalContent.of(
        ResourceReference.of("https://example.com/image.png", mime_type="image/png")
    )
    with pytest.raises(ValueError, match=r"image reference \(image/png\)") as error:
        _content_images(content)

    # a uri can carry credentials within its userinfo or query, it must not be echoed
    assert "example.com" not in str(error.value)


def test_nested_tool_parameters_survive_serialization() -> None:
    # `Tool.Function.Parameters` validation would erase nested schemas
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "nested": {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
                "additionalProperties": False,
            },
        },
        "required": ["nested"],
        "additionalProperties": False,
    }
    tool = _tool_specification_as_tool(
        ModelToolSpecification(
            name="probe",
            description="probe",
            parameters=parameters,
        )
    )

    request = ChatRequest(
        model="ollama-test",
        messages=[Message(role="user", content="probe")],
        # the client copies provided tools through `Tool.model_validate`
        tools=list(_copy_tools([tool])),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        serialized = request.model_dump(exclude_none=True)

    # sequences are normalized into tuples by the specification validation
    assert json.loads(json.dumps(serialized["tools"][0]["function"]["parameters"])) == parameters
