from base64 import b64encode
from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai.types import (
    BlockedReason,
    Candidate,
    Content,
    FinishReason,
    FunctionCall,
    GenerateContentResponse,
    GenerateContentResponsePromptFeedback,
    HarmCategory,
    Part,
    SafetyRating,
)
from haiway import ctx

from draive.gemini.config import GeminiConfig, GeminiSafetyConfig
from draive.gemini.content import block_parts
from draive.gemini.generating import GeminiGenerating, _request_config
from draive.models import (
    ModelInputInvalid,
    ModelOutput,
    ModelOutputFailed,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelTools,
)
from draive.multimodal import MultimodalContent


def _iter_async[T](items: Sequence[T]) -> AsyncIterator[T]:
    async def _iterator() -> AsyncIterator[T]:
        for item in items:
            yield item

    return _iterator()


@pytest.mark.asyncio
async def test_gemini_stream_fails_on_blocked_prompt() -> None:
    chunk = GenerateContentResponse(
        prompt_feedback=GenerateContentResponsePromptFeedback(
            block_reason=BlockedReason.PROHIBITED_CONTENT,
            block_reason_message="Prompt violates policies",
            safety_ratings=[
                SafetyRating(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    blocked=True,
                )
            ],
        ),
    )

    async def _generate_content_stream(**_: Any) -> AsyncIterator[Any]:
        return _iter_async([chunk])

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content_stream=_generate_content_stream,
            )
        )
    )

    model = object.__new__(GeminiGenerating)
    model._client = fake_client

    async with ctx.scope("test"):
        stream = model.completion(
            instructions="system",
            context=(),
            tools=ModelTools.none,
            output="text",
            config=GeminiConfig(model="gemini-test"),
        )
        with pytest.raises(ModelOutputFailed) as failure:
            _ = [element async for element in stream]

    assert "PROHIBITED_CONTENT" in failure.value.reason
    assert "Prompt violates policies" in failure.value.reason


def test_request_config_safety_settings_use_supported_categories() -> None:
    configuration = _request_config(
        instructions="",
        tools=ModelTools.none,
        output="text",
        config=GeminiConfig(model="gemini-test", safety=GeminiSafetyConfig()),
    )

    categories = [setting["category"] for setting in configuration["safety_settings"]]
    # image specific categories are rejected by the api with an invalid argument error
    assert categories == [
        HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        HarmCategory.HARM_CATEGORY_HARASSMENT,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
        HarmCategory.HARM_CATEGORY_JAILBREAK,
    ]


def test_request_config_omits_safety_settings_by_default() -> None:
    configuration = _request_config(
        instructions="",
        tools=ModelTools.none,
        output="text",
        config=GeminiConfig(model="gemini-test"),
    )

    assert "safety_settings" not in configuration


def _fake_client(chunks: Sequence[Any]) -> SimpleNamespace:
    async def _generate_content_stream(**_: Any) -> AsyncIterator[Any]:
        return _iter_async(chunks)

    return SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content_stream=_generate_content_stream)
        )
    )


def _response(*parts: Part, finish_reason: FinishReason | None = None) -> GenerateContentResponse:
    return GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(role="model", parts=list(parts)),
                finish_reason=finish_reason,
            )
        ]
    )


@pytest.mark.asyncio
async def test_gemini_stream_drops_the_detached_turn_signature() -> None:
    # the turn signature arrives on a trailing part of its own, after the answer text,
    # detached from the reasoning it belongs to - only reasoning itself is streamed
    model = object.__new__(GeminiGenerating)
    model._client = _fake_client(
        [
            _response(Part(text="thinking", thought=True)),
            _response(Part(text="answer")),
            _response(
                Part(text="", thought_signature=b"signature"),
                finish_reason=FinishReason.STOP,
            ),
        ]
    )

    async with ctx.scope("test"):
        elements = [
            element
            async for element in model.completion(
                instructions="",
                context=(),
                tools=ModelTools.none,
                output="text",
                config=GeminiConfig(model="gemini-test"),
            )
        ]

    reasoning = [element for element in elements if isinstance(element, ModelReasoningChunk)]
    assert len(reasoning) == 1
    assert reasoning[0].reasoning_chunk.text == "thinking"
    assert "signature" not in reasoning[0].meta

    # nothing carries the signature back to the provider on a context replay
    blocks = ModelReasoning.blocks(reasoning)
    assert all("thought_signature" not in part for part in block_parts(blocks))


@pytest.mark.asyncio
async def test_gemini_stream_keeps_a_tool_request_signature() -> None:
    # function call signatures are the ones validated on replay, they ride the call part
    model = object.__new__(GeminiGenerating)
    model._client = _fake_client(
        [
            _response(
                Part(
                    function_call=FunctionCall(id="call_id", name="tool", args={}),
                    thought_signature=b"signature",
                ),
                finish_reason=FinishReason.STOP,
            ),
        ]
    )

    async with ctx.scope("test"):
        elements = [
            element
            async for element in model.completion(
                instructions="",
                context=(),
                tools=ModelTools.none,
                output="text",
                config=GeminiConfig(model="gemini-test"),
            )
        ]

    requests = [element for element in elements if isinstance(element, ModelToolRequest)]
    assert len(requests) == 1
    assert requests[0].meta.get_str("signature") == b64encode(b"signature").decode()
    assert next(iter(block_parts(requests)))["thought_signature"] == b"signature"


@pytest.mark.asyncio
async def test_gemini_reports_invalid_context_as_input_failure() -> None:
    model = object.__new__(GeminiGenerating)
    model._client = _fake_client([])

    async with ctx.scope("test"):
        with pytest.raises(ModelInputInvalid):
            _ = [
                element
                async for element in model.completion(
                    instructions="",
                    context=(
                        ModelOutput.of(
                            ModelReasoning.of(
                                MultimodalContent.of("secret"),
                                meta={"kind": "encrypted"},
                            )
                        ),
                    ),
                    tools=ModelTools.none,
                    output="text",
                    config=GeminiConfig(model="gemini-test"),
                )
            ]


def test_request_config_rejects_unsupported_output() -> None:
    # an unsupported configuration is a caller error, not a failed generation
    with pytest.raises(NotImplementedError):
        _request_config(
            instructions="",
            tools=ModelTools.none,
            output="video",
            config=GeminiConfig(model="gemini-test"),
        )
