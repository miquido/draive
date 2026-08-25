"""Provider context encoding has to survive a round trip through the model context."""

from collections.abc import Sequence
from typing import Any

import pytest
from google.genai.types import FunctionCall, Part
from haiway import Meta, State

from draive.bedrock.converse import _context_messages as _bedrock_messages
from draive.bedrock.converse import _request_messages as _bedrock_request_messages
from draive.bedrock.converse import _verify_output as _bedrock_verify_output
from draive.gemini.content import block_parts, function_response, part_as_stream_elements
from draive.mistral.completions import _context_messages as _mistral_messages
from draive.models import (
    ModelInput,
    ModelOutput,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelToolResponse,
)
from draive.multimodal import MultimodalContent, TextContent
from draive.ollama.chat import _context_messages as _ollama_messages
from draive.resources import ResourceContent, ResourceReference
from draive.vllm.messages import _context_messages as _vllm_messages


def _reasoning_chunk(
    text: str,
    *,
    final: bool = False,
    **meta: Any,
) -> ModelReasoningChunk:
    return ModelReasoningChunk.of(
        TextContent.of(text),
        final=final,
        meta=Meta.of({"kind": "thought", **meta}),
    )


class TestReasoningBlocks:
    def test_fragments_group_into_a_single_block_by_default(self) -> None:
        blocks = ModelReasoning.blocks(
            (
                _reasoning_chunk("first "),
                _reasoning_chunk("second"),
            )
        )

        assert len(blocks) == 1
        assert blocks[0].reasoning.to_str() == "first second"

    def test_each_final_fragment_closes_its_own_block(self) -> None:
        # merging the fragments of two provider blocks would attach the identity of
        # one to the text of the other
        blocks = ModelReasoning.blocks(
            (
                _reasoning_chunk("alpha", final=True, signature="sig-a"),
                _reasoning_chunk("beta", final=True, signature="sig-b"),
            )
        )

        assert [block.reasoning.to_str() for block in blocks] == ["alpha", "beta"]
        assert [block.meta.get_str("signature") for block in blocks] == ["sig-a", "sig-b"]

    def test_a_block_left_unterminated_is_still_reported(self) -> None:
        blocks = ModelReasoning.blocks(
            (
                _reasoning_chunk("closed", final=True, signature="sig-a"),
                _reasoning_chunk("trailing"),
            )
        )

        assert [block.reasoning.to_str() for block in blocks] == ["closed", "trailing"]
        assert blocks[0].meta.get_str("signature") == "sig-a"
        assert blocks[1].meta.get_str("signature") is None

    def test_no_fragments_produce_no_blocks(self) -> None:
        assert ModelReasoning.blocks(()) == []


class TestGeminiReasoningSignatures:
    def test_an_unsigned_fragment_does_not_erase_a_preceding_signature(self) -> None:
        # metadata is merged within a block, an explicit None would overwrite it
        signed = list(
            part_as_stream_elements(
                Part(text="thinking", thought=True, thought_signature=b"signature")
            )
        )
        unsigned = list(part_as_stream_elements(Part(text="more", thought=True)))

        blocks = ModelReasoning.blocks(
            [chunk for chunk in (*signed, *unsigned) if isinstance(chunk, ModelReasoningChunk)]
        )

        # the signature closes its block, the unsigned fragment starts another
        assert len(blocks) == 2
        assert blocks[0].meta.get_str("signature") is not None
        assert blocks[1].meta.get_str("signature") is None

    def test_a_signed_fragment_closes_its_block(self) -> None:
        chunks = list(
            part_as_stream_elements(
                Part(text="thinking", thought=True, thought_signature=b"signature")
            )
        )

        assert isinstance(chunks[0], ModelReasoningChunk)
        assert chunks[0].final is True

    def test_an_unsigned_fragment_stays_open(self) -> None:
        chunks = list(part_as_stream_elements(Part(text="thinking", thought=True)))

        assert isinstance(chunks[0], ModelReasoningChunk)
        assert chunks[0].final is False

    def test_a_signature_survives_a_context_round_trip(self) -> None:
        chunks = [
            chunk
            for chunk in part_as_stream_elements(
                Part(text="thinking", thought=True, thought_signature=b"signature")
            )
            if isinstance(chunk, ModelReasoningChunk)
        ]
        blocks = ModelReasoning.blocks(chunks)

        parts = list(block_parts(blocks))

        assert parts == [
            {
                "text": "thinking",
                "thought": True,
                "thought_signature": b"signature",
            }
        ]

    def test_a_tool_call_without_an_identifier_gets_a_local_one(self) -> None:
        # the api reports none outside of live sessions while the response has to
        # correlate with its request
        requests = list(
            part_as_stream_elements(Part(function_call=FunctionCall(name="echo", args={"x": 1})))
        )

        assert len(requests) == 1
        assert isinstance(requests[0], ModelToolRequest)
        assert requests[0].tool == "echo"
        assert requests[0].identifier


class TestGeminiToolResponses:
    def test_text_travels_within_the_payload(self) -> None:
        encoded = function_response(
            ModelToolResponse.of(
                "call-1",
                tool="echo",
                content=MultimodalContent.of("result"),
            )
        )

        assert encoded["response"] == {"output": "result"}
        assert "parts" not in encoded

    def test_an_error_selects_the_error_payload_key(self) -> None:
        encoded = function_response(
            ModelToolResponse.of(
                "call-1",
                tool="echo",
                status="error",
                content=MultimodalContent.of("broken"),
            )
        )

        assert encoded["response"] == {"error": "broken"}

    def test_media_travels_within_dedicated_parts(self) -> None:
        # inlining it into the payload would deliver a base64 blob the model
        # cannot interpret as media
        encoded = function_response(
            ModelToolResponse.of(
                "call-1",
                tool="render",
                content=MultimodalContent.of(
                    TextContent.of("here"),
                    ResourceContent.of(b"\x89PNG", mime_type="image/png"),
                ),
            )
        )

        assert encoded["response"] == {"output": "here"}
        assert encoded["parts"] == [
            {
                "inline_data": {
                    "data": b"\x89PNG",
                    "mime_type": "image/png",
                }
            }
        ]

    def test_references_travel_within_dedicated_parts(self) -> None:
        encoded = function_response(
            ModelToolResponse.of(
                "call-1",
                tool="render",
                content=MultimodalContent.of(
                    ResourceReference.of("gs://bucket/file.png", mime_type="image/png")
                ),
            )
        )

        assert encoded["parts"] == [
            {
                "file_data": {
                    "file_uri": "gs://bucket/file.png",
                    "mime_type": "image/png",
                }
            }
        ]


class TestBedrockJsonOutput:
    class Example(State, serializable=True):
        name: str

    @pytest.mark.parametrize("output", ("json", Example))
    def test_json_output_leaves_the_context_untouched(self, output: Any) -> None:
        # the Converse API implements neither json mode, so both selections fall
        # back to plain output without the provider appending content of its own
        context = (
            ModelInput.of(MultimodalContent.of("question")),
            ModelOutput.of(MultimodalContent.of("answer")),
        )

        assert _bedrock_request_messages(context, output=output) == _bedrock_messages(context)

    @pytest.mark.parametrize("output", ("auto", "text", "json", ("text",), Example))
    def test_deliverable_output_selections_are_accepted(self, output: Any) -> None:
        assert _bedrock_verify_output(output) is None

    @pytest.mark.parametrize("output", ("image", "audio", "video"))
    def test_unsupported_output_selections_are_rejected(self, output: Any) -> None:
        with pytest.raises(NotImplementedError):
            _bedrock_verify_output(output)


class TestVLLMMessages:
    def test_empty_instructions_produce_no_system_message(self) -> None:
        messages = list(
            _vllm_messages(
                instructions="",
                context=(ModelInput.of(MultimodalContent.of("question")),),
                vision_details="auto",
            )
        )

        assert [message["role"] for message in messages] == ["user"]

    def test_instructions_produce_a_system_message(self) -> None:
        messages = list(
            _vllm_messages(
                instructions="system",
                context=(ModelInput.of(MultimodalContent.of("question")),),
                vision_details="auto",
            )
        )

        assert [message["role"] for message in messages] == ["system", "user"]

    def test_an_assistant_turn_without_tool_calls_omits_the_key(self) -> None:
        # an empty tool calls array is rejected by the api
        messages = list(
            _vllm_messages(
                instructions="",
                context=(ModelOutput.of(MultimodalContent.of("answer")),),
                vision_details="auto",
            )
        )

        assert len(messages) == 1
        assert "tool_calls" not in messages[0]

    def test_an_assistant_turn_with_tool_calls_keeps_the_key(self) -> None:
        messages = list(
            _vllm_messages(
                instructions="",
                context=(
                    ModelOutput.of(
                        ModelToolRequest.of("call-1", tool="echo", arguments={"x": 1}),
                    ),
                ),
                vision_details="auto",
            )
        )

        assert len(messages) == 1
        tool_calls: Sequence[Any] = messages[0]["tool_calls"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
        assert [call["id"] for call in tool_calls] == ["call-1"]

    def test_a_reasoning_only_turn_is_skipped(self) -> None:
        # reasoning is unsupported here, keeping the turn would send a message
        # carrying neither content nor tool calls
        messages = list(
            _vllm_messages(
                instructions="",
                context=(ModelOutput.of(ModelReasoning.of(TextContent.of("thinking"))),),
                vision_details="auto",
            )
        )

        assert messages == []


class TestOllamaReasoningReplay:
    def test_reasoning_travels_through_the_thinking_field(self) -> None:
        # dropping it would break the thinking continuity across turns
        messages = list(
            _ollama_messages(
                instructions="",
                context=(
                    ModelOutput.of(
                        ModelReasoning.of(TextContent.of("thinking")),
                        MultimodalContent.of("answer"),
                    ),
                ),
            )
        )

        assert len(messages) == 1
        assert messages[0].thinking == "thinking"
        assert messages[0].content == "answer"

    def test_a_turn_without_reasoning_carries_no_thinking(self) -> None:
        messages = list(
            _ollama_messages(
                instructions="",
                context=(ModelOutput.of(MultimodalContent.of("answer")),),
            )
        )

        assert messages[0].thinking is None


class TestMistralReasoningReplay:
    def test_reasoning_travels_as_a_thinking_chunk(self) -> None:
        messages = list(
            _mistral_messages(
                (
                    ModelOutput.of(
                        ModelReasoning.of(
                            TextContent.of("thinking"),
                            meta={"kind": "thinking", "signature": "sig-1"},
                        ),
                    ),
                )
            )
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == [
            {
                "type": "thinking",
                "thinking": [{"type": "text", "text": "thinking"}],
                "closed": True,
                "signature": "sig-1",
            }
        ]

    def test_reasoning_without_a_signature_omits_the_key(self) -> None:
        messages = list(
            _mistral_messages(
                (ModelOutput.of(ModelReasoning.of(TextContent.of("thinking"))),),
            )
        )

        assert "signature" not in messages[0]["content"][0]
