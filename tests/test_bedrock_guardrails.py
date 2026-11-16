from typing import Any

import pytest
from haiway import ctx

from draive.bedrock.config import BedrockInputGuardraisConfig, BedrockOutputGuardraisConfig
from draive.bedrock.guardrails import BedrockGuardrails
from draive.guardrails import GuardrailsFailure, GuardrailsModerationException
from draive.multimodal import MultimodalContent
from draive.resources import ResourceContent

_INPUT_CONFIG = BedrockInputGuardraisConfig(guardrail_identifier="gid", guardrail_version="1")
_OUTPUT_CONFIG = BedrockOutputGuardraisConfig(guardrail_identifier="gid", guardrail_version="1")


class _FakeClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response: dict[str, Any] = response
        self.calls: list[dict[str, Any]] = []

    def apply_guardrail(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.response


def _guardrails(response: dict[str, Any]) -> tuple[BedrockGuardrails, _FakeClient]:
    guardrails = BedrockGuardrails()
    client = _FakeClient(response)
    guardrails._client = client  # pyright: ignore[reportPrivateUsage]
    return guardrails, client


@pytest.mark.asyncio
async def test_bedrock_guardrails_pass_without_intervention() -> None:
    guardrails, client = _guardrails({"action": "NONE"})
    async with ctx.scope("test"):
        await guardrails.content_input_verification("hi", config=_INPUT_CONFIG)
        await guardrails.content_output_verification("hi", config=_OUTPUT_CONFIG)

    assert [call["source"] for call in client.calls] == ["INPUT", "OUTPUT"]
    assert client.calls[0]["content"] == [{"text": {"text": "hi", "qualifiers": ["query"]}}]
    assert client.calls[1]["content"] == [{"text": {"text": "hi", "qualifiers": ["guard_content"]}}]
    assert client.calls[1]["outputScope"] == "INTERVENTIONS"


@pytest.mark.asyncio
async def test_bedrock_input_intervention_reports_detected_content_filters() -> None:
    guardrails, _ = _guardrails(
        {
            "action": "GUARDRAIL_INTERVENED",
            "assessments": [
                {
                    "contentPolicy": {
                        "filters": [
                            {"type": "HATE", "detected": True},
                            {"type": "VIOLENCE", "detected": False},
                        ]
                    }
                }
            ],
        }
    )
    async with ctx.scope("test"):
        with pytest.raises(GuardrailsModerationException) as raised:
            await guardrails.content_input_verification("hi", config=_INPUT_CONFIG)

    assert dict(raised.value.violations) == {"HATE": 1.0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "assessment",
    (
        {"topicPolicy": {"topics": [{"name": "x", "type": "DENY", "action": "BLOCKED"}]}},
        {
            "sensitiveInformationPolicy": {
                "piiEntities": [{"type": "EMAIL", "action": "BLOCKED", "match": "a@b.c"}]
            }
        },
        {"wordPolicy": {"customWords": [{"match": "x", "action": "BLOCKED"}]}},
    ),
)
async def test_bedrock_input_intervention_reports_non_content_policies(
    assessment: dict[str, Any],
) -> None:
    guardrails, _ = _guardrails({"action": "GUARDRAIL_INTERVENED", "assessments": [assessment]})
    async with ctx.scope("test"):
        with pytest.raises(GuardrailsModerationException) as raised:
            await guardrails.content_input_verification("hi", config=_INPUT_CONFIG)

    assert raised.value.violations


@pytest.mark.asyncio
async def test_bedrock_output_intervention_reports_guardrail_identifier() -> None:
    guardrails, _ = _guardrails({"action": "GUARDRAIL_INTERVENED", "actionReason": "blocked"})
    async with ctx.scope("test"):
        with pytest.raises(GuardrailsModerationException, match="blocked") as raised:
            await guardrails.content_output_verification("hi", config=_OUTPUT_CONFIG)

    assert dict(raised.value.violations) == {"gid": 1.0}


@pytest.mark.asyncio
async def test_bedrock_guardrails_convert_supported_images_and_text_fallback() -> None:
    guardrails, client = _guardrails({"action": "NONE"})
    png: bytes = b"\x89PNGdata"
    content = MultimodalContent.of(
        "caption",
        ResourceContent.of(png, mime_type="image/png"),
        ResourceContent.of(b"jpg", mime_type="image/jpeg"),
        ResourceContent.of(b"plain text", mime_type="text/plain"),
    )
    async with ctx.scope("test"):
        await guardrails.content_input_verification(content, config=_INPUT_CONFIG)

    converted = client.calls[0]["content"]
    assert converted[0] == {"text": {"text": "caption", "qualifiers": ["query"]}}
    assert converted[1] == {"image": {"format": "png", "source": {"bytes": png}}}
    assert converted[2] == {"image": {"format": "jpeg", "source": {"bytes": b"jpg"}}}
    assert converted[3] == {"text": {"text": "plain text", "qualifiers": ["query"]}}


@pytest.mark.asyncio
async def test_bedrock_output_intervention_message_without_reason() -> None:
    guardrails, _ = _guardrails({"action": "GUARDRAIL_INTERVENED"})
    async with ctx.scope("test"):
        with pytest.raises(GuardrailsModerationException) as raised:
            await guardrails.content_output_verification("hi", config=_OUTPUT_CONFIG)

    assert "None" not in str(raised.value)


@pytest.mark.asyncio
async def test_bedrock_intervention_reports_grounding_and_skips_undetected_rules() -> None:
    guardrails, _ = _guardrails(
        {
            "action": "GUARDRAIL_INTERVENED",
            "assessments": [
                {
                    "contextualGroundingPolicy": {
                        "filters": [
                            {"type": "GROUNDING", "action": "BLOCKED", "detected": True},
                            {"type": "RELEVANCE", "action": "NONE", "detected": False},
                        ]
                    },
                    "topicPolicy": {"topics": [{"name": "t", "type": "DENY", "detected": False}]},
                }
            ],
        }
    )
    async with ctx.scope("test"):
        with pytest.raises(GuardrailsModerationException) as raised:
            await guardrails.content_output_verification("hi", config=_OUTPUT_CONFIG)

    assert dict(raised.value.violations) == {"grounding:GROUNDING": 1.0}


@pytest.mark.asyncio
async def test_bedrock_pii_violation_does_not_leak_matched_value() -> None:
    guardrails, _ = _guardrails(
        {
            "action": "GUARDRAIL_INTERVENED",
            "assessments": [
                {
                    "sensitiveInformationPolicy": {
                        "piiEntities": [{"type": "EMAIL", "action": "BLOCKED", "match": "a@b.c"}]
                    }
                }
            ],
        }
    )
    async with ctx.scope("test"):
        with pytest.raises(GuardrailsModerationException) as raised:
            await guardrails.content_input_verification("hi", config=_INPUT_CONFIG)

    assert dict(raised.value.violations) == {"pii:EMAIL": 1.0}
    assert "a@b.c" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("mime_type", ("audio/mpeg", "video/mp4", "application/octet-stream"))
async def test_bedrock_guardrails_reject_unsupported_media_instead_of_fake_text(
    mime_type: str,
) -> None:
    guardrails, client = _guardrails({"action": "NONE"})
    async with ctx.scope("test"):
        with pytest.raises(GuardrailsFailure):
            await guardrails.content_input_verification(
                MultimodalContent.of("hi", ResourceContent.of(b"bytes", mime_type=mime_type)),
                config=_INPUT_CONFIG,
            )

    assert client.calls == []


@pytest.mark.asyncio
async def test_bedrock_guardrails_skip_empty_content_without_calling_service() -> None:
    guardrails, client = _guardrails({"action": "NONE"})
    async with ctx.scope("test"):
        await guardrails.content_input_verification("", config=_INPUT_CONFIG)
        await guardrails.content_output_verification(
            MultimodalContent.of("", "  "), config=_OUTPUT_CONFIG
        )

    assert client.calls == []
