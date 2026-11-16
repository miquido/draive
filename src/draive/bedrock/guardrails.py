from base64 import b64decode
from collections.abc import Mapping
from typing import Any, Literal

from haiway import asynchronous, ctx

from draive.bedrock.api import BedrockAPI
from draive.bedrock.config import BedrockInputGuardraisConfig, BedrockOutputGuardraisConfig
from draive.guardrails import GuardrailsFailure, GuardrailsModerationException
from draive.models import record_guardrails_invocation
from draive.multimodal import Multimodal, MultimodalContent, TextContent
from draive.resources import ResourceContent, ResourceReference

__all__ = ("BedrockGuardrails",)


# (policy, entries, identifying field, violation prefix) - matched values of sensitive
# information are intentionally not reported, only the kind of information
_ASSESSED_RULES: tuple[tuple[str, str, str, str], ...] = (
    ("contentPolicy", "filters", "type", ""),
    ("topicPolicy", "topics", "name", "topic:"),
    ("sensitiveInformationPolicy", "piiEntities", "type", "pii:"),
    ("sensitiveInformationPolicy", "regexes", "name", "regex:"),
    ("wordPolicy", "customWords", "match", "word:"),
    ("wordPolicy", "managedWordLists", "match", "word:"),
    ("contextualGroundingPolicy", "filters", "type", "grounding:"),
)


def _triggered(entry: Mapping[str, Any], /) -> bool:
    detected: Any = entry.get("detected")
    if detected is not None:
        return bool(detected)

    return entry.get("action") in ("BLOCKED", "ANONYMIZED")


def _violations(
    response: Mapping[str, Any],
    /,
    *,
    default: str,
) -> dict[str, float]:
    # bedrock do not provide scores in response, every triggered rule counts as 1.0
    violations: dict[str, float] = {}
    assessment: Mapping[str, Any]
    for assessment in response.get("assessments", ()):
        for policy, entries, field, prefix in _ASSESSED_RULES:
            entry: Mapping[str, Any]
            for entry in assessment.get(policy, {}).get(entries, ()):
                if field in entry and _triggered(entry):
                    violations[f"{prefix}{entry[field]}"] = 1.0

    return violations or {default: 1.0}


class BedrockGuardrails(BedrockAPI):
    async def content_input_verification(
        self,
        content: Multimodal,
        /,
        *,
        config: BedrockInputGuardraisConfig | None = None,
        **extra: Any,
    ) -> None:
        guardrails_config: BedrockInputGuardraisConfig = config or ctx.state(
            BedrockInputGuardraisConfig
        )
        async with ctx.scope("guardrails.invocation"):
            record_guardrails_invocation(
                provider="bedrock",
                identifier=guardrails_config.guardrail_identifier,
                version=guardrails_config.guardrail_version,
                source="input",
            )

            content = MultimodalContent.of(content)
            await self._content_input_verification(
                content,
                config=guardrails_config,
            )

    @asynchronous
    def _content_input_verification(
        self,
        content: MultimodalContent,
        /,
        *,
        config: BedrockInputGuardraisConfig,
        **extra: Any,
    ) -> None:
        moderated_content: list[dict[str, Any]] = self._convert_content(
            content,
            qualifier="query",
        )
        if not moderated_content:
            return  # nothing to verify, the service rejects empty content

        response: dict[str, Any] = self._client.apply_guardrail(
            guardrailIdentifier=config.guardrail_identifier,
            guardrailVersion=config.guardrail_version,
            source="INPUT",
            content=moderated_content,
        )

        if response.get("action") == "GUARDRAIL_INTERVENED":
            raise GuardrailsModerationException(
                f"Content violated guardrails ({config.guardrail_identifier}) rules",
                violations=_violations(response, default=config.guardrail_identifier),
                content=content,
            )

    async def content_output_verification(
        self,
        content: Multimodal,
        /,
        *,
        config: BedrockOutputGuardraisConfig | None = None,
        **extra: Any,
    ) -> None:
        guardrails_config: BedrockOutputGuardraisConfig = config or ctx.state(
            BedrockOutputGuardraisConfig
        )
        async with ctx.scope("guardrails.invocation"):
            record_guardrails_invocation(
                provider="bedrock",
                identifier=guardrails_config.guardrail_identifier,
                version=guardrails_config.guardrail_version,
                source="output",
            )

            content = MultimodalContent.of(content)
            await self._content_output_verification(
                content,
                config=guardrails_config,
            )

    @asynchronous
    def _content_output_verification(
        self,
        content: MultimodalContent,
        /,
        *,
        config: BedrockOutputGuardraisConfig,
        **extra: Any,
    ) -> None:
        moderated_content: list[dict[str, Any]] = self._convert_content(
            content,
            qualifier="guard_content",
        )
        if not moderated_content:
            return  # nothing to verify, the service rejects empty content

        response: dict[str, Any] = self._client.apply_guardrail(
            guardrailIdentifier=config.guardrail_identifier,
            guardrailVersion=config.guardrail_version,
            source="OUTPUT",
            content=moderated_content,
            outputScope="INTERVENTIONS",
        )

        if response.get("action") == "GUARDRAIL_INTERVENED":
            reason: str | None = response.get("actionReason")
            raise GuardrailsModerationException(
                f"Content violated guardrails ({config.guardrail_identifier}) rules"
                if reason is None
                else f"Violated guardrails rules: {reason}",
                violations=_violations(response, default=config.guardrail_identifier),
                content=content,
            )

    def _convert_content(
        self,
        content: MultimodalContent,
        /,
        *,
        qualifier: Literal["guard_content", "query"],
    ) -> list[dict[str, Any]]:
        moderated_content: list[dict[str, Any]] = []
        for part in content.parts:
            match part:
                case TextContent() as text:
                    if not text.text.strip():
                        continue  # blank text is rejected by the service

                    moderated_content.append(
                        {
                            "text": {
                                "text": text.text,
                                "qualifiers": [qualifier],
                            },
                        }
                    )

                case ResourceContent() as content_part:
                    converted: dict[str, Any] | None = _convert_resource(
                        content_part,
                        qualifier=qualifier,
                    )
                    if converted is not None:
                        moderated_content.append(converted)

                case ResourceReference() as ref:
                    raise GuardrailsFailure(
                        "Bedrock guardrails require content bytes, got a resource reference",
                        cause=ValueError(f"Unsupported resource reference: {ref.mime_type}"),
                    )

                case other:
                    ctx.log_warning(
                        f"Attempting to use guardrails on unsupported content ({type(other)}),"
                        " verifying as text..."
                    )
                    other_text: str = other.to_str()
                    if other_text.strip():
                        moderated_content.append(
                            {
                                "text": {
                                    "text": other_text,
                                    "qualifiers": [qualifier],
                                }
                            }
                        )

        return moderated_content


def _convert_resource(
    resource: ResourceContent,
    /,
    *,
    qualifier: str,
) -> dict[str, Any] | None:
    # only png and jpeg images and text are supported by Bedrock guardrails
    if resource.mime_type in ("image/jpeg", "image/png"):
        return {
            "image": {
                "format": resource.mime_type.removeprefix("image/"),
                "source": {"bytes": b64decode(resource.data)},
            }
        }

    if resource.mime_type.startswith("text/"):
        text: str = resource.to_bytes().decode("utf-8", errors="replace")
        if not text.strip():
            return None

        return {"text": {"text": text, "qualifiers": [qualifier]}}

    # verifying a placeholder as text would report unverified content as safe
    raise GuardrailsFailure(
        f"Bedrock guardrails support only png and jpeg images and text, got {resource.mime_type}",
        cause=ValueError(f"Unsupported media type: {resource.mime_type}"),
    )
