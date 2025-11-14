from base64 import urlsafe_b64decode
from typing import Any, Literal

from haiway import asynchronous, ctx

from draive.bedrock.api import BedrockAPI
from draive.bedrock.config import BedrockInputGuardraisConfig, BedrockOutputGuardraisConfig
from draive.guardrails import GuardrailsModeration, GuardrailsModerationException
from draive.multimodal import Multimodal, MultimodalContent, TextContent
from draive.resources import ResourceContent, ResourceReference

__all__ = ("BedrockGuardrails",)


class BedrockGuardrails(BedrockAPI):
    def guardrails_moderation(self) -> GuardrailsModeration:
        return GuardrailsModeration(
            input_checking=self.content_input_verification,
            output_checking=self.content_output_verification,
        )

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
        response: dict[str, Any] = self._client.apply_guardrail(
            guardrailIdentifier=config.guardrail_identifier,
            guardrailVersion=config.guardrail_version,
            source="INPUT",
            content=self._convert_content(
                content,
                qualifier="query",
            ),
        )

        if response.get("action") == "GUARDRAIL_INTERVENED":
            raise GuardrailsModerationException(
                f"Content violated guardrails ({config.guardrail_identifier}) rules",
                violations={
                    filter["type"]: 1.0  # bedrock do not provide scores in response
                    for assessment in response.get("assessments", ())
                    for filter in assessment.get("contentPolicy", {}).get("filters", ())  # noqa: A001
                    if "type" in filter and filter.get("detected", False)
                },
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
        async with ctx.scope("bedrock_guardrails"):
            ctx.record_info(
                attributes={
                    "guardrails.provider": "bedrock",
                    "guardrails.identifier": guardrails_config.guardrail_identifier,
                    "guardrails.version": guardrails_config.guardrail_version,
                },
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
        response: dict[str, Any] = self._client.apply_guardrail(
            guardrailIdentifier=config.guardrail_identifier,
            guardrailVersion=config.guardrail_version,
            source="OUTPUT",
            content=self._convert_content(
                content,
                qualifier="guard_content",
            ),
            outputScope="INTERVENTIONS",
        )

        if response.get("action") == "GUARDRAIL_INTERVENED":
            raise GuardrailsModerationException(
                f"Violated guardrails rules: {response.get('actionReason')}",
                # bedrock do not provide scores in response
                violations={config.guardrail_identifier: 1.0},
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
                    moderated_content.append(
                        {
                            "text": {
                                "text": text.text,
                                "qualifiers": [qualifier],
                            },
                        }
                    )

                case ResourceContent() as content_part:
                    # Only selected image resources are supported by Bedrock guardrails
                    if content_part.mime_type == "image/jpeg":
                        moderated_content.append(
                            {
                                "image": {
                                    "format": "jpeg",
                                    "source": {"bytes": urlsafe_b64decode(content_part.data)},
                                }
                            }
                        )

                    elif content_part.mime_type == "image/png":
                        moderated_content.append(
                            {
                                "image": {
                                    "format": "png",
                                    "source": {"bytes": urlsafe_b64decode(content_part.data)},
                                }
                            }
                        )

                    elif content_part.mime_type == "image/gif":
                        moderated_content.append(
                            {
                                "image": {
                                    "format": "gif",
                                    "source": {"bytes": urlsafe_b64decode(content_part.data)},
                                }
                            }
                        )

                    else:
                        ctx.log_warning(
                            "Attempting to use guardrails on unsupported media "
                            f"({content_part.mime_type}), verifying as text..."
                        )
                        moderated_content.append(
                            {
                                "text": {
                                    "text": content_part.to_str(),
                                    "qualifiers": [qualifier],
                                }
                            }
                        )

                case ResourceReference() as ref:
                    ctx.log_warning(
                        "Bedrock guardrails require image bytes; got ResourceReference. "
                        "Verifying reference as JSON text."
                    )
                    moderated_content.append(
                        {
                            "text": {
                                "text": ref.to_str(),
                                "qualifiers": [qualifier],
                            }
                        }
                    )

                case other:
                    ctx.log_warning(
                        f"Attempting to use guardrails on unsupported content ({type(other)}),"
                        " verifying as text..."
                    )
                    moderated_content.append(
                        {
                            "text": {
                                "text": other.to_str(),
                                "qualifiers": [qualifier],
                            }
                        }
                    )

        return moderated_content
