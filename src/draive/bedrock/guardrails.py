from typing import Any, Literal

from haiway import ObservabilityLevel, asynchronous, ctx

from draive.bedrock.api import BedrockAPI
from draive.bedrock.config import BedrockInputGuardraisConfig, BedrockOutputGuardraisConfig
from draive.guardrails import GuardrailsModeration, GuardrailsModerationException
from draive.multimodal import MediaData, Multimodal, MultimodalContent, TextContent

__all__ = ("BedrockGuardrais",)


class BedrockGuardrais(BedrockAPI):
    def content_guardrails(self) -> GuardrailsModeration:
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
        ).updated(**extra)

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
                violations=tuple(
                    filter["type"]
                    for assessment in response.get("assessments", ())
                    for filter in assessment.get("contentPolicy", {}).get("filters", ())  # noqa: A001
                    if "type" in filter and filter.get("detected", False)
                ),
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
        with ctx.scope("bedrock_guardrails", guardrails_config):
            ctx.record(
                ObservabilityLevel.INFO,
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
                violations=(config.guardrail_identifier,),
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

                case MediaData() as media_data:
                    # Check media type properly using startswith for MIME types
                    if media_data.media.startswith("image/jpeg"):
                        moderated_content.append(
                            {
                                "image": {
                                    "format": "jpeg",
                                    "source": {"bytes": media_data.data},
                                }
                            }
                        )

                    elif media_data.media.startswith("image/png"):
                        moderated_content.append(
                            {
                                "image": {
                                    "format": "png",
                                    "source": {"bytes": media_data.data},
                                }
                            }
                        )

                    else:
                        ctx.log_warning(
                            f"Attempting to use guardrails on unsupported media"
                            f" ({media_data.media}), verifying as json text..."
                        )
                        moderated_content.append(
                            {
                                "text": {
                                    "text": media_data.to_json(),
                                    "qualifiers": [qualifier],
                                }
                            }
                        )

                case other:
                    ctx.log_warning(
                        f"Attempting to use guardrails on unsupported content ({type(other)}),"
                        " verifying as json text..."
                    )
                    moderated_content.append(
                        {
                            "text": {
                                "text": other.to_json(),
                                "qualifiers": [qualifier],
                            }
                        }
                    )

        return moderated_content
