from collections.abc import (
    AsyncGenerator,
    Collection,
    Generator,
)
from typing import Any, Final, cast

from google.genai.errors import APIError
from google.genai.types import (
    Candidate,
    FinishReason,
    FunctionCallingConfigMode,
    FunctionDeclarationDict,
    GenerateContentConfigDict,
    GenerateContentResponse,
    GenerateContentResponsePromptFeedback,
    GenerateContentResponseUsageMetadata,
    HarmBlockThreshold,
    HarmCategory,
    MediaResolution,
    Modality,
)
from haiway import MISSING, as_list, ctx

from draive.gemini.api import GeminiAPI
from draive.gemini.config import (
    GeminiConfig,
)
from draive.gemini.content import block_parts, part_as_stream_elements
from draive.gemini.utils import (
    RATE_LIMIT_STATUS_CODE,
    combined_input_tokens,
    speech_config,
    thinking_config,
    unwrap_missing,
)
from draive.models import (
    ModelContext,
    ModelException,
    ModelInput,
    ModelInputInvalid,
    ModelInstructions,
    ModelOutput,
    ModelOutputFailed,
    ModelOutputLimit,
    ModelOutputSelection,
    ModelOutputStream,
    ModelTools,
    model_rate_limit,
    record_model_invocation,
    record_usage_metrics,
)

__all__ = ("GeminiGenerating",)

# finish reasons caused by content policies instead of a generation failure
_SAFETY_FINISH_REASONS: Final[frozenset[FinishReason]] = frozenset(
    (
        FinishReason.SAFETY,
        FinishReason.RECITATION,
        FinishReason.BLOCKLIST,
        FinishReason.PROHIBITED_CONTENT,
        FinishReason.SPII,
        FinishReason.IMAGE_SAFETY,
        FinishReason.IMAGE_PROHIBITED_CONTENT,
        FinishReason.IMAGE_RECITATION,
    )
)


class GeminiGenerating(GeminiAPI):
    async def completion(  # noqa: C901, PLR0912
        self,
        *,
        instructions: ModelInstructions,
        context: ModelContext,
        tools: ModelTools,
        output: ModelOutputSelection,
        config: GeminiConfig | None = None,
        **extra: Any,
    ) -> ModelOutputStream:
        async with ctx.scope("model.invocation"):
            config = config or ctx.state(GeminiConfig)
            record_model_invocation(
                provider="gemini",
                model=config.model,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                tools=tools,
                output=output,
                stop_sequences=config.stop_sequences,
                top_p=config.top_p,
                top_k=config.top_k,
                seed=config.seed,
                thinking_budget=config.thinking_budget,
                thinking_level=config.thinking_level,
            )

            # built before the request to let an unsupported configuration surface
            # as itself instead of being reported as a failed generation
            request_config: GenerateContentConfigDict = _request_config(
                instructions=instructions,
                tools=tools,
                output=output,
                config=config,
            )

            request_content: list[dict[str, Any]]
            try:
                # eagerly materialize to convert context errors to ModelInputInvalid here
                request_content = list(_request_content(context))

            except Exception as exc:
                raise ModelInputInvalid(
                    provider="gemini",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            usage_meta: GenerateContentResponseUsageMetadata | None = None
            # the client declares a plain iterator while always producing an async
            # generator, the narrower type allows releasing it explicitly
            response_stream: AsyncGenerator[GenerateContentResponse] | None = None
            try:
                response_stream = cast(
                    AsyncGenerator[GenerateContentResponse],
                    await self._client.aio.models.generate_content_stream(  # pyright: ignore[reportUnknownMemberType]
                        model=config.model,
                        config=request_config,
                        contents=request_content,
                    ),
                )

                async for chunk in response_stream:
                    if chunk.usage_metadata is not None:
                        usage_meta = chunk.usage_metadata

                    if not chunk.candidates:
                        if chunk.prompt_feedback is not None:
                            # prompt feedback is delivered only when the prompt was blocked
                            raise _prompt_blocked_failure(
                                chunk.prompt_feedback,
                                model=config.model,
                            )

                        continue

                    chunk_candidate: Candidate = chunk.candidates[0]  # we always request only one

                    if (
                        chunk_candidate.content is not None
                        and chunk_candidate.content.parts is not None
                    ):
                        # gemini ends a turn with a part carrying only the turn signature,
                        # detached from the reasoning it belongs to - it converts to no
                        # elements, dropping that signature. Only function call signatures
                        # are validated on replay and those travel on the call part itself.
                        for part in chunk_candidate.content.parts:
                            for element in part_as_stream_elements(part):
                                yield element

                    if chunk_candidate.finish_reason is None:
                        continue  # continue stream

                    elif chunk_candidate.finish_reason == FinishReason.STOP:
                        continue  # not expecting more parts but finish regularily

                    elif chunk_candidate.finish_reason in _SAFETY_FINISH_REASONS:
                        if chunk_candidate.safety_ratings is not None:
                            ctx.record_info(
                                event="model.safety.results",
                                attributes={
                                    "results": [
                                        f"{rating.category} |blocked: {rating.blocked}"
                                        f" |probability:{rating.probability_score}"
                                        f" |severity:{rating.severity_score}"
                                        for rating in chunk_candidate.safety_ratings
                                        if rating.category
                                    ],
                                },
                            )

                        raise ModelOutputFailed(
                            provider="gemini",
                            model=config.model,
                            reason=(
                                f"Safety filtering ({chunk_candidate.finish_reason.value}):"
                                f" {chunk_candidate.finish_message or ''}"
                            ),
                        )

                    elif chunk_candidate.finish_reason == FinishReason.MAX_TOKENS:
                        raise ModelOutputLimit(
                            provider="gemini",
                            model=config.model,
                            max_output_tokens=unwrap_missing(
                                config.max_output_tokens,
                                default=0,
                            ),
                        )

                    else:
                        raise ModelOutputFailed(
                            provider="gemini",
                            model=config.model,
                            reason=(
                                f"Completion error ({chunk_candidate.finish_reason.value}):"
                                f" {chunk_candidate.finish_message or ''}"
                            ),
                        )

            except APIError as exc:
                if exc.code == RATE_LIMIT_STATUS_CODE:
                    raise model_rate_limit(
                        provider="gemini",
                        model=config.model,
                        retry_after=None,
                    ) from exc

                raise ModelOutputFailed(
                    provider="gemini",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            except ModelException as exc:
                raise exc

            except Exception as exc:
                # Convert to ModelOutputFailed for consistency
                raise ModelOutputFailed(
                    provider="gemini",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            finally:
                if usage_meta is not None:
                    record_usage_metrics(
                        provider="gemini",
                        model=config.model,
                        input_tokens=combined_input_tokens(
                            usage_meta.prompt_token_count,
                            usage_meta.tool_use_prompt_token_count,
                        ),
                        cached_input_tokens=usage_meta.cached_content_token_count,
                        output_tokens=usage_meta.candidates_token_count,
                        # thinking tokens are not included within candidates count
                        reasoning_output_tokens=usage_meta.thoughts_token_count,
                    )

                if response_stream is not None:
                    # release the http stream, iteration may have ended
                    # before the response was completed. Closing it unwinds only the
                    # outermost sdk generator - the layers below it iterate their source
                    # without closing it, so the http response of a stream ended early is
                    # released by their finalization instead, at the next collection.
                    await response_stream.aclose()


def _prompt_blocked_failure(
    prompt_feedback: GenerateContentResponsePromptFeedback,
    /,
    *,
    model: str,
) -> ModelOutputFailed:
    if prompt_feedback.safety_ratings is not None:
        ctx.record_info(
            event="model.safety.results",
            attributes={
                "results": [
                    f"{rating.category} |blocked: {rating.blocked}"
                    f" |probability:{rating.probability_score}"
                    f" |severity:{rating.severity_score}"
                    for rating in prompt_feedback.safety_ratings
                    if rating.category
                ],
            },
        )

    return ModelOutputFailed(
        provider="gemini",
        model=model,
        reason=(
            "Prompt blocked"
            f" ({prompt_feedback.block_reason.value if prompt_feedback.block_reason else 'OTHER'}):"
            f" {prompt_feedback.block_reason_message or ''}"
        ),
    )


def _output_config(
    output: ModelOutputSelection,
    /,
) -> GenerateContentConfigDict:
    """Resolve response modality settings for the requested output.

    Returns
    -------
    GenerateContentConfigDict
        Settings to merge into the request configuration, empty when the defaults apply.

    Raises
    ------
    NotImplementedError
        When the requested modalities are not available - silently answering with
        text would not match what was asked for.
    """
    if isinstance(output, type):
        return {
            "response_modalities": [Modality.TEXT],
            "response_mime_type": "application/json",
            "response_json_schema": output.__SPECIFICATION__,
        }

    if output == "auto":
        return {}  # not specified - use defaults through missing

    if output == "text":
        return {
            "response_modalities": [Modality.TEXT],
            "response_mime_type": "text/plain",
        }

    if output == "json":
        return {
            "response_modalities": [Modality.TEXT],
            "response_mime_type": "application/json",
        }

    if isinstance(output, str):  # a single modality literal
        return _modalities_config((output,))

    return _modalities_config(output)


def _modalities_config(
    output: Collection[str],
    /,
) -> GenerateContentConfigDict:
    # every requested modality has to be available
    requested: set[str] = set(output)
    if requested - {"text", "image", "audio"}:
        raise NotImplementedError(f"{output} output is not supported by Gemini")

    if "audio" in requested:
        # audio cannot be combined with the other modalities
        if requested != {"audio"}:
            raise NotImplementedError(f"{output} output is not supported by Gemini")

        return {"response_modalities": [Modality.AUDIO]}

    if "image" in requested:
        # the api does not allow requesting image without text
        return {"response_modalities": [Modality.TEXT, Modality.IMAGE]}

    return {
        "response_modalities": [Modality.TEXT],
        "response_mime_type": "text/plain",
    }


def _request_config(  # noqa: C901, PLR0912
    *,
    instructions: ModelInstructions,
    tools: ModelTools,
    output: ModelOutputSelection,
    config: GeminiConfig,
) -> GenerateContentConfigDict:
    configuration: GenerateContentConfigDict = {
        "temperature": unwrap_missing(config.temperature),
        "top_p": unwrap_missing(config.top_p),
        "top_k": unwrap_missing(config.top_k),
        "seed": unwrap_missing(config.seed),
        "max_output_tokens": unwrap_missing(config.max_output_tokens),
        "stop_sequences": unwrap_missing(
            config.stop_sequences,
            transform=as_list,
        ),
        "candidate_count": 1,
    }

    if instructions:
        configuration["system_instruction"] = instructions

    output_config: GenerateContentConfigDict = _output_config(output)
    if "response_modalities" in output_config:
        configuration["response_modalities"] = output_config["response_modalities"]

    if "response_mime_type" in output_config:
        configuration["response_mime_type"] = output_config["response_mime_type"]

    if "response_json_schema" in output_config:
        configuration["response_json_schema"] = output_config["response_json_schema"]

    if thinking := thinking_config(config):
        configuration["thinking_config"] = thinking

    if tools.specification:
        configuration["tools"] = [
            {
                "function_declarations": [
                    FunctionDeclarationDict(
                        name=tool.name,
                        description=tool.description,
                        parameters_json_schema=tool.parameters,
                    )
                    for tool in tools.specification
                ]
            }
        ]

        if tools.selection == "auto":
            configuration["tool_config"] = {
                "function_calling_config": {
                    "mode": FunctionCallingConfigMode.AUTO,
                }
            }

        elif tools.selection == "none":
            configuration["tool_config"] = {
                "function_calling_config": {
                    "mode": FunctionCallingConfigMode.NONE,
                }
            }

        elif tools.selection == "required":
            configuration["tool_config"] = {
                "function_calling_config": {
                    "mode": FunctionCallingConfigMode.ANY,
                }
            }

        else:  # handle specific tool selection
            configuration["tool_config"] = {
                "function_calling_config": {
                    "mode": FunctionCallingConfigMode.ANY,
                    "allowed_function_names": [tools.selection.name],
                },
            }

    else:  # no functions
        configuration["tool_config"] = {
            "function_calling_config": {
                "mode": FunctionCallingConfigMode.NONE,
            }
        }

    if config.safety is not MISSING:
        configuration["safety_settings"] = [
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": HarmBlockThreshold(config.safety.harm_category_hate_speech_threshold),
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold(
                    config.safety.harm_category_dangerous_content_threshold
                ),
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": HarmBlockThreshold(config.safety.harm_category_harassment_threshold),
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": HarmBlockThreshold(
                    config.safety.harm_category_sexually_explicit_threshold
                ),
            },
            {
                "category": HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                "threshold": HarmBlockThreshold(
                    config.safety.harm_category_civic_integrity_threshold
                ),
            },
            # image and jailbreak categories have no dedicated configuration,
            # they reuse the thresholds of their matching content categories
            {
                "category": HarmCategory.HARM_CATEGORY_JAILBREAK,
                "threshold": HarmBlockThreshold(
                    config.safety.harm_category_dangerous_content_threshold
                ),
            },
        ]

    if config.media_resolution is MISSING:
        pass  # not specified - use defaults through missing

    elif config.media_resolution == "low":
        configuration["media_resolution"] = MediaResolution.MEDIA_RESOLUTION_LOW

    elif config.media_resolution == "medium":
        configuration["media_resolution"] = MediaResolution.MEDIA_RESOLUTION_MEDIUM

    elif config.media_resolution == "high":
        configuration["media_resolution"] = MediaResolution.MEDIA_RESOLUTION_HIGH

    if speech := speech_config(config):
        configuration["speech_config"] = speech

    return configuration


def _request_content(
    context: ModelContext,
) -> Generator[dict[str, Any]]:
    for element in context:
        if isinstance(element, ModelInput):
            yield {
                "role": "user",
                "parts": list(block_parts(element.input)),
            }

        else:
            assert isinstance(element, ModelOutput)  # nosec: B101
            yield {
                "role": "model",
                "parts": list(block_parts(element.output)),
            }
