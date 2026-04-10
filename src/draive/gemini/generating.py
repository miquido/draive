import random
from base64 import b64decode, b64encode, urlsafe_b64decode
from collections.abc import (
    AsyncIterable,
    Generator,
)
from typing import Any, cast
from uuid import uuid4

from google.api_core.exceptions import ResourceExhausted  # pyright: ignore[reportMissingImport]
from google.genai.errors import ClientError
from google.genai.types import (
    Candidate,
    FinishReason,
    FunctionCallingConfigMode,
    FunctionDeclarationDict,
    GenerateContentConfigDict,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    HarmBlockThreshold,
    HarmCategory,
    MediaResolution,
    Modality,
    Part,
    PartDict,
    SchemaDict,
)
from haiway import MISSING, Meta, as_dict, as_list, ctx

from draive.gemini.api import GeminiAPI
from draive.gemini.config import (
    GeminiConfig,
)
from draive.gemini.utils import speech_config, unwrap_missing
from draive.models import (
    ModelContext,
    ModelException,
    ModelInput,
    ModelInputBlocks,
    ModelInstructions,
    ModelOutput,
    ModelOutputBlocks,
    ModelOutputChunk,
    ModelOutputFailed,
    ModelOutputLimit,
    ModelOutputSelection,
    ModelOutputStream,
    ModelRateLimit,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
    record_model_invocation,
    record_usage_metrics,
)
from draive.multimodal import ArtifactContent, MultimodalContent, TextContent
from draive.resources import ResourceContent, ResourceReference

__all__ = ("GeminiGenerating",)

RATE_LIMIT_STATUS_CODE = 429


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
                thinking_budget=config.thinking_budget,
            )

            usage_meta = GenerateContentResponseUsageMetadata()
            usage_meta_recorded = False
            try:
                response_stream: AsyncIterable[
                    GenerateContentResponse
                ] = await self._client.aio.models.generate_content_stream(  # pyright: ignore[reportUnknownMemberType]
                    model=config.model,
                    config=_request_config(
                        instructions=instructions,
                        tools=tools,
                        output=output,
                        config=config,
                    ),
                    contents=list(_request_content(context)),
                )

                async for chunk in response_stream:
                    if chunk.usage_metadata is not None:
                        usage_meta = chunk.usage_metadata
                        usage_meta_recorded = True

                    if not chunk.candidates:
                        continue

                    chunk_candidate: Candidate = chunk.candidates[0]  # we always request only one

                    if (
                        chunk_candidate.content is not None
                        and chunk_candidate.content.parts is not None
                    ):
                        for part in chunk_candidate.content.parts:
                            for element in _part_as_stream_elements(part):
                                yield element

                    if chunk_candidate.finish_reason is None:
                        continue  # continue stream

                    elif chunk_candidate.finish_reason == FinishReason.STOP:
                        continue  # not expecting more parts but finish regularily

                    elif chunk_candidate.finish_reason == FinishReason.SAFETY:
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
                            reason=f"Safety filtering: {chunk_candidate.finish_message or ''}",
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
                                f"Completion error: {chunk_candidate.finish_message}"
                                if chunk_candidate.finish_message
                                else "Completion error"
                            ),
                        )

            except ResourceExhausted as exc:
                delay: float = random.uniform(0.3, 3.0)  # nosec: B311
                ctx.record_warning(
                    event="model.rate_limit",
                    attributes={
                        "model.provider": "gemini",
                        "model.name": config.model,
                        "retry_after": delay,
                    },
                )
                # Propagate as ModelRateLimit with randomized backoff window
                raise ModelRateLimit(
                    provider="gemini", model=config.model, retry_after=delay
                ) from exc

            except ClientError as exc:
                if exc.code == RATE_LIMIT_STATUS_CODE:
                    delay: float = random.uniform(0.3, 3.0)  # nosec: B311
                    ctx.record_warning(
                        event="model.rate_limit",
                        attributes={
                            "model.provider": "gemini",
                            "model.name": config.model,
                            "retry_after": delay,
                        },
                    )
                    raise ModelRateLimit(
                        provider="gemini",
                        model=config.model,
                        retry_after=delay,
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
                if usage_meta_recorded:
                    record_usage_metrics(
                        provider="gemini",
                        model=config.model,
                        input_tokens=usage_meta.prompt_token_count,
                        cached_input_tokens=usage_meta.cached_content_token_count,
                        output_tokens=usage_meta.candidates_token_count,
                    )


def _request_config(  # noqa: C901, PLR0912
    *,
    instructions: ModelInstructions,
    tools: ModelTools,
    output: ModelOutputSelection,
    config: GeminiConfig,
) -> GenerateContentConfigDict:
    configuration: GenerateContentConfigDict = {
        "system_instruction": instructions,
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

    if isinstance(output, type):
        configuration["response_modalities"] = [Modality.TEXT]
        configuration["response_mime_type"] = "application/json"
        configuration["response_json_schema"] = output.__SPECIFICATION__

    elif output == "auto":
        pass  # not specified - use defaults through missing

    elif output == "text":
        configuration["response_modalities"] = [Modality.TEXT]
        configuration["response_mime_type"] = "text/plain"

    elif output == "json":
        configuration["response_modalities"] = [Modality.TEXT]
        configuration["response_mime_type"] = "application/json"

    elif output == "image":
        configuration["response_modalities"] = [
            Modality.TEXT,
            Modality.IMAGE,
        ]  # google api does not allow to specify only image

    elif output == "audio":
        configuration["response_modalities"] = [Modality.AUDIO]

    elif "text" in output and "image" in output:
        configuration["response_modalities"] = [
            Modality.TEXT,
            Modality.IMAGE,
        ]

    else:
        raise NotImplementedError(f"{output} output is not supported by Gemini")

    if config.thinking_budget is not MISSING:
        configuration["thinking_config"] = {
            "include_thoughts": True,
            "thinking_budget": cast(int, config.thinking_budget),
        }

    if tools.specification:
        configuration["tools"] = [
            {
                "function_declarations": [
                    FunctionDeclarationDict(
                        name=tool.name,
                        description=tool.description,
                        parameters_json_schema=cast(SchemaDict, tool.parameters),
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
                "parts": list(_block_parts(element.input)),
            }

        else:
            assert isinstance(element, ModelOutput)  # nosec: B101
            yield {
                "role": "model",
                "parts": list(_block_parts(element.output)),
            }


def _part_as_stream_elements(
    part: Part,
) -> Generator[ModelOutputChunk]:
    if part.text:
        if part.thought:
            yield ModelReasoningChunk.of(
                TextContent.of(part.text),
                meta={
                    "kind": "thought",
                    "signature": b64encode(part.thought_signature).decode()
                    if part.thought_signature
                    else None,
                },
            )

        else:
            yield TextContent.of(part.text)

    if part.function_call and part.function_call.name:
        yield ModelToolRequest(
            identifier=str(part.function_call.id or uuid4()),
            tool=part.function_call.name,
            arguments=part.function_call.args if part.function_call.args is not None else {},
            meta=Meta.of(
                {
                    "signature": b64encode(part.thought_signature).decode(),
                }
            )
            if part.thought_signature
            else Meta.empty,
        )

    if part.inline_data and part.inline_data.data:  # there is no content without content...
        yield ResourceContent.of(
            part.inline_data.data,
            mime_type=part.inline_data.mime_type or "application/octet-stream",
        )

    if part.file_data and part.file_data.file_uri:  # there is no content without content...
        yield ResourceReference.of(
            part.file_data.file_uri,
            mime_type=part.file_data.mime_type,
        )


def _block_parts(  # noqa: PLR0912
    blocks: ModelInputBlocks | ModelOutputBlocks,
    /,
) -> Generator[PartDict]:
    for block in blocks:
        if isinstance(block, ModelToolRequest):
            if signature := block.meta.get_str("signature"):
                yield {
                    "function_call": {
                        "id": block.identifier,
                        "name": block.tool,
                        "args": as_dict(block.arguments),
                    },
                    "thought_signature": b64decode(signature),
                }

            else:
                yield {
                    "function_call": {
                        "id": block.identifier,
                        "name": block.tool,
                        "args": as_dict(block.arguments),
                    }
                }

        elif isinstance(block, ModelToolResponse):
            if block.status == "error":
                yield {
                    "function_response": {
                        "id": block.identifier,
                        "name": block.tool,
                        "response": {
                            "error": list(_content_parts(block.content)),
                        },
                    }
                }

            else:
                yield {
                    "function_response": {
                        "id": block.identifier,
                        "name": block.tool,
                        "response": {
                            "output": list(_content_parts(block.content)),
                        },
                    }
                }

        elif isinstance(block, ModelReasoning):
            if block.meta.kind == "thought":
                if signature := block.meta.get_str("signature"):
                    yield {
                        "text": block.reasoning.to_str(),
                        "thought": True,
                        "thought_signature": b64decode(signature),
                    }

                else:
                    yield {
                        "text": block.reasoning.to_str(),
                        "thought": True,
                    }

            else:
                raise ValueError(f"Unsupported reasoning element: {block.meta.kind}")

        else:
            yield from _content_parts(block)


def _content_parts(
    content: MultimodalContent,
    /,
) -> Generator[PartDict]:
    for part in content.parts:
        if isinstance(part, TextContent):
            yield {
                "text": part.text,
            }

        elif isinstance(part, ResourceContent):
            yield {
                "inline_data": {
                    # decode urlsafe base64 back to raw bytes for provider
                    "data": urlsafe_b64decode(part.data),
                    "mime_type": part.mime_type,
                },
            }

        elif isinstance(part, ResourceReference):
            yield {
                "file_data": {
                    "file_uri": part.uri,
                    "mime_type": part.mime_type,
                }
            }

        else:
            assert isinstance(part, ArtifactContent)  # nosec: B101
            # Skip artifacts that are marked as hidden
            if part.hidden:
                continue

            yield {
                "text": part.to_str(),
            }
