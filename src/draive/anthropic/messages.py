import random
from collections.abc import AsyncGenerator, Coroutine, Generator, Iterable, Mapping, Sequence
from typing import Any, Literal, cast, overload

from anthropic import Omit, omit
from anthropic import RateLimitError as AnthropicRateLimitError
from anthropic.types import (
    CitationCharLocation,
    CitationContentBlockLocation,
    CitationPageLocation,
    CitationsSearchResultLocation,
    ContentBlock,
    ImageBlockParam,
    Message,
    MessageParam,
    RedactedThinkingBlock,
    TextBlock,
    TextBlockParam,
    ThinkingBlock,
    ThinkingConfigParam,
    ToolChoiceParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
)
from anthropic.types.redacted_thinking_block_param import RedactedThinkingBlockParam
from anthropic.types.thinking_block_param import ThinkingBlockParam
from haiway import (
    META_EMPTY,
    MISSING,
    Meta,
    Missing,
    as_dict,
    as_list,
    ctx,
    unwrap_missing,
)

from draive.anthropic.api import AnthropicAPI
from draive.anthropic.config import AnthropicConfig
from draive.models import (
    GenerativeModel,
    ModelContext,
    ModelInput,
    ModelInputInvalid,
    ModelInstructions,
    ModelOutput,
    ModelOutputFailed,
    ModelOutputLimit,
    ModelOutputSelection,
    ModelRateLimit,
    ModelReasoning,
    ModelStreamOutput,
    ModelToolRequest,
    ModelToolsDeclaration,
    ModelToolSpecification,
    ModelToolsSelection,
)
from draive.multimodal import ArtifactContent, Multimodal, MultimodalContent, TextContent
from draive.resources import ResourceContent, ResourceReference

__all__ = ("AnthropicMessages",)


class AnthropicMessages(AnthropicAPI):
    def generative_model(self) -> GenerativeModel:
        return GenerativeModel(generating=self.completion)

    @overload
    def completion(
        self,
        *,
        instructions: ModelInstructions,
        context: ModelContext,
        tools: ModelToolsDeclaration,
        output: ModelOutputSelection,
        stream: Literal[False] = False,
        config: AnthropicConfig | None = None,
        prefill: Multimodal | None = None,
        **extra: Any,
    ) -> Coroutine[None, None, ModelOutput]: ...

    @overload
    def completion(
        self,
        *,
        instructions: ModelInstructions,
        context: ModelContext,
        tools: ModelToolsDeclaration,
        output: ModelOutputSelection,
        stream: Literal[True],
        config: AnthropicConfig | None = None,
        prefill: Multimodal | None = None,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput]: ...

    def completion(
        self,
        *,
        instructions: ModelInstructions,
        context: ModelContext,
        tools: ModelToolsDeclaration,
        output: ModelOutputSelection,
        stream: bool = False,
        config: AnthropicConfig | None = None,
        prefill: Multimodal | None = None,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput] | Coroutine[None, None, ModelOutput]:
        if stream:
            return self._completion_stream(
                instructions=instructions,
                context=context,
                tools=tools,
                output=output,
                config=config or ctx.state(AnthropicConfig),
                prefill=prefill,
                **extra,
            )

        else:
            return self._completion(
                instructions=instructions,
                context=context,
                tools=tools,
                output=output,
                config=config or ctx.state(AnthropicConfig),
                prefill=prefill,
                **extra,
            )

    async def _completion(  # noqa: C901, PLR0912
        self,
        *,
        instructions: ModelInstructions,
        context: ModelContext,
        tools: ModelToolsDeclaration,
        output: ModelOutputSelection,
        config: AnthropicConfig,
        prefill: Multimodal | None = None,
        **extra: Any,
    ) -> ModelOutput:
        async with ctx.scope("model.completion"):
            ctx.record_info(
                attributes={
                    "model.provider": self._provider,
                    "model.name": config.model,
                    "model.temperature": config.temperature,
                    "model.thinking_budget": config.thinking_budget,
                    "model.stop_sequences": config.stop_sequences,
                    "model.output": str(output),
                    "model.max_output_tokens": config.max_output_tokens,
                    "model.tools.count": len(tools.specifications),
                    "model.tools.selection": tools.selection,
                    "model.stream": False,
                },
            )
            ctx.record_debug(
                attributes={
                    "model.instructions": instructions,
                    "model.tools": [tool.name for tool in tools.specifications],
                    "model.context": [element.to_str() for element in context],
                },
            )

            if output not in ("auto", "text", "json"):
                raise NotImplementedError(f"{output} output is not supported by Anthropic")

            messages: Iterable[MessageParam]
            try:  # Build base messages from context
                if prefill is not None:
                    messages = (
                        *_context_messages(context),
                        {
                            "role": "assistant",
                            "content": _content_elements(
                                MultimodalContent.of(prefill),
                                cache_type=None,
                            ),
                        },
                    )

                elif output == "json" or isinstance(output, type):
                    messages = (
                        *_context_messages(context),
                        {
                            "role": "assistant",
                            # Bias the model toward JSON by pre-filling the opening brace
                            "content": [{"type": "text", "text": "{"}],
                        },
                    )

                else:
                    messages = _context_messages(context)

            except Exception as exc:
                raise ModelInputInvalid(
                    provider=self._provider,
                    model=config.model,
                ) from exc

            tools_list: Iterable[ToolParam] | Omit
            tool_choice: ToolChoiceParam | Omit
            if tools.specifications:
                tool_choice = _tools_selection_as_tool_choice(tools.selection)
                tools_list = _tools_as_tool_params(tools.specifications)

            else:
                tool_choice = omit
                tools_list = omit

            completion: Message
            try:
                completion = await self._client.messages.create(
                    model=config.model,
                    system=instructions if instructions else omit,
                    messages=messages,
                    temperature=config.temperature,
                    max_tokens=config.max_output_tokens,
                    thinking=_thinking_budget_config(config.thinking_budget),
                    tools=tools_list,
                    tool_choice=tool_choice,
                    stop_sequences=as_list(cast(Sequence[str], config.stop_sequences))
                    if config.stop_sequences is not MISSING
                    else omit,
                    stream=False,
                )

            except AnthropicRateLimitError as exc:  # retry on rate limit after delay
                if retry_after := exc.response.headers.get("Retry-After"):
                    ctx.record_warning(
                        event="model.rate_limit",
                        attributes={
                            "model.provider": self._provider,
                            "model.name": config.model,
                            "retry_after": retry_after,
                        },
                    )
                    try:
                        raise ModelRateLimit(
                            provider=self._provider,
                            model=config.model,
                            # adding random extra delay to prevent immediate retry of all pending
                            retry_after=float(retry_after) + random.uniform(0.1, 3.0),  # nosec: B311,
                        ) from exc

                    except ValueError:
                        raise ModelRateLimit(
                            provider=self._provider,
                            model=config.model,
                            # using random delay if value not available
                            retry_after=random.uniform(0.3, 5.0),  # nosec: B311,
                        ) from exc

                else:
                    ctx.record_warning(
                        event="model.rate_limit",
                        attributes={
                            "model.provider": self._provider,
                            "model.name": config.model,
                        },
                    )
                    raise ModelRateLimit(
                        provider=self._provider,
                        model=config.model,
                        # using random delay if value not available
                        retry_after=random.uniform(0.3, 5.0),  # nosec: B311,
                    ) from exc

            except Exception as exc:
                raise ModelOutputFailed(
                    provider=self._provider,
                    model=config.model,
                    reason=str(exc),
                ) from exc

            ctx.record_info(
                metric="model.input_tokens",
                value=completion.usage.input_tokens,
                unit="tokens",
                kind="counter",
                attributes={
                    "model.provider": self._provider,
                    "model.name": completion.model,
                },
            )
            # Tokens served from cache (input side)
            ctx.record_info(
                metric="model.input_tokens.cached",
                value=completion.usage.cache_read_input_tokens or 0,
                unit="tokens",
                kind="counter",
                attributes={
                    "model.provider": self._provider,
                    "model.name": completion.model,
                },
            )
            ctx.record_info(
                metric="model.output_tokens",
                value=completion.usage.output_tokens,
                unit="tokens",
                kind="counter",
                attributes={
                    "model.provider": self._provider,
                    "model.name": completion.model,
                },
            )
            # Anthropic does not expose cached output tokens; omit to avoid mislabeling

            if completion.stop_reason == "refusal":
                raise ModelOutputFailed(
                    provider=self._provider,
                    model=config.model,
                    reason=completion.stop_reason,
                )

            if completion.stop_reason == "max_tokens":
                raise ModelOutputLimit(
                    provider=self._provider,
                    model=config.model,
                    max_output_tokens=unwrap_missing(config.max_output_tokens, default=0),
                    content=tuple(_completion_as_content(completion.content)),
                )

            return ModelOutput.of(
                *_completion_as_content(completion.content),
                meta={
                    "identifier": completion.id,
                    "model": config.model,
                    "stop_reason": completion.stop_reason,
                },
            )

    async def _completion_stream(
        self,
        *,
        instructions: ModelInstructions,
        context: ModelContext,
        tools: ModelToolsDeclaration,
        output: ModelOutputSelection,
        config: AnthropicConfig,
        prefill: Multimodal | None = None,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput]:
        async with ctx.scope("model.completion.stream"):
            ctx.log_warning(
                "Anthropic completion streaming is not supported yet,"
                " using regular response instead."
            )

            model_output: ModelOutput = await self._completion(
                instructions=instructions,
                context=context,
                tools=tools,
                output=output,
                config=config,
                prefill=prefill,
            )

            for block in model_output.blocks:
                if isinstance(block, MultimodalContent):
                    for part in block.parts:
                        yield part

                else:
                    assert isinstance(block, ModelReasoning | ModelToolRequest)  # nosec: B101
                    yield block


def _context_messages(  # noqa: PLR0912
    context: ModelContext,
    /,
) -> Generator[MessageParam]:
    for element in context:
        content: list[
            TextBlockParam
            | ImageBlockParam
            | ThinkingBlockParam
            | RedactedThinkingBlockParam
            | ToolUseBlockParam
            | ToolResultBlockParam
        ] = []

        if isinstance(element, ModelInput):
            for block in element.blocks:
                if isinstance(block, MultimodalContent):
                    content.extend(
                        _content_elements(
                            block,
                            cache_type=element.meta.get_str("cache"),
                        )
                    )

                else:
                    content.append(
                        {
                            "tool_use_id": block.identifier,
                            "type": "tool_result",
                            "is_error": block.handling == "error",
                            "content": cast(  # there will be no thinking within tool results
                                Iterable[TextBlockParam | ImageBlockParam],
                                _content_elements(
                                    block.content,
                                    cache_type=None,
                                ),
                            ),
                        }
                    )

            yield {
                "role": "user",
                "content": content,
            }

        else:
            assert isinstance(element, ModelOutput)  # nosec: B101
            for block in element.blocks:
                if isinstance(block, MultimodalContent):
                    content.extend(
                        _content_elements(
                            block,
                            cache_type=element.meta.get_str("cache"),
                        )
                    )

                elif isinstance(block, ModelReasoning):
                    match block.meta.kind:
                        case "thinking":
                            content.append(
                                {
                                    "type": "thinking",
                                    "thinking": block.content.to_str(),
                                    "signature": block.meta.get_str(
                                        "signature",
                                        default="",
                                    ),
                                }
                            )

                        case "redacted_thinking":
                            content.append(
                                {
                                    "type": "redacted_thinking",
                                    "data": block.content.to_str(),
                                }
                            )

                        case other:
                            raise ValueError(f"Unsupported reasoning element: {other}")

                else:
                    assert isinstance(block, ModelToolRequest)  # nosec: B101
                    content.append(
                        {
                            "id": block.identifier,
                            "type": "tool_use",
                            "name": block.tool,
                            "input": as_dict(block.arguments),
                        }
                    )

            yield {
                "role": "assistant",
                "content": content,
            }


def _content_elements(
    content: MultimodalContent,
    /,
    cache_type: str | None,
) -> Generator[TextBlockParam | ImageBlockParam]:
    last_cacheable: TextBlockParam | ImageBlockParam | None = None
    for part in content.parts:
        if isinstance(part, TextContent):
            text_block: TextBlockParam = {
                "type": "text",
                "text": part.text,
            }
            last_cacheable = text_block
            yield text_block

        elif isinstance(part, ResourceContent):
            # Only selected image resources are supported by Anthropic message blocks
            if not part.mime_type.startswith("image"):
                raise ValueError(f"Unsupported message content mime type: {part.mime_type}")

            image_block: ImageBlockParam = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": cast(Any, part.mime_type),
                    "data": part.data,
                },
            }
            last_cacheable = image_block
            yield image_block

        elif isinstance(part, ResourceReference):
            # Only image resources are supported by Anthropic message blocks
            if part.mime_type and not part.mime_type.startswith("image"):
                raise ValueError(f"Unsupported message content mime type: {part.mime_type}")

            # TODO: auto resolve non http resources
            image_block: ImageBlockParam = {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": part.uri,
                },
            }

            last_cacheable = image_block
            yield image_block

        else:
            assert isinstance(part, ArtifactContent)  # nosec: B101
            # Skip artifacts that are marked as hidden
            if part.hidden:
                continue

            text_block: TextBlockParam = {
                "type": "text",
                "text": part.artifact.to_str(),
            }
            last_cacheable = text_block
            yield text_block

    if cache_type is None or last_cacheable is None:
        return

    # insert cache marker to the last cacheable element
    last_cacheable["cache_control"] = {  # pyright: ignore[reportGeneralTypeIssues]
        "type": cache_type,
    }


def _completion_as_content(
    completion: Iterable[ContentBlock],
    /,
) -> Generator[MultimodalContent | ModelReasoning | ModelToolRequest]:
    for block in completion:
        if isinstance(block, TextBlock):
            if block.citations:
                yield MultimodalContent.of(
                    TextContent(
                        text=block.text,
                        meta=Meta.of({"citations": _extract_text_citations(block)}),
                    )
                )

            else:
                yield MultimodalContent.of(
                    TextContent(
                        text=block.text,
                    )
                )

        elif isinstance(block, ThinkingBlock):
            yield ModelReasoning.of(
                block.thinking,
                meta={
                    "kind": "thinking",
                    "signature": block.signature,
                },
            )

        elif isinstance(block, RedactedThinkingBlock):
            yield ModelReasoning.of(
                block.data,
                meta={"kind": "redacted_thinking"},
            )

        elif isinstance(block, ToolUseBlock):
            yield ModelToolRequest(
                identifier=block.id,
                tool=block.name,
                arguments=cast(dict[str, Any], block.input),
                meta=META_EMPTY,
            )

        else:
            raise ValueError(f"Unsupported content block {type(block)}")


def _extract_text_citations(
    block: TextBlock,
) -> Sequence[Mapping[str, Any]]:
    if block.citations is None:
        return ()

    citations: list[dict[str, Any]] = []
    for citation in block.citations:
        source: str | None
        kind: str
        start_index: int
        end_index: int
        text: str
        if isinstance(citation, CitationCharLocation):
            source = citation.document_title or citation.file_id
            kind = citation.type
            start_index = citation.start_char_index
            end_index = citation.end_char_index
            text = citation.cited_text

        elif isinstance(citation, CitationContentBlockLocation):
            source = citation.document_title or citation.file_id
            kind = citation.type
            start_index = citation.start_block_index
            end_index = citation.end_block_index
            text = citation.cited_text

        elif isinstance(citation, CitationPageLocation):
            source = citation.document_title or citation.file_id
            kind = citation.type
            start_index = citation.start_page_number
            end_index = citation.end_page_number
            text = citation.cited_text

        elif isinstance(citation, CitationsSearchResultLocation):
            source = citation.source
            kind = citation.type
            start_index = citation.start_block_index
            end_index = citation.end_block_index
            text = citation.cited_text

        else:
            continue  # skip other/unsupported

        citations.append(
            {
                "source": source,
                "kind": kind,
                "start": start_index,
                "end": end_index,
                "text": text,
            }
        )

    return citations


def _thinking_budget_config(
    budget: int | None | Missing,
) -> ThinkingConfigParam | Omit:
    if budget is MISSING:
        return omit

    assert isinstance(budget, int | None)  # nosec: B101

    if budget is None or budget <= 0:
        return {"type": "disabled"}

    return {
        "type": "enabled",
        "budget_tokens": budget,
    }


def _tools_as_tool_params(
    tools: Sequence[ModelToolSpecification],
    /,
) -> Sequence[ToolParam]:
    tool_params: list[ToolParam] = []
    for tool in tools:
        input_schema: dict[str, Any]
        if parameters := tool.parameters:
            input_schema = cast(dict[str, Any], parameters)

        else:
            # Anthropic requires input_schema; provide an empty object schema when None
            input_schema = {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }

        tool_params.append(
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": input_schema,
            }
        )

    return tool_params


def _tools_selection_as_tool_choice(
    selection: ModelToolsSelection,
    /,
) -> ToolChoiceParam:
    return _TOOL_CHOICE_PARAM.get(
        selection,
        {
            "type": "tool",
            "name": selection,
        },
    )


_TOOL_CHOICE_PARAM: Mapping[str, ToolChoiceParam] = {
    "auto": {"type": "auto"},
    "required": {"type": "any"},
    "none": {"type": "none"},
}
