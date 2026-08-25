from base64 import b64decode, b64encode
from collections.abc import Generator, MutableSequence
from uuid import uuid4

from google.genai.types import FunctionResponseDict, FunctionResponsePartDict, Part, PartDict
from haiway import Meta, as_dict

from draive.models import (
    ModelInputBlocks,
    ModelOutputBlock,
    ModelOutputBlocks,
    ModelOutputChunk,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelToolResponse,
)
from draive.multimodal import ArtifactContent, MultimodalContent, TextContent
from draive.resources import ResourceContent, ResourceReference

__all__ = (
    "block_parts",
    "content_parts",
    "function_response",
    "part_as_output_blocks",
    "part_as_stream_elements",
)


def _reasoning_meta(
    part: Part,
    /,
) -> Meta:
    # the signature key is omitted instead of carrying None - reasoning fragment
    # metadata is merged within a block, so an explicit None would erase the
    # signature of a preceding fragment
    if part.thought_signature:
        return Meta.of(
            {
                "kind": "thought",
                "signature": b64encode(part.thought_signature).decode(),
            }
        )

    return Meta.of({"kind": "thought"})


def part_as_stream_elements(
    part: Part,
    /,
) -> Generator[ModelOutputChunk]:
    """Convert a provider content part into output stream chunks."""
    if part.text:
        if part.thought:
            yield ModelReasoningChunk.of(
                TextContent.of(part.text),
                # a signature closes the thought block it was produced for
                final=part.thought_signature is not None,
                meta=_reasoning_meta(part),
            )

        else:
            yield TextContent.of(part.text)

    if part.function_call and part.function_call.name:
        yield ModelToolRequest.of(
            # the api reports no identifier outside of live sessions, a local one
            # is what correlates the response with its request
            part.function_call.id or str(uuid4()),
            tool=part.function_call.name,
            arguments=part.function_call.args,
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


def part_as_output_blocks(
    part: Part,
    /,
) -> Generator[ModelOutputBlock]:
    """Convert a provider content part into context output blocks."""
    if part.text:
        if part.thought:
            yield ModelReasoning.of(
                TextContent.of(part.text),
                meta=_reasoning_meta(part),
            )

        else:
            yield MultimodalContent.of(TextContent.of(part.text))

    if part.function_call and part.function_call.name:
        yield ModelToolRequest.of(
            # the api reports no identifier outside of live sessions, a local one
            # is what correlates the response with its request
            part.function_call.id or str(uuid4()),
            tool=part.function_call.name,
            arguments=part.function_call.args,
            meta=Meta.of(
                {
                    "signature": b64encode(part.thought_signature).decode(),
                }
            )
            if part.thought_signature
            else Meta.empty,
        )

    if part.inline_data and part.inline_data.data:
        yield MultimodalContent.of(
            ResourceContent.of(
                part.inline_data.data,
                mime_type=part.inline_data.mime_type or "application/octet-stream",
            )
        )

    if part.file_data and part.file_data.file_uri:
        yield MultimodalContent.of(
            ResourceReference.of(
                part.file_data.file_uri,
                mime_type=part.file_data.mime_type,
            )
        )


def function_response(
    response: ModelToolResponse,
    /,
) -> FunctionResponseDict:
    """Encode a tool response as a provider function response.

    The ``response`` payload is plain json, so media has to travel through the
    dedicated ``parts`` field instead - inlining it into the payload would deliver
    a base64 blob the model cannot interpret as media.
    """
    text: MutableSequence[str] = []
    parts: MutableSequence[FunctionResponsePartDict] = []
    for part in response.content.parts:
        if isinstance(part, TextContent):
            text.append(part.text)

        elif isinstance(part, ResourceContent):
            parts.append(
                {
                    "inline_data": {
                        "data": part.to_bytes(),
                        "mime_type": part.mime_type,
                    },
                }
            )

        elif isinstance(part, ResourceReference):
            parts.append(
                {
                    "file_data": {
                        "file_uri": part.uri,
                        "mime_type": part.mime_type,
                    },
                }
            )

        else:
            assert isinstance(part, ArtifactContent)  # nosec: B101
            if part.hidden:
                continue  # skip hidden

            text.append(part.to_str())

    payload: FunctionResponseDict = {
        "id": response.identifier,
        "name": response.tool,
        # the payload key selects how the model treats the result
        "response": {"error" if response.status == "error" else "output": "".join(text)},
    }
    if parts:
        payload["parts"] = parts

    return payload


def block_parts(
    blocks: ModelInputBlocks | ModelOutputBlocks,
    /,
) -> Generator[PartDict]:
    """Convert context blocks into provider content parts."""
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
            yield {"function_response": function_response(block)}

        elif isinstance(block, ModelReasoning):
            if block.meta.kind != "thought":
                raise ValueError(f"Unsupported reasoning element: {block.meta.kind}")

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
            assert isinstance(block, MultimodalContent)  # nosec: B101
            yield from content_parts(block)


def content_parts(
    content: MultimodalContent,
    /,
) -> Generator[PartDict]:
    """Convert multimodal content into provider content parts."""
    for part in content.parts:
        if isinstance(part, TextContent):
            yield {"text": part.text}

        elif isinstance(part, ResourceContent):
            yield {
                "inline_data": {
                    # decode base64 back to raw bytes for provider
                    "data": part.to_bytes(),
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

            yield {"text": part.to_str()}
