import json
from collections.abc import Mapping, Sequence
from typing import Any

from draive.models import ModelContext, ModelInput, ModelToolRequest, ModelToolResponse
from draive.multimodal import MultimodalContent, MultimodalTag, TextContent
from draive.multimodal.artifact import ArtifactContent
from draive.resources import ResourceContent

__all__ = (
    "prompt_blocks",
    "render_transcript",
)

_HISTORY_INTRODUCTION = "The conversation so far, the oldest message first:"
_HISTORY_CONCLUSION = "Continue the conversation by answering the latest user message below."


def render_transcript(
    context: ModelContext,
    /,
) -> str:
    """Render the conversation as a text a fresh Claude Code session can continue.

    Parameters
    ----------
    context
        Conversation to render, reasoning is left out.

    Returns
    -------
    str
        Messages tagged with their roles, tool calls and results by the name of the tool.
    """
    tags: list[MultimodalTag] = []
    for element in context:
        if isinstance(element, ModelInput):
            for input_block in element.input:
                if isinstance(input_block, ModelToolResponse):
                    tags.append(
                        MultimodalTag.of(
                            *input_block.content.parts,
                            name="tool_response",
                            meta={"tool": input_block.tool, "status": input_block.status},
                        )
                    )

                else:
                    tags.append(MultimodalTag.of(*input_block.parts, name="user"))

        else:
            for output_block in element.output:
                if isinstance(output_block, ModelToolRequest):
                    tags.append(
                        MultimodalTag.of(
                            json.dumps(output_block.arguments, default=str),
                            name="tool_request",
                            meta={"tool": output_block.tool},
                        )
                    )

                elif isinstance(output_block, MultimodalContent):
                    tags.append(MultimodalTag.of(*output_block.parts, name="assistant"))

    return MultimodalContent.of(*tags).to_str()


def prompt_blocks(
    content: MultimodalContent,
    /,
    *,
    history: str | None = None,
) -> Sequence[Mapping[str, Any]]:
    """Prepare the content blocks of the user message.

    Parameters
    ----------
    content
        Content of the message, text and images are supported, hidden artifacts are left out.
    history
        Rendered conversation preceding the message.

    Returns
    -------
    Sequence[Mapping[str, Any]]
        Content blocks of the message.

    Raises
    ------
    ValueError
        When the content has a part which can not be sent, or nothing to send.
    """
    blocks: list[dict[str, Any]] = []
    if history:
        blocks.append(
            {
                "type": "text",
                "text": f"{_HISTORY_INTRODUCTION}\n{history}\n\n{_HISTORY_CONCLUSION}\n",
            }
        )

    for part in content.parts:
        if isinstance(part, TextContent):
            blocks.append({"type": "text", "text": part.text})

        elif isinstance(part, ResourceContent) and part.mime_type.startswith("image/"):
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.mime_type,
                        "data": part.data,
                    },
                }
            )

        elif isinstance(part, ArtifactContent) and part.hidden:
            continue

        else:
            raise ValueError(f"Claude Code accepts text and images only, got {type(part).__name__}")

    if not any(block["type"] == "text" and block["text"].strip() for block in blocks) and not any(
        block["type"] == "image" for block in blocks
    ):
        raise ValueError("Claude Code requires a message to answer")

    return tuple(blocks)
