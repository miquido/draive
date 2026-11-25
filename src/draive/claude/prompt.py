from collections.abc import Mapping, Sequence
from typing import Any

from draive.multimodal import MultimodalContent, TextContent
from draive.multimodal.artifact import ArtifactContent
from draive.resources import ResourceContent

__all__ = ("prompt_blocks",)


def prompt_blocks(
    content: MultimodalContent,
    /,
) -> Sequence[Mapping[str, Any]]:
    """Prepare the content blocks of the user message.

    Parameters
    ----------
    content
        Content of the message, text and images are supported, hidden artifacts are left out.

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
