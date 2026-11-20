from collections.abc import Iterable, MutableSequence

from draive.models.types import (
    ModelOutputBlock,
    ModelOutputBlocks,
    ModelOutputChunk,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
)
from draive.multimodal import MultimodalContent, MultimodalContentPart, TextContent

__all__ = ("model_output_blocks",)


def model_output_blocks(
    chunks: Iterable[ModelOutputChunk],
    /,
) -> ModelOutputBlocks:
    """Group output stream chunks into output blocks preserving their order.

    Consecutive content parts collapse into a single ``MultimodalContent`` block
    and reasoning fragments into their provider blocks, while tool requests stay
    as they were received. Switching between the three flushes whatever was
    accumulated so far, which keeps the resulting blocks in stream order.

    Parameters
    ----------
    chunks
        Output chunks in stream order.

    Returns
    -------
    ModelOutputBlocks
        Aggregated output blocks, empty when no chunks were provided.
    """
    blocks: MutableSequence[ModelOutputBlock] = []
    content: MutableSequence[MultimodalContentPart] = []
    reasoning: MutableSequence[ModelReasoningChunk] = []

    for chunk in chunks:
        if type(chunk) is TextContent:
            # text dominates the streams, skip the slow State instance checks below
            if reasoning:
                _flush(blocks, content, reasoning)

            content.append(chunk)
            continue

        match chunk:
            case ModelReasoningChunk():
                if content:
                    _flush(blocks, content, reasoning)

                reasoning.append(chunk)

            case ModelToolRequest():
                _flush(blocks, content, reasoning)
                blocks.append(chunk)

            case content_part:
                if reasoning:
                    _flush(blocks, content, reasoning)

                content.append(content_part)

    _flush(blocks, content, reasoning)

    return tuple(blocks)


def _flush(
    blocks: MutableSequence[ModelOutputBlock],
    content: MutableSequence[MultimodalContentPart],
    reasoning: MutableSequence[ModelReasoningChunk],
    /,
) -> None:
    # only one of the two can be pending - accumulating either flushes the other
    if content:
        blocks.append(MultimodalContent.of(*content))
        content.clear()

    if reasoning:
        blocks.extend(ModelReasoning.blocks(reasoning))
        reasoning.clear()
