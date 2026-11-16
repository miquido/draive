from collections.abc import Sequence
from itertools import pairwise
from random import Random

import pytest
from haiway import Meta

from draive.gemini.content import coalesce_audio_blocks
from draive.models import ModelOutputBlock, ModelReasoning, ModelToolRequest
from draive.multimodal import MultimodalContent, TextContent
from draive.resources import ResourceContent, ResourceReference


def _audio(data: bytes, mime: str = "audio/pcm;rate=24000") -> MultimodalContent:
    return MultimodalContent.of(ResourceContent.of(data, mime_type=mime))


def _random_blocks(rng: Random) -> list[ModelOutputBlock]:
    blocks: list[ModelOutputBlock] = []
    for index in range(rng.randint(0, 40)):
        match rng.randint(0, 6):
            case 0:
                blocks.append(MultimodalContent.of(TextContent.of(f"t{index}")))
            case 1:
                blocks.append(ModelReasoning.of(TextContent.of(f"r{index}")))
            case 2:
                blocks.append(ModelToolRequest.of(f"id{index}", tool="tool", arguments={}))
            case 3:
                blocks.append(
                    MultimodalContent.of(
                        ResourceReference.of(f"file://{index}", mime_type="audio/x")
                    )
                )
            case 4:
                blocks.append(_audio(rng.randbytes(rng.randint(1, 8)), "audio/other"))
            case 5:
                blocks.append(
                    MultimodalContent.of(
                        ResourceContent.of(rng.randbytes(3), mime_type="image/png")
                    )
                )
            case _:
                blocks.append(_audio(rng.randbytes(rng.randint(1, 8))))
    return blocks


def _is_audio(block: ModelOutputBlock) -> bool:
    return (
        isinstance(block, MultimodalContent)
        and len(block.parts) == 1
        and isinstance(block.parts[0], ResourceContent)
        and block.parts[0].mime_type.startswith("audio/")
    )


def _audio_runs(
    blocks: Sequence[ModelOutputBlock],
) -> list[tuple[str, bytes]]:
    runs: list[tuple[str, bytes]] = []
    previous_audio: str | None = None
    for block in blocks:
        if _is_audio(block):
            assert isinstance(block, MultimodalContent)
            part = block.parts[0]
            assert isinstance(part, ResourceContent)
            if previous_audio == part.mime_type:
                runs[-1] = (part.mime_type, runs[-1][1] + part.to_bytes())
            else:
                runs.append((part.mime_type, part.to_bytes()))
            previous_audio = part.mime_type
        else:
            previous_audio = None
    return runs


@pytest.mark.parametrize("seed", range(25))
def test_coalesce_preserves_bytes_order_and_is_idempotent(seed: int) -> None:
    blocks = _random_blocks(Random(seed))
    merged = coalesce_audio_blocks(blocks)

    # audio bytes are the concatenation of each adjacent same-mime run
    assert _audio_runs(merged) == _audio_runs(blocks)
    # no two adjacent audio blocks of the same mime remain
    audio_mimes = [
        (
            b.parts[0].mime_type
            if _is_audio(b)
            and isinstance(b, MultimodalContent)
            and isinstance(b.parts[0], ResourceContent)
            else None
        )
        for b in merged
    ]
    assert all(not (a is not None and a == b) for a, b in pairwise(audio_mimes))
    # all other blocks are untouched and keep their order
    assert [b for b in merged if not _is_audio(b)] == [b for b in blocks if not _is_audio(b)]
    assert coalesce_audio_blocks(merged) == tuple(merged)


def test_coalesce_does_not_merge_across_other_blocks_or_mime_types() -> None:
    text = MultimodalContent.of(TextContent.of("x"))
    merged = coalesce_audio_blocks(
        [_audio(b"a"), _audio(b"b"), text, _audio(b"c"), _audio(b"d", "audio/pcm;rate=16000")]
    )

    assert len(merged) == 4
    assert merged[2] == _audio(b"c")
    assert merged[0] == _audio(b"ab")


def test_coalesced_block_keeps_first_meta() -> None:
    first = MultimodalContent.of(
        ResourceContent.of(b"a", mime_type="audio/pcm", meta=Meta.of({"k": "v"}))
    )
    merged = coalesce_audio_blocks([first, _audio(b"b", "audio/pcm")])

    part = merged[0]
    assert isinstance(part, MultimodalContent)
    resource = part.parts[0]
    assert isinstance(resource, ResourceContent)
    assert resource.meta == Meta.of({"k": "v"})
    assert resource.to_bytes() == b"ab"
