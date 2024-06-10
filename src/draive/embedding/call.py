from collections.abc import Iterable
from typing import Any

from draive.embedding.embedded import Embedded
from draive.embedding.state import ImageEmbedding, TextEmbedding
from draive.scope import ctx

__all__ = [
    "embed_text",
    "embed_image",
]


async def embed_text(
    values: Iterable[str],
    **extra: Any,
) -> list[Embedded[str]]:
    return await ctx.state(TextEmbedding).embed(
        values=values,
        **extra,
    )


async def embed_image(
    values: Iterable[bytes],
    **extra: Any,
) -> list[Embedded[bytes]]:
    return await ctx.state(ImageEmbedding).embed(
        values=values,
        **extra,
    )
