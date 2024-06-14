from collections.abc import Sequence
from typing import Any

from draive.embedding.embedded import Embedded
from draive.embedding.state import ImageEmbedding, TextEmbedding
from draive.scope import ctx

__all__ = [
    "embed_text",
    "embed_texts",
    "embed_image",
    "embed_images",
]


async def embed_text(
    text: str,
    /,
    **extra: Any,
) -> Embedded[str]:
    return (
        await ctx.state(TextEmbedding).embed(
            [text],
            **extra,
        )
    )[0]


async def embed_texts(
    texts: Sequence[str],
    /,
    **extra: Any,
) -> list[Embedded[str]]:
    return await ctx.state(TextEmbedding).embed(
        texts,
        **extra,
    )


async def embed_image(
    image: bytes,
    /,
    **extra: Any,
) -> Embedded[bytes]:
    return (
        await ctx.state(ImageEmbedding).embed(
            [image],
            **extra,
        )
    )[0]


async def embed_images(
    images: Sequence[bytes],
    /,
    **extra: Any,
) -> list[Embedded[bytes]]:
    return await ctx.state(ImageEmbedding).embed(
        images,
        **extra,
    )
