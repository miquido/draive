from collections.abc import Sequence
from typing import Any

from draive.embedding import Embedded
from draive.gemini.client import GeminiClient
from draive.gemini.config import GeminiEmbeddingConfig
from draive.scope import ctx

__all__ = [
    "gemini_embed_text",
]


async def gemini_embed_text(
    values: Sequence[str],
    **extra: Any,
) -> list[Embedded[str]]:
    config: GeminiEmbeddingConfig = ctx.state(GeminiEmbeddingConfig).updated(**extra)
    with ctx.nested("gemini_embed_text", metrics=[config]):  # pyright: ignore[reportDeprecated]
        results: list[list[float]] = await ctx.dependency(GeminiClient).embedding(  # pyright: ignore[reportDeprecated]
            config=config,
            inputs=values,
        )

        return [
            Embedded(
                value=embedded[0],
                vector=embedded[1],
            )
            for embedded in zip(
                values,
                results,
                strict=True,
            )
        ]
