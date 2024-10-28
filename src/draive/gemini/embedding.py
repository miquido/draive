from collections.abc import Sequence
from typing import Any

from haiway import ctx

from draive.embedding import Embedded, ValueEmbedder
from draive.gemini.client import SHARED, GeminiClient
from draive.gemini.config import GeminiEmbeddingConfig

__all__ = [
    "gemini_text_embedding",
]


def gemini_text_embedding(
    client: GeminiClient = SHARED,
    /,
) -> ValueEmbedder[str]:
    async def gemini_embed_text(
        values: Sequence[str],
        **extra: Any,
    ) -> list[Embedded[str]]:
        config: GeminiEmbeddingConfig = ctx.state(GeminiEmbeddingConfig).updated(**extra)
        with ctx.scope("embed_text"):
            results: list[list[float]] = await client.embedding(
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

    return gemini_embed_text
