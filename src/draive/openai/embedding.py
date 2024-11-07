from collections.abc import Sequence
from typing import Any

from haiway import ctx

from draive.embedding import Embedded, TextEmbedding
from draive.openai.client import OpenAIClient
from draive.openai.config import OpenAIEmbeddingConfig

__all__ = [
    "openai_text_embedding",
]


def openai_text_embedding(
    client: OpenAIClient | None = None,
    /,
) -> TextEmbedding:
    client = client or OpenAIClient.shared()

    async def openai_embed_text(
        values: Sequence[str],
        **extra: Any,
    ) -> list[Embedded[str]]:
        with ctx.scope("embed_text"):
            config: OpenAIEmbeddingConfig = ctx.state(OpenAIEmbeddingConfig).updated(**extra)
            results: list[list[float]] = await client.embedding(  # pyright: ignore[reportDeprecated]
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

    return TextEmbedding(embed=openai_embed_text)
