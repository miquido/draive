from collections.abc import Sequence
from typing import Any

from haiway import ctx

from draive.embedding import Embedded, TextEmbedding
from draive.mistral.client import MistralClient
from draive.mistral.config import MistralEmbeddingConfig

__all__ = [
    "mistral_text_embedding",
]


def mistral_text_embedding(
    client: MistralClient | None = None,
    /,
) -> TextEmbedding:
    client = client or MistralClient.shared()

    async def mistral_embed_text(
        values: Sequence[str],
        **extra: Any,
    ) -> list[Embedded[str]]:
        config: MistralEmbeddingConfig = ctx.state(MistralEmbeddingConfig).updated(**extra)
        with ctx.scope("mistral_embed_text", config):
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

    return TextEmbedding(embed=mistral_embed_text)
