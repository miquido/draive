from collections.abc import Sequence
from typing import Any

from draive.embedding import Embedded
from draive.mistral.client import MistralClient
from draive.mistral.config import MistralEmbeddingConfig
from draive.scope import ctx

__all__ = [
    "mistral_embed_text",
]


async def mistral_embed_text(
    values: Sequence[str],
    **extra: Any,
) -> list[Embedded[str]]:
    config: MistralEmbeddingConfig = ctx.state(MistralEmbeddingConfig).updated(**extra)
    with ctx.nested("mistral_embed_text", metrics=[config]):
        results: list[list[float]] = await ctx.dependency(MistralClient).embedding(
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
