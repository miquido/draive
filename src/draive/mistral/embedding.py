from collections.abc import Iterable
from typing import Any

from draive.embedding import Embedded
from draive.mistral.client import MistralClient
from draive.mistral.config import MistralEmbeddingConfig
from draive.scope import ctx

__all__ = [
    "mistral_embed_text",
]


async def mistral_embed_text(
    values: Iterable[str],
    **extra: Any,
) -> list[Embedded[str]]:
    config: MistralEmbeddingConfig = ctx.state(MistralEmbeddingConfig).updated(**extra)
    with ctx.nested("text_embedding", metrics=[config]):
        results: list[list[float]] = await ctx.dependency(MistralClient).embedding(
            config=config,
            inputs=[str(value) for value in values],
        )

        return [
            Embedded(
                value=embedded[0],
                vector=embedded[1],
            )
            for embedded in zip(
                values,
                results,  # [result.embedding for result in results.data],
                strict=True,
            )
        ]
