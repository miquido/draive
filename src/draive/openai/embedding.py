from collections.abc import Iterable

from draive.embedding import Embedded
from draive.openai.client import OpenAIClient
from draive.openai.config import OpenAIEmbeddingConfig
from draive.scope import ctx

__all__ = [
    "openai_embed_text",
]


async def openai_embed_text(
    values: Iterable[str],
) -> list[Embedded[str]]:
    config: OpenAIEmbeddingConfig = ctx.state(OpenAIEmbeddingConfig)
    with ctx.nested("text_embedding", metrics=[config]):
        results: list[list[float]] = await ctx.dependency(OpenAIClient).embedding(
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
