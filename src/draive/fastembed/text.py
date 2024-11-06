from collections.abc import Sequence
from typing import Any

from fastembed.text.text_embedding import (  # pyright: ignore[reportMissingTypeStubs]
    TextEmbedding as FastembedTextEmbedding,
)
from haiway import asynchronous, ctx

from draive.embedding import Embedded, TextEmbedding

__all__ = [
    "fastembed_text_embedding",
]


async def fastembed_text_embedding(
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    cache_dir: str | None = "./embedding_models/",
) -> TextEmbedding:
    # TODO: verify if loading model should be asynchronous here
    embedding_model: FastembedTextEmbedding = await _text_embedding_model(
        model_name=model_name,
        cache_dir=cache_dir,
    )

    async def fastembed_embed_text(
        values: Sequence[str],
        **extra: Any,
    ) -> list[Embedded[str]]:
        with ctx.scope("text_embedding"):
            return await _fastembed_text_embedding(
                embedding_model,
                values,
            )

    return TextEmbedding(embed=fastembed_embed_text)


@asynchronous
def _text_embedding_model(
    model_name: str,
    cache_dir: str | None = "./embedding_models/",
) -> FastembedTextEmbedding:
    return FastembedTextEmbedding(
        model_name=model_name,
        cache_dir=cache_dir,
    )


@asynchronous
def _fastembed_text_embedding(
    embedding_model: FastembedTextEmbedding,
    texts: Sequence[str],
    /,
) -> list[Embedded[str]]:
    return [
        Embedded(
            value=value,
            vector=embedding.tolist(),
        )
        for (
            value,
            embedding,  # pyright: ignore[reportUnknownVariableType]
        ) in zip(
            texts,
            embedding_model.embed(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                texts
            ),
            strict=True,
        )
    ]
