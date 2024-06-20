from collections.abc import Sequence
from typing import Any

from fastembed.text.text_embedding import TextEmbedding  # pyright: ignore[reportMissingTypeStubs]

from draive.embedding import Embedded
from draive.fastembed.config import FastembedTextConfig
from draive.scope import ctx
from draive.utils import cache, run_async

__all__ = [
    "fastembed_text_embedding",
]


async def fastembed_text_embedding(
    values: Sequence[str],
    **extra: Any,
) -> list[Embedded[str]]:
    config: FastembedTextConfig = ctx.state(FastembedTextConfig).updated(**extra)
    with ctx.nested(
        "fastembed_text_embedding",
        metrics=[config],
    ):
        return await _fastembed_text_embedding(
            config,
            values,
        )


@cache(limit=1)
def _text_embedding_model(
    model_name: str,
    cache_dir: str | None,
) -> TextEmbedding:
    return TextEmbedding(
        model_name=model_name,
        cache_dir=cache_dir,
    )


@run_async
def _fastembed_text_embedding(
    config: FastembedTextConfig,
    texts: Sequence[str],
    /,
) -> list[Embedded[str]]:
    embedding_model: TextEmbedding = _text_embedding_model(
        model_name=config.model,
        cache_dir=config.cache_dir,
    )
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
