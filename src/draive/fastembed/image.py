from collections.abc import Iterable
from io import BytesIO
from typing import Any

from fastembed.image.image_embedding import (  # pyright: ignore[reportMissingTypeStubs]
    ImageEmbedding,
)

from draive.embedding import Embedded
from draive.fastembed.config import FastembedImageConfig
from draive.scope import ctx
from draive.utils import cache, run_async

__all__ = [
    "fastembed_image_embedding",
]


async def fastembed_image_embedding(
    values: Iterable[bytes],
    **extra: Any,
) -> list[Embedded[bytes]]:
    config: FastembedImageConfig = ctx.state(FastembedImageConfig).updated(**extra)
    with ctx.nested(
        "fastembed_image_embedding",
        metrics=[config],
    ):
        return await _fastembed_image_embedding(
            config,
            values,
        )


@cache(limit=1)
def _image_embedding_model(
    model_name: str,
    cache_dir: str | None,
) -> ImageEmbedding:
    return ImageEmbedding(
        model_name=model_name,
        cache_dir=cache_dir,
    )


@run_async
def _fastembed_image_embedding(
    config: FastembedImageConfig,
    values: Iterable[bytes],
) -> list[Embedded[bytes]]:
    embedding_model: ImageEmbedding = _image_embedding_model(
        model_name=config.model,
        cache_dir=config.cache_dir,
    )
    return [
        Embedded(
            value=value,  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            vector=embedding.tolist(),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )
        for (
            value,  # pyright: ignore[reportUnknownVariableType]
            embedding,  # pyright: ignore[reportUnknownVariableType]
        ) in zip(
            values,
            embedding_model.embed(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                [BytesIO(value) for value in values]  # pyright: ignore[reportArgumentType]
            ),
            strict=True,
        )
    ]
