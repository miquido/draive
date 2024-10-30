from collections.abc import Sequence
from io import BytesIO
from typing import Any

from fastembed.image.image_embedding import (  # pyright: ignore[reportMissingTypeStubs]
    ImageEmbedding,
)
from haiway import asynchronous, ctx

from draive.embedding import Embedded, ValueEmbedder

__all__ = [
    "fastembed_image_embedding",
]


async def fastembed_image_embedding(
    model_name: str,
    cache_dir: str | None = "./embedding_models/",
    /,
) -> ValueEmbedder[bytes]:
    # TODO: verify if loading model should be asynchronous here
    embedding_model: ImageEmbedding = await _image_embedding_model(
        model_name=model_name,
        cache_dir=cache_dir,
    )

    async def fastembed_embed_image(
        values: Sequence[bytes],
        **extra: Any,
    ) -> list[Embedded[bytes]]:
        with ctx.scope("image_embedding"):
            return await _fastembed_image_embedding(
                embedding_model,
                values,
            )

    return fastembed_embed_image


@asynchronous
def _image_embedding_model(
    model_name: str,
    cache_dir: str | None,
) -> ImageEmbedding:
    return ImageEmbedding(
        model_name=model_name,
        cache_dir=cache_dir,
    )


@asynchronous
def _fastembed_image_embedding(
    embedding_model: ImageEmbedding,
    images: Sequence[bytes],
    /,
) -> list[Embedded[bytes]]:
    return [
        Embedded(
            value=value,  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            vector=embedding.tolist(),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )
        for (
            value,  # pyright: ignore[reportUnknownVariableType]
            embedding,  # pyright: ignore[reportUnknownVariableType]
        ) in zip(
            images,
            embedding_model.embed(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                [BytesIO(image) for image in images]  # pyright: ignore[reportArgumentType]
            ),
            strict=True,
        )
    ]
