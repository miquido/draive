from asyncio import gather
from base64 import b64encode
from collections.abc import Sequence
from itertools import chain
from typing import Any, cast

from cohere import EmbedByTypeResponse
from haiway import as_list, ctx

from draive.cohere.api import CohereAPI
from draive.cohere.config import CohereImageEmbeddingConfig, CohereTextEmbeddingConfig
from draive.embedding import Embedded, ImageEmbedding, TextEmbedding

__all__ = [
    "CohereEmbedding",
]


class CohereEmbedding(CohereAPI):
    def text_embedding(self) -> TextEmbedding:
        """
        Prepare TextEmbedding implementation using Cohere service.
        """
        return TextEmbedding(embed=self.create_text_embeddings)

    async def create_text_embeddings(
        self,
        values: Sequence[str],
        /,
        *,
        config: CohereTextEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[str]]:
        """
        Create texts embedding with Cohere embedding service.
        """
        embedding_config: CohereTextEmbeddingConfig = config or ctx.state(
            CohereTextEmbeddingConfig
        ).updated(**extra)
        with ctx.scope("cohere_text_embedding", embedding_config):
            ctx.record(embedding_config)
            texts: list[str] = as_list(values)
            responses: list[EmbedByTypeResponse] = await gather(
                *[
                    self._client.embed(
                        model=embedding_config.model,
                        texts=texts[index : index + embedding_config.batch_size],
                        embedding_types=["float"],
                        input_type=embedding_config.purpose,
                    )
                    for index in range(0, len(texts), embedding_config.batch_size)
                ]
            )

            return [
                Embedded(
                    value=embedded[0],
                    vector=embedded[1],
                )
                for embedded in zip(
                    texts,
                    chain.from_iterable(
                        [
                            cast(list[list[float]], response.embeddings.float_)
                            for response in responses
                        ]
                    ),
                    strict=True,
                )
            ]

    def image_embedding(self) -> ImageEmbedding:
        """
        Prepare ImageEmbedding implementation using Cohere service.
        """
        return ImageEmbedding(embed=self.create_image_embeddings)

    async def create_image_embeddings(
        self,
        values: Sequence[bytes],
        /,
        *,
        config: CohereImageEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[bytes]]:
        """
        Create image embedding with Cohere embedding service.
        """
        embedding_config: CohereImageEmbeddingConfig = config or ctx.state(
            CohereImageEmbeddingConfig
        ).updated(**extra)
        with ctx.scope("cohere_image_embedding", embedding_config):
            ctx.record(embedding_config)
            images: list[bytes] = as_list(values)
            responses: list[EmbedByTypeResponse] = await gather(
                *[
                    self._client.embed(
                        model=embedding_config.model,
                        images=[
                            f"data:image/jpeg;base64,{b64encode(image).decode('utf-8')}"
                            for image in images[index : index + embedding_config.batch_size]
                        ],
                        embedding_types=["float"],
                        input_type="image",
                    )
                    for index in range(0, len(images), embedding_config.batch_size)
                ]
            )

            return [
                Embedded(
                    value=embedded[0],
                    vector=embedded[1],
                )
                for embedded in zip(
                    images,
                    chain.from_iterable(
                        [
                            cast(list[list[float]], response.embeddings.float_)
                            for response in responses
                        ]
                    ),
                    strict=True,
                )
            ]
