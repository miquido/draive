from asyncio import gather
from base64 import urlsafe_b64encode
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, cast

from cohere import EmbedByTypeResponse
from haiway import State, as_list, ctx

from draive.cohere.api import CohereAPI
from draive.cohere.config import CohereImageEmbeddingConfig, CohereTextEmbeddingConfig
from draive.embedding import Embedded, ImageEmbedding, TextEmbedding
from draive.parameters import DataModel

__all__ = ("CohereEmbedding",)


class CohereEmbedding(CohereAPI):
    def text_embedding(self) -> TextEmbedding:
        return TextEmbedding(embedding=self.create_texts_embedding)

    async def create_texts_embedding[Value: DataModel | State](
        self,
        values: Sequence[Value] | Sequence[str],
        /,
        attribute: Callable[[Value], str] | None = None,
        *,
        config: CohereTextEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Value]] | Sequence[Embedded[str]]:
        embedding_config: CohereTextEmbeddingConfig = config or ctx.state(CohereTextEmbeddingConfig)
        async with ctx.scope("cohere_text_embedding"):
            ctx.record_info(
                attributes={
                    "embedding.model": embedding_config.model,
                    "embedding.purpose": embedding_config.purpose,
                    "embedding.batch_size": embedding_config.batch_size,
                },
            )

            attributes: list[str]
            if attribute is None:
                attributes = cast(list[str], as_list(values))

            else:
                attributes = [attribute(cast(Value, value)) for value in values]

            assert all(isinstance(element, str) for element in attributes)  # nosec: B101

            ctx.record_info(
                metric="embedding.items",
                value=len(attributes),
                unit="count",
                kind="counter",
                attributes={
                    "embedding.provider": "cohere",
                    "embedding.model": embedding_config.model,
                    "embedding.type": "text",
                },
            )

            if not attributes:
                return ()  # empty

            ctx.record_info(
                metric="embedding.batches",
                value=(len(attributes) + embedding_config.batch_size - 1)
                // embedding_config.batch_size,
                unit="count",
                kind="counter",
                attributes={
                    "embedding.provider": "cohere",
                    "embedding.model": embedding_config.model,
                    "embedding.type": "text",
                },
            )

            responses: list[EmbedByTypeResponse] = await gather(
                *[
                    self._client.embed(
                        model=embedding_config.model,
                        texts=attributes[index : index + embedding_config.batch_size],
                        embedding_types=["float"],
                        input_type=embedding_config.purpose,
                    )
                    for index in range(0, len(attributes), embedding_config.batch_size)
                ]
            )

            return cast(
                Sequence[Embedded[Value]] | Sequence[Embedded[str]],
                [
                    Embedded(
                        value=value,
                        vector=embedding,
                    )
                    for value, embedding in zip(
                        values,
                        chain.from_iterable(
                            [
                                cast(list[list[float]], response.embeddings.float_)
                                for response in responses
                            ]
                        ),
                        strict=True,
                    )
                ],
            )

    def image_embedding(self) -> ImageEmbedding:
        return ImageEmbedding(embedding=self.create_images_embedding)

    async def create_images_embedding[Value: DataModel | State](
        self,
        values: Sequence[Value] | Sequence[bytes],
        /,
        attribute: Callable[[Value], bytes] | None = None,
        *,
        config: CohereImageEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Value]] | Sequence[Embedded[bytes]]:
        embedding_config: CohereImageEmbeddingConfig = config or ctx.state(
            CohereImageEmbeddingConfig
        )
        async with ctx.scope("cohere_image_embedding"):
            ctx.record_info(
                attributes={
                    "embedding.model": embedding_config.model,
                    "embedding.batch_size": embedding_config.batch_size,
                },
            )
            attributes: list[bytes]
            if attribute is None:
                attributes = cast(list[bytes], as_list(values))

            else:
                attributes = [attribute(cast(Value, value)) for value in values]

            assert all(isinstance(element, bytes) for element in attributes)  # nosec: B101

            ctx.record_info(
                metric="embedding.items",
                value=len(attributes),
                unit="count",
                kind="counter",
                attributes={
                    "embedding.provider": "cohere",
                    "embedding.model": embedding_config.model,
                    "embedding.type": "image",
                },
            )

            if not attributes:
                return ()  # empty

            ctx.record_info(
                metric="embedding.batches",
                value=(len(attributes) + embedding_config.batch_size - 1)
                // embedding_config.batch_size,
                unit="count",
                kind="counter",
                attributes={
                    "embedding.provider": "cohere",
                    "embedding.model": embedding_config.model,
                    "embedding.type": "image",
                },
            )

            responses: list[EmbedByTypeResponse] = await gather(
                *[
                    self._client.embed(
                        model=embedding_config.model,
                        images=[
                            f"data:image/jpeg;base64,{urlsafe_b64encode(image).decode('utf-8')}"
                            for image in attributes[index : index + embedding_config.batch_size]
                        ],
                        embedding_types=["float"],
                        input_type="image",
                    )
                    for index in range(0, len(attributes), embedding_config.batch_size)
                ]
            )

            return cast(
                Sequence[Embedded[Value]] | Sequence[Embedded[bytes]],
                [
                    Embedded(
                        value=value,
                        vector=embedding,
                    )
                    for value, embedding in zip(
                        values,
                        chain.from_iterable(
                            [
                                cast(list[list[float]], response.embeddings.float_)
                                for response in responses
                            ]
                        ),
                        strict=True,
                    )
                ],
            )
