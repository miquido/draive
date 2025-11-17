from asyncio import gather
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, cast

from haiway import State, as_list, ctx
from mistralai import EmbeddingResponse

from draive.embedding import Embedded, TextEmbedding
from draive.mistral.api import MistralAPI
from draive.mistral.config import MistralEmbeddingConfig
from draive.parameters import DataModel

__all__ = ("MistralEmbedding",)


class MistralEmbedding(MistralAPI):
    def text_embedding(self) -> TextEmbedding:
        return TextEmbedding(embedding=self.create_texts_embedding)

    async def create_texts_embedding[Value: DataModel | State](
        self,
        values: Sequence[Value] | Sequence[str],
        /,
        attribute: Callable[[Value], str] | None = None,
        *,
        config: MistralEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Value]] | Sequence[Embedded[str]]:
        embedding_config: MistralEmbeddingConfig = config or ctx.state(MistralEmbeddingConfig)
        async with ctx.scope("mistral_text_embedding"):
            ctx.record_info(
                attributes={
                    "embedding.provider": "mistral",
                    "embedding.model": embedding_config.model,
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
                    "embedding.provider": "mistral",
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
                    "embedding.provider": "mistral",
                    "embedding.model": embedding_config.model,
                    "embedding.type": "text",
                },
            )

            responses: list[EmbeddingResponse] = await gather(
                *[
                    self._client.embeddings.create_async(
                        model=embedding_config.model,
                        inputs=attributes[index : index + embedding_config.batch_size],
                    )
                    for index in range(0, len(attributes), embedding_config.batch_size)
                ]
            )

            return cast(
                Sequence[Embedded[Value]] | Sequence[Embedded[str]],
                [
                    Embedded(
                        value=value,
                        vector=embedding.embedding,
                    )
                    for value, embedding in zip(
                        values,
                        chain.from_iterable([response.data for response in responses]),
                        strict=True,
                    )
                    # filter out missing embeddings, although all should be available
                    if embedding.embedding
                ],
            )
