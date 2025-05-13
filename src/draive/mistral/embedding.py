from asyncio import gather
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, cast

from haiway import ObservabilityLevel, State, as_list, ctx
from mistralai import EmbeddingResponse

from draive.embedding import Embedded, TextEmbedding
from draive.mistral.api import MistralAPI
from draive.mistral.config import MistralEmbeddingConfig
from draive.parameters import DataModel

__all__ = ("MistralEmbedding",)


class MistralEmbedding(MistralAPI):
    def text_embedding(self) -> TextEmbedding:
        """
        Prepare TextEmbedding implementation using Mistral service.
        """
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
        """
        Create texts embedding with Mistral embedding service.
        """

        embedding_config: MistralEmbeddingConfig = config or ctx.state(MistralEmbeddingConfig)
        with ctx.scope("mistral_text_embedding", embedding_config):
            ctx.record(
                ObservabilityLevel.INFO,
                attributes={
                    "embedding.provider": "mistral",
                    "embedding.model": embedding_config.model,
                },
            )
            attributes: list[str]
            if attribute is None:
                attributes = cast(list[str], as_list(values))

            else:
                attributes = [attribute(cast(Value, value)) for value in values]

            assert all(isinstance(element, str) for element in attributes)  # nosec: B101

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
