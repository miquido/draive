from asyncio import gather
from collections.abc import Sequence
from itertools import chain
from typing import Any

from haiway import as_list, ctx
from mistralai import EmbeddingResponse

from draive.embedding import Embedded, TextEmbedding
from draive.mistral.api import MistralAPI
from draive.mistral.config import MistralEmbeddingConfig

__all__ = [
    "MistralEmbedding",
]


class MistralEmbedding(MistralAPI):
    def text_embedding(self) -> TextEmbedding:
        """
        Prepare TextEmbedding implementation using Mistral service.
        """
        return TextEmbedding(embed=self.create_text_embeddings)

    async def create_text_embeddings(
        self,
        values: Sequence[str],
        /,
        *,
        config: MistralEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[str]]:
        """
        Create texts embedding with Mistral embedding service.
        """
        embedding_config: MistralEmbeddingConfig = config or ctx.state(
            MistralEmbeddingConfig
        ).updated(**extra)
        with ctx.scope("mistral_text_embedding", embedding_config):
            ctx.record(embedding_config)
            texts: list[str] = as_list(values)
            responses: list[EmbeddingResponse] = await gather(
                *[
                    self._client.embeddings.create_async(
                        model=embedding_config.model,
                        inputs=texts[index : index + embedding_config.batch_size],
                    )
                    for index in range(0, len(texts), embedding_config.batch_size)
                ]
            )

            return [
                Embedded(
                    value=embedded[0],
                    vector=embedded[1].embedding,
                )
                for embedded in zip(
                    texts,
                    chain.from_iterable([response.data for response in responses]),
                    strict=True,
                )
                # filter out missing embeddings, although all should be available
                if embedded[1].embedding
            ]
