from asyncio import gather
from collections.abc import Sequence
from itertools import chain
from typing import Any

from haiway import as_list, ctx
from openai.types.create_embedding_response import CreateEmbeddingResponse

from draive.embedding import Embedded, TextEmbedding
from draive.vllm.api import VLLMAPI
from draive.vllm.config import VLLMEmbeddingConfig
from draive.vllm.utils import unwrap_missing

__all__ = [
    "VLLMEmbedding",
]


class VLLMEmbedding(VLLMAPI):
    def text_embedding(self) -> TextEmbedding:
        """
        Prepare TextEmbedding implementation using VLLM service.
        """
        return TextEmbedding(embed=self.create_text_embeddings)

    async def create_text_embeddings(
        self,
        values: Sequence[str],
        /,
        *,
        config: VLLMEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[str]]:
        """
        Create texts embedding with VLLM embedding service.
        """
        embedding_config: VLLMEmbeddingConfig = config or ctx.state(VLLMEmbeddingConfig).updated(
            **extra
        )
        with ctx.scope("vllm_text_embedding", embedding_config):
            ctx.record(embedding_config)
            texts: list[str] = as_list(values)
            responses: list[CreateEmbeddingResponse] = await gather(
                *[
                    self._client.embeddings.create(
                        input=texts[index : index + embedding_config.batch_size],
                        model=embedding_config.model,
                        dimensions=unwrap_missing(embedding_config.dimensions),
                        encoding_format="float",
                        timeout=unwrap_missing(embedding_config.timeout),
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
            ]
