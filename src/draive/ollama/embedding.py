from asyncio import gather
from collections.abc import Sequence
from itertools import chain
from typing import Any

from haiway import as_list, ctx
from ollama import EmbedResponse

from draive.embedding import Embedded, TextEmbedding
from draive.ollama.api import OllamaAPI
from draive.ollama.config import OllamaEmbeddingConfig

__all__ = [
    "OllamaEmbedding",
]


class OllamaEmbedding(OllamaAPI):
    def text_embedding(self) -> TextEmbedding:
        """
        Prepare TextEmbedding implementation using Ollama service.
        """
        return TextEmbedding(embed=self.create_text_embeddings)

    async def create_text_embeddings(
        self,
        values: Sequence[str],
        /,
        *,
        config: OllamaEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[str]]:
        """
        Create texts embedding with Ollama embedding service.
        """
        embedding_config: OllamaEmbeddingConfig = config or ctx.state(
            OllamaEmbeddingConfig
        ).updated(**extra)
        with ctx.scope("ollama_text_embedding", embedding_config):
            ctx.record(embedding_config)
            texts: list[str] = as_list(values)
            responses: list[EmbedResponse]
            if embedding_config.concurrent:
                responses = await gather(
                    *[
                        self._client.embed(
                            model=embedding_config.model,
                            input=texts[index : index + embedding_config.batch_size],
                        )
                        for index in range(0, len(texts), embedding_config.batch_size)
                    ]
                )

            else:
                responses = []
                for index in range(0, len(texts), embedding_config.batch_size):
                    responses.append(
                        await self._client.embed(
                            model=embedding_config.model,
                            input=texts[index : index + embedding_config.batch_size],
                        )
                    )

            return [
                Embedded(
                    value=embedded[0],
                    vector=embedded[1],
                )
                for embedded in zip(
                    texts,
                    chain.from_iterable([response.embeddings for response in responses]),
                    strict=True,
                )
            ]
