from asyncio import gather
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, cast

from haiway import ObservabilityLevel, State, as_list, ctx
from ollama import EmbedResponse

from draive.embedding import Embedded, TextEmbedding
from draive.ollama.api import OllamaAPI
from draive.ollama.config import OllamaEmbeddingConfig
from draive.parameters import DataModel

__all__ = ("OllamaEmbedding",)


class OllamaEmbedding(OllamaAPI):
    def text_embedding(self) -> TextEmbedding:
        """
        Prepare TextEmbedding implementation using Ollama service.
        """
        return TextEmbedding(embedding=self.create_texts_embedding)

    async def create_texts_embedding[Value: DataModel | State](
        self,
        values: Sequence[Value] | Sequence[str],
        /,
        attribute: Callable[[Value], str] | None = None,
        *,
        config: OllamaEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Value]] | Sequence[Embedded[str]]:
        """
        Create texts embedding with Ollama embedding service.
        """
        embedding_config: OllamaEmbeddingConfig = config or ctx.state(OllamaEmbeddingConfig)
        with ctx.scope("ollama_text_embedding", embedding_config):
            ctx.record(
                ObservabilityLevel.INFO,
                attributes={
                    "embedding.provider": "ollama",
                    "embedding.model": embedding_config.model,
                },
            )
            ctx.record(attributes=embedding_config.to_mapping())
            attributes: list[str]
            if attribute is None:
                attributes = cast(list[str], as_list(values))

            else:
                attributes = [attribute(cast(Value, value)) for value in values]

            assert all(isinstance(element, str) for element in attributes)  # nosec: B101

            responses: list[EmbedResponse]
            if embedding_config.concurrent:
                responses = await gather(
                    *[
                        self._client.embed(
                            model=embedding_config.model,
                            input=attributes[index : index + embedding_config.batch_size],
                        )
                        for index in range(0, len(attributes), embedding_config.batch_size)
                    ]
                )

            else:
                responses = []
                for index in range(0, len(attributes), embedding_config.batch_size):
                    responses.append(
                        await self._client.embed(
                            model=embedding_config.model,
                            input=attributes[index : index + embedding_config.batch_size],
                        )
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
                        chain.from_iterable([response.embeddings for response in responses]),
                        strict=True,
                    )
                ],
            )
