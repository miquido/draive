from asyncio import gather
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, cast

from haiway import State, as_list, ctx
from ollama import EmbedResponse

from draive.embedding import Embedded
from draive.models.metrics import record_embedding_invocation, record_embedding_metrics
from draive.ollama.api import OllamaAPI
from draive.ollama.config import OllamaEmbeddingConfig

__all__ = ("OllamaEmbedding",)


class OllamaEmbedding(OllamaAPI):
    async def create_texts_embedding[Value: State](
        self,
        values: Sequence[Value] | Sequence[str],
        /,
        attribute: Callable[[Value], str] | None = None,
        *,
        config: OllamaEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Value]] | Sequence[Embedded[str]]:
        embedding_config: OllamaEmbeddingConfig = config or ctx.state(OllamaEmbeddingConfig)
        async with ctx.scope("ollama.text_embedding"):
            attributes: list[str]
            if attribute is None:
                attributes = cast(list[str], as_list(values))

            else:
                attributes = [attribute(cast(Value, value)) for value in values]

            assert all(isinstance(element, str) for element in attributes)  # nosec: B101

            record_embedding_invocation(
                provider="ollama",
                model=embedding_config.model,
                embedding_type="text",
                batch_size=embedding_config.batch_size,
                concurrent=embedding_config.concurrent,
            )
            record_embedding_metrics(
                provider="ollama",
                model=embedding_config.model,
                embedding_type="text",
                items=len(attributes),
                batches=(
                    (len(attributes) + embedding_config.batch_size - 1)
                    // embedding_config.batch_size
                    if attributes
                    else 0
                ),
            )

            if not attributes:
                return ()  # empty

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
