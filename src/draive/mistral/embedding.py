from asyncio import gather
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, cast

from haiway import State, as_list, ctx
from mistralai import EmbeddingResponse

from draive.embedding import Embedded
from draive.mistral.api import MistralAPI
from draive.mistral.config import MistralEmbeddingConfig
from draive.models.metrics import record_embedding_invocation, record_embedding_metrics

__all__ = ("MistralEmbedding",)


class MistralEmbedding(MistralAPI):
    async def create_texts_embedding[Value: State](
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
            attributes: list[str]
            if attribute is None:
                attributes = cast(list[str], as_list(values))

            else:
                attributes = [attribute(cast(Value, value)) for value in values]

            assert all(isinstance(element, str) for element in attributes)  # nosec: B101

            record_embedding_invocation(
                provider="mistral",
                model=embedding_config.model,
                embedding_type="text",
                batch_size=embedding_config.batch_size,
            )
            record_embedding_metrics(
                provider="mistral",
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
