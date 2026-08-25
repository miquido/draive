from asyncio import gather
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, cast

from haiway import State, as_list, ctx
from openai.types.create_embedding_response import CreateEmbeddingResponse

from draive.embedding import Embedded
from draive.models.metrics import record_embedding_invocation, record_embedding_metrics
from draive.vllm.api import VLLMAPI
from draive.vllm.config import VLLMEmbeddingConfig
from draive.vllm.utils import unwrap_missing

__all__ = ("VLLMEmbedding",)


class VLLMEmbedding(VLLMAPI):
    async def create_texts_embedding[Value: State](
        self,
        values: Sequence[Value] | Sequence[str],
        /,
        attribute: Callable[[Value], str] | None = None,
        *,
        config: VLLMEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Value]] | Sequence[Embedded[str]]:
        async with ctx.scope("embedding.invocation"):
            embedding_config: VLLMEmbeddingConfig = config or ctx.state(VLLMEmbeddingConfig)

            record_embedding_invocation(
                provider=self._provider,
                model=embedding_config.model,
                embedding_type="text",
                batch_size=embedding_config.batch_size,
                dimensions=embedding_config.dimensions,
            )
            record_embedding_metrics(
                provider=self._provider,
                model=embedding_config.model,
                embedding_type="text",
                items=len(values),
                batches=(
                    (len(values) + embedding_config.batch_size - 1) // embedding_config.batch_size
                    if values
                    else 0
                ),
            )

            if not values:
                return ()  # empty

            responses: list[CreateEmbeddingResponse]
            if attribute is None:
                responses = await gather(
                    *[
                        self._client.embeddings.create(
                            input=as_list(
                                cast(Sequence[str], values)[
                                    index : index + embedding_config.batch_size
                                ]
                            ),
                            model=embedding_config.model,
                            dimensions=unwrap_missing(embedding_config.dimensions),
                            encoding_format="float",
                        )
                        for index in range(0, len(values), embedding_config.batch_size)
                    ]
                )

            else:
                responses = await gather(
                    *[
                        self._client.embeddings.create(
                            input=as_list(
                                attribute(cast(Value, value))
                                for value in values[index : index + embedding_config.batch_size]
                            ),
                            model=embedding_config.model,
                            dimensions=unwrap_missing(embedding_config.dimensions),
                            encoding_format="float",
                        )
                        for index in range(0, len(values), embedding_config.batch_size)
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
                        # the api documents no ordering guarantee, the index within each
                        # entry is what pairs it back with its input
                        chain.from_iterable(
                            sorted(response.data, key=lambda entry: entry.index)
                            for response in responses
                        ),
                        strict=True,
                    )
                ],
            )
