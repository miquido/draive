from asyncio import gather
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, cast

from haiway import State, as_list, ctx, unwrap_missing
from openai import omit
from openai.types.create_embedding_response import CreateEmbeddingResponse

from draive.embedding import Embedded
from draive.models.metrics import record_embedding_invocation, record_embedding_metrics
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIEmbeddingConfig

__all__ = ("OpenAIEmbedding",)


class OpenAIEmbedding(OpenAIAPI):
    async def create_texts_embedding[Value: State](
        self,
        values: Sequence[Value] | Sequence[str],
        /,
        attribute: Callable[[Value], str] | None = None,
        *,
        config: OpenAIEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Value]] | Sequence[Embedded[str]]:
        embedding_config: OpenAIEmbeddingConfig = config or ctx.state(OpenAIEmbeddingConfig)
        async with ctx.scope("openai.text_embedding"):
            attributes: list[str]
            if attribute is None:
                attributes = cast(list[str], as_list(values))

            else:
                attributes = [attribute(cast(Value, value)) for value in values]

            assert all(isinstance(element, str) for element in attributes)  # nosec: B101

            record_embedding_invocation(
                provider="openai",
                model=embedding_config.model,
                embedding_type="text",
                batch_size=embedding_config.batch_size,
                dimensions=embedding_config.dimensions,
            )
            record_embedding_metrics(
                provider="openai",
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

            responses: list[CreateEmbeddingResponse] = await gather(
                *[
                    self._client.embeddings.create(
                        input=attributes[index : index + embedding_config.batch_size],
                        model=embedding_config.model,
                        dimensions=unwrap_missing(
                            embedding_config.dimensions,
                            default=omit,
                        ),
                        encoding_format="float",
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
                ],
            )
