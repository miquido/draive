from asyncio import gather
from collections.abc import Callable, Sequence
from typing import Any, cast

from haiway import State, as_list, ctx
from mistralai.client.models import EmbeddingResponse

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
        async with ctx.scope("embedding.invocation"):
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

            # the api documents no ordering guarantee, the index within each entry
            # is what pairs it back with its input
            vectors: list[Sequence[float]] = []
            for response in responses:
                for entry in sorted(response.data, key=lambda entry: entry.index or 0):
                    if not entry.embedding:
                        # dropping it would silently shorten the result and misalign
                        # every value following it
                        raise ValueError(
                            f"Mistral embedding using {embedding_config.model} returned"
                            f" no vector for input {len(vectors)}"
                        )

                    vectors.append(entry.embedding)

            return cast(
                Sequence[Embedded[Value]] | Sequence[Embedded[str]],
                [
                    Embedded(
                        value=value,
                        vector=vector,
                    )
                    for value, vector in zip(
                        values,
                        vectors,
                        strict=True,
                    )
                ],
            )
