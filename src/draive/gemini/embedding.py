from asyncio import gather
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, cast

from google.genai.types import EmbedContentConfigDict, EmbedContentResponse
from haiway import State, as_list, ctx, not_missing

from draive.embedding import Embedded
from draive.gemini.api import GeminiAPI
from draive.gemini.config import GeminiEmbeddingConfig
from draive.models.metrics import record_embedding_invocation, record_embedding_metrics

__all__ = ("GeminiEmbedding",)


class GeminiEmbedding(GeminiAPI):
    async def create_texts_embedding[Value: State](
        self,
        values: Sequence[Value] | Sequence[str],
        /,
        attribute: Callable[[Value], str] | None = None,
        *,
        config: GeminiEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Value]] | Sequence[Embedded[str]]:
        async with ctx.scope("gemini.text_embedding"):
            config = config or ctx.state(GeminiEmbeddingConfig)
            attributes: list[str]
            if attribute is None:
                attributes = cast(list[str], as_list(values))

            else:
                attributes = [attribute(cast(Value, value)) for value in values]

            assert all(isinstance(element, str) for element in attributes)  # nosec: B101

            record_embedding_invocation(
                provider="gemini",
                model=config.model,
                embedding_type="text",
                batch_size=config.batch_size,
                dimensions=config.dimensions,
            )
            record_embedding_metrics(
                provider="gemini",
                model=config.model,
                embedding_type="text",
                items=len(attributes),
                batches=(
                    (len(attributes) + config.batch_size - 1) // config.batch_size
                    if attributes
                    else 0
                ),
            )

            if not attributes:
                return ()  # empty

            config_dict: EmbedContentConfigDict | None
            if not_missing(config.dimensions):
                config_dict = {"output_dimensionality": config.dimensions}

            else:
                config_dict = None

            responses: list[EmbedContentResponse] = await gather(
                *[
                    self._client.aio.models.embed_content(  # pyright: ignore[reportUnknownMemberType]
                        model=config.model,
                        config=config_dict,
                        contents=as_list(
                            attributes[index : index + config.batch_size],
                        ),
                    )
                    for index in range(0, len(attributes), config.batch_size)
                ]
            )

            return cast(
                Sequence[Embedded[Value]] | Sequence[Embedded[str]],
                [
                    Embedded(
                        value=value,
                        vector=embedding.values,
                    )
                    for value, embedding in zip(
                        values,
                        chain.from_iterable(
                            # filter out missing embeddings, although all should be available
                            [response.embeddings for response in responses if response.embeddings]
                        ),
                        strict=True,
                    )
                    # filter out missing embeddings, although all should be available
                    if embedding.values
                ],
            )
