from asyncio import gather
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, cast

from google.genai.types import EmbedContentConfigDict, EmbedContentResponse
from haiway import State, as_list, ctx, not_missing

from draive.embedding import Embedded, TextEmbedding
from draive.gemini.api import GeminiAPI
from draive.gemini.config import GeminiEmbeddingConfig
from draive.parameters import DataModel

__all__ = ("GeminiEmbedding",)


class GeminiEmbedding(GeminiAPI):
    def text_embedding(self) -> TextEmbedding:
        return TextEmbedding(embedding=self.create_texts_embedding)

    async def create_texts_embedding[Value: DataModel | State](
        self,
        values: Sequence[Value] | Sequence[str],
        /,
        attribute: Callable[[Value], str] | None = None,
        *,
        config: GeminiEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Value]] | Sequence[Embedded[str]]:
        embedding_config: GeminiEmbeddingConfig = config or ctx.state(GeminiEmbeddingConfig)

        async with ctx.scope("gemini_text_embedding"):
            ctx.record_info(
                attributes={
                    "embedding.provider": "gemini",
                    "embedding.model": embedding_config.model,
                    "embedding.dimensions": embedding_config.dimensions,
                    "embedding.batch_size": embedding_config.batch_size,
                },
            )
            attributes: list[str]
            if attribute is None:
                attributes = cast(list[str], as_list(values))

            else:
                attributes = [attribute(cast(Value, value)) for value in values]

            assert all(isinstance(element, str) for element in attributes)  # nosec: B101

            ctx.record_info(
                metric="embedding.items",
                value=len(attributes),
                unit="count",
                kind="counter",
                attributes={
                    "embedding.provider": "gemini",
                    "embedding.model": embedding_config.model,
                    "embedding.type": "text",
                },
            )

            if not attributes:
                return ()  # empty

            ctx.record_info(
                metric="embedding.batches",
                value=(len(attributes) + embedding_config.batch_size - 1)
                // embedding_config.batch_size,
                unit="count",
                kind="counter",
                attributes={
                    "embedding.provider": "gemini",
                    "embedding.model": embedding_config.model,
                    "embedding.type": "text",
                },
            )

            config_dict: EmbedContentConfigDict | None
            if not_missing(embedding_config.dimensions):
                config_dict = {"output_dimensionality": embedding_config.dimensions}

            else:
                config_dict = None

            responses: list[EmbedContentResponse] = await gather(
                *[
                    self._client.aio.models.embed_content(  # pyright: ignore[reportUnknownMemberType]
                        model=embedding_config.model,
                        config=config_dict,
                        contents=as_list(
                            attributes[index : index + embedding_config.batch_size],
                        ),
                    )
                    for index in range(0, len(attributes), embedding_config.batch_size)
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
