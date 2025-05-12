from asyncio import gather
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, cast

from google.genai.types import EmbedContentConfigDict, EmbedContentResponse
from haiway import ObservabilityLevel, State, as_list, ctx, not_missing

from draive.embedding import Embedded, TextEmbedding
from draive.gemini.api import GeminiAPI
from draive.gemini.config import GeminiEmbeddingConfig
from draive.parameters import DataModel

__all__ = ("GeminiEmbedding",)


class GeminiEmbedding(GeminiAPI):
    def text_embedding(self) -> TextEmbedding:
        """
        Prepare TextEmbedding implementation using Gemini services.
        """
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
        """
        Create texts embedding with Gemini embedding service.
        """

        embedding_config: GeminiEmbeddingConfig = config or ctx.state(GeminiEmbeddingConfig)
        config_dict: EmbedContentConfigDict | None
        if not_missing(embedding_config.dimensions):
            config_dict = {"output_dimensionality": embedding_config.dimensions}

        else:
            config_dict = None

        with ctx.scope("gemini_text_embedding", embedding_config):
            ctx.record(
                ObservabilityLevel.INFO,
                attributes={
                    "embedding.provider": "gemini",
                    "embedding.model": embedding_config.model,
                    "embedding.dimensions": embedding_config.dimensions,
                },
            )
            attributes: list[str]
            if attribute is None:
                attributes = cast(list[str], as_list(values))

            else:
                attributes = [attribute(cast(Value, value)) for value in values]

            assert all(isinstance(element, str) for element in attributes)  # nosec: B101

            responses: list[EmbedContentResponse] = await gather(
                *[
                    self._client.aio.models.embed_content(
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
