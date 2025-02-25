from asyncio import gather
from collections.abc import Sequence
from itertools import chain
from typing import Any, cast

from google.genai.types import ContentUnion, EmbedContentConfigDict, EmbedContentResponse
from haiway import as_list, ctx, not_missing

from draive.embedding import Embedded, TextEmbedding
from draive.gemini.api import GeminiAPI
from draive.gemini.config import GeminiEmbeddingConfig

__all__ = [
    "GeminiEmbedding",
]


class GeminiEmbedding(GeminiAPI):
    def text_embedding(self) -> TextEmbedding:
        """
        Prepare TextEmbedding implementation using Gemini services.
        """
        return TextEmbedding(embed=self.create_text_embeddings)

    async def create_text_embeddings(
        self,
        values: Sequence[str],
        /,
        *,
        config: GeminiEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[str]]:
        """
        Create texts embedding with Gemini embedding service.
        """
        embedding_config: GeminiEmbeddingConfig = config or ctx.state(
            GeminiEmbeddingConfig
        ).updated(**extra)
        config_dict: EmbedContentConfigDict | None
        if not_missing(embedding_config.dimensions):
            config_dict = {"output_dimensionality": embedding_config.dimensions}

        else:
            config_dict = None

        with ctx.scope("gemini_text_embedding", embedding_config):
            ctx.record(embedding_config)
            texts: list[str] = as_list(values)
            responses: list[EmbedContentResponse] = await gather(
                *[
                    self._client.aio.models.embed_content(
                        model=embedding_config.model,
                        config=config_dict,
                        contents=cast(
                            list[ContentUnion],  # it is actually list[str]
                            texts[index : index + embedding_config.batch_size],
                        ),
                    )
                    for index in range(0, len(texts), embedding_config.batch_size)
                ]
            )

            return [
                Embedded(
                    value=embedded[0],
                    vector=embedded[1].values,
                )
                for embedded in zip(
                    texts,
                    chain.from_iterable(
                        # filter out missing embeddings, although all should be available
                        [response.embeddings for response in responses if response.embeddings]
                    ),
                    strict=True,
                )
                # filter out missing embeddings, although all should be available
                if embedded[1].values
            ]
