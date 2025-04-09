from asyncio import gather
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, cast

from haiway import State, as_list, ctx
from openai.types.create_embedding_response import CreateEmbeddingResponse

from draive.embedding import Embedded, TextEmbedding
from draive.parameters import DataModel
from draive.vllm.api import VLLMAPI
from draive.vllm.config import VLLMEmbeddingConfig
from draive.vllm.utils import unwrap_missing

__all__ = ("VLLMEmbedding",)


class VLLMEmbedding(VLLMAPI):
    def text_embedding(self) -> TextEmbedding:
        """
        Prepare TextEmbedding implementation using VLLM service.
        """
        return TextEmbedding(embedding=self.create_texts_embedding)

    async def create_texts_embedding[Value: DataModel | State](
        self,
        values: Sequence[Value] | Sequence[str],
        /,
        attribute: Callable[[Value], str] | None = None,
        *,
        config: VLLMEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Value]] | Sequence[Embedded[str]]:
        """
        Create texts embedding with VLLM embedding service.
        """
        embedding_config: VLLMEmbeddingConfig = config or ctx.state(VLLMEmbeddingConfig).updated(
            **extra
        )
        with ctx.scope("vllm_text_embedding", embedding_config):
            ctx.record(embedding_config)
            attributes: list[str]
            if attribute is None:
                attributes = cast(list[str], as_list(values))

            else:
                attributes = [attribute(cast(Value, value)) for value in values]

            assert all(isinstance(element, str) for element in attributes)  # nosec: B101

            responses: list[CreateEmbeddingResponse] = await gather(
                *[
                    self._client.embeddings.create(
                        input=attributes[index : index + embedding_config.batch_size],
                        model=embedding_config.model,
                        dimensions=unwrap_missing(embedding_config.dimensions),
                        encoding_format="float",
                        timeout=unwrap_missing(embedding_config.timeout),
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
