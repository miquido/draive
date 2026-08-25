from asyncio import gather
from base64 import b64encode
from collections.abc import Callable, Sequence
from typing import Any, cast

from cohere import EmbedByTypeResponse
from haiway import State, as_list, ctx

from draive.cohere.api import CohereAPI
from draive.cohere.config import CohereImageEmbeddingConfig, CohereTextEmbeddingConfig
from draive.embedding import Embedded, ImageEmbedding, TextEmbedding
from draive.models.metrics import record_embedding_invocation, record_embedding_metrics

__all__ = ("CohereEmbedding",)


class CohereEmbedding(CohereAPI):
    def text_embedding(self) -> TextEmbedding:
        return TextEmbedding(embedding=self.create_texts_embedding)

    async def create_texts_embedding[Value: State](
        self,
        values: Sequence[Value] | Sequence[str],
        /,
        attribute: Callable[[Value], str] | None = None,
        *,
        config: CohereTextEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Value]] | Sequence[Embedded[str]]:
        embedding_config: CohereTextEmbeddingConfig = config or ctx.state(CohereTextEmbeddingConfig)
        async with ctx.scope("embedding.invocation"):
            attributes: list[str]
            if attribute is None:
                attributes = cast(list[str], as_list(values))

            else:
                attributes = [attribute(cast(Value, value)) for value in values]

            assert all(isinstance(element, str) for element in attributes)  # nosec: B101

            record_embedding_invocation(
                provider="cohere",
                model=embedding_config.model,
                embedding_type="text",
                batch_size=embedding_config.batch_size,
                purpose=embedding_config.purpose,
            )
            record_embedding_metrics(
                provider="cohere",
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

            responses: list[EmbedByTypeResponse] = await gather(
                *[
                    self._client.embed(
                        model=embedding_config.model,
                        texts=attributes[index : index + embedding_config.batch_size],
                        embedding_types=["float"],
                        input_type=embedding_config.purpose,
                    )
                    for index in range(0, len(attributes), embedding_config.batch_size)
                ]
            )

            return cast(
                Sequence[Embedded[Value]] | Sequence[Embedded[str]],
                [
                    Embedded(
                        value=value,
                        vector=vector,
                    )
                    for value, vector in zip(
                        values,
                        _response_vectors(
                            responses,
                            model=embedding_config.model,
                        ),
                        strict=True,
                    )
                ],
            )

    def image_embedding(self) -> ImageEmbedding:
        return ImageEmbedding(embedding=self.create_images_embedding)

    async def create_images_embedding[Value: State](
        self,
        values: Sequence[Value] | Sequence[bytes],
        /,
        attribute: Callable[[Value], bytes] | None = None,
        *,
        config: CohereImageEmbeddingConfig | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Value]] | Sequence[Embedded[bytes]]:
        embedding_config: CohereImageEmbeddingConfig = config or ctx.state(
            CohereImageEmbeddingConfig
        )
        async with ctx.scope("embedding.invocation"):
            attributes: list[bytes]
            if attribute is None:
                attributes = cast(list[bytes], as_list(values))

            else:
                attributes = [attribute(cast(Value, value)) for value in values]

            assert all(isinstance(element, bytes) for element in attributes)  # nosec: B101

            record_embedding_invocation(
                provider="cohere",
                model=embedding_config.model,
                embedding_type="image",
                batch_size=embedding_config.batch_size,
            )
            record_embedding_metrics(
                provider="cohere",
                model=embedding_config.model,
                embedding_type="image",
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

            responses: list[EmbedByTypeResponse] = await gather(
                *[
                    self._client.embed(
                        model=embedding_config.model,
                        images=[
                            f"data:{_image_mime_type(image)};base64,"
                            f"{b64encode(image).decode('utf-8')}"
                            for image in attributes[index : index + embedding_config.batch_size]
                        ],
                        embedding_types=["float"],
                        input_type="image",
                    )
                    for index in range(0, len(attributes), embedding_config.batch_size)
                ]
            )

            return cast(
                Sequence[Embedded[Value]] | Sequence[Embedded[bytes]],
                [
                    Embedded(
                        value=value,
                        vector=vector,
                    )
                    for value, vector in zip(
                        values,
                        _response_vectors(
                            responses,
                            model=embedding_config.model,
                        ),
                        strict=True,
                    )
                ],
            )


def _response_vectors(
    responses: Sequence[EmbedByTypeResponse],
    /,
    *,
    model: str,
) -> list[Sequence[float]]:
    # the api reports no index, its results are documented to correspond to the
    # request order - a response without the requested type has no position to
    # recover, refusing it beats silently misaligning every following value
    vectors: list[Sequence[float]] = []
    for response in responses:
        if not response.embeddings.float_:
            raise ValueError(
                f"Cohere embedding using {model} returned no float vectors,"
                " the model may not support the requested embedding type"
            )

        vectors.extend(response.embeddings.float_)

    return vectors


def _image_mime_type(
    data: bytes,
    /,
) -> str:
    # data uri requires a mime type while only raw image data is available,
    # it has to be recognized by the image header
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"

    elif data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"

    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"

    else:
        return "image/jpeg"  # jpeg is the default for anything unrecognized
