import json
from collections.abc import Sequence
from types import TracebackType
from typing import Any, Literal

from cohere import EmbedByTypeResponse
from haiway import asynchronous

__all__ = ("CohereBedrock",)


class CohereBedrock:
    """Minimal Cohere client for models served through AWS Bedrock.

    Bedrock exposes Cohere models through `bedrock-runtime.invoke_model`, where the
    request body is the regular Cohere API body with `model` moved into the request
    itself, and the response body is the regular Cohere API response. Only the
    `embed` endpoint is provided, which is all the Cohere integration uses.
    """

    __slots__ = (
        "_aws_region",
        "_client",
    )

    def __init__(
        self,
        *,
        aws_region: str | None = None,
    ) -> None:
        self._aws_region: str | None = aws_region
        self._client: Any  # lazily initialized

    # preparing it lazily on demand, boto does a lot of stuff on initialization
    @asynchronous
    def _initialize_client(self) -> None:
        if hasattr(self, "_client"):
            return  # already initialized

        # postponing import of boto3 as late as possible, it does a lot of stuff
        try:
            import boto3  # pyright: ignore[reportMissingTypeStubs]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "draive.cohere with the 'bedrock' provider requires the 'cohere_bedrock' extra."
                " Install via `pip install draive[cohere_bedrock]`."
            ) from exc

        self._client = boto3.Session(region_name=self._aws_region).client("bedrock-runtime")  # pyright: ignore[reportUnknownMemberType]

    @asynchronous
    def _deinitialize_client(self) -> None:
        if not hasattr(self, "_client"):
            return  # already deinitialized

        self._client.close()
        del self._client

    async def __aenter__(self) -> None:
        await self._initialize_client()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._deinitialize_client()

    async def embed(
        self,
        *,
        model: str,
        input_type: Literal[
            "search_document",
            "search_query",
            "classification",
            "clustering",
            "image",
        ],
        texts: Sequence[str] | None = None,
        images: Sequence[str] | None = None,
        embedding_types: Sequence[Literal["float", "int8", "uint8", "binary", "ubinary"]]
        | None = None,
        output_dimension: int | None = None,
        truncate: Literal["NONE", "START", "END"] | None = None,
    ) -> EmbedByTypeResponse:
        # the body matches the regular Cohere API request without `model`,
        # which Bedrock takes as a separate argument instead
        body: dict[str, Any] = {"input_type": input_type}
        if texts is not None:
            body["texts"] = list(texts)

        if images is not None:
            body["images"] = list(images)

        if embedding_types is not None:
            body["embedding_types"] = list(embedding_types)

        if output_dimension is not None:
            body["output_dimension"] = output_dimension

        if truncate is not None:
            body["truncate"] = truncate

        return EmbedByTypeResponse.model_validate(
            await self._invoke_model(
                model=model,
                body=body,
            )
        )

    @asynchronous
    def _invoke_model(
        self,
        *,
        model: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        response: Any = self._client.invoke_model(
            modelId=model,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body).encode("utf-8"),
        )

        payload: dict[str, Any] = json.loads(response["body"].read())
        # `id` is required by the Cohere response model but Bedrock does not always return it
        if "id" not in payload:
            payload["id"] = ""

        if (meta := _response_meta(response)) is not None:
            payload["meta"] = meta

        return payload


def _response_meta(
    response: Any,
    /,
) -> dict[str, Any] | None:
    headers: Any = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    input_tokens: str | None = headers.get("x-amzn-bedrock-input-token-count")
    output_tokens: str | None = headers.get("x-amzn-bedrock-output-token-count")
    if input_tokens is None and output_tokens is None:
        return None

    tokens: dict[str, Any] = {
        "input_tokens": int(input_tokens) if input_tokens is not None else None,
        "output_tokens": int(output_tokens) if output_tokens is not None else None,
    }

    return {
        "tokens": tokens,
        "billed_units": tokens,
    }
