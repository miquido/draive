import json
from asyncio import gather
from collections.abc import Sequence
from http import HTTPStatus
from itertools import chain
from types import TracebackType
from typing import Any, ClassVar, Literal, Self, final, overload

from haiway import getenv_str, not_missing
from httpx import AsyncClient, Response

from draive.gemini.config import GeminiConfig, GeminiEmbeddingConfig
from draive.gemini.models import (
    GeminiFunctionsTool,
    GeminiGenerationResult,
    GeminiRequestMessage,
)
from draive.gemini.types import GeminiException
from draive.parameters import DataModel

__all__ = [
    "GeminiClient",
]


@final
class GeminiClient:
    _SHARED: ClassVar[Self]

    @classmethod
    def shared(cls) -> Self:
        if shared := getattr(cls, "_SHARED", None):
            return shared

        else:
            cls._SHARED = cls()  # pyright: ignore[reportConstantRedefinition]
            return cls._SHARED

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._endpoint: str = endpoint or getenv_str(
            "GEMINI_ENDPOINT",
            default="https://generativelanguage.googleapis.com",
        )
        self._api_key: str | None = api_key or getenv_str("GEMINI_API_KEY")
        self._timeout: float = timeout or 60
        self._client: AsyncClient = self._initialize_client()

    def _initialize_client(self) -> AsyncClient:
        return AsyncClient(  # nosec: B113 - false positive
            base_url=self._endpoint,
            params={
                "key": self._api_key,
            },
            timeout=self._timeout,
        )

    async def generate(  # noqa: PLR0913
        self,
        *,
        config: GeminiConfig,
        instruction: str,
        messages: list[GeminiRequestMessage],
        tools: list[GeminiFunctionsTool] | None = None,
        tool_calling_mode: Literal["AUTO", "ANY", "NONE"] = "AUTO",
        response_schema: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> GeminiGenerationResult:
        if stream:
            raise NotImplementedError("Gemini streaming is not supported yet")

        else:
            return await self._generate_content(
                model=config.model,
                request={
                    "generationConfig": {
                        "responseMimeType": config.response_format
                        if not_missing(config.response_format)
                        else "text/plain",
                        "temperature": config.temperature,
                        "topP": config.top_p if not_missing(config.top_p) else None,
                        "topK": config.top_k if not_missing(config.top_k) else None,
                        "maxOutputTokens": config.max_tokens
                        if not_missing(config.max_tokens)
                        else None,
                        "responseSchema": response_schema if response_schema else None,
                        "candidateCount": 1,
                        "stopSequences": config.stop_sequences
                        if not_missing(config.stop_sequences)
                        else None,
                    },
                    "systemInstruction": {
                        "parts": ({"text": instruction},),
                    },
                    "contents": messages,
                    "tools": tools or [],
                    "toolConfig": {
                        "functionCallingConfig": {
                            "mode": tool_calling_mode if tools else "NONE",
                        },
                    },
                    "safetySettings": [  # google moderation is terrible, disabling it all
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE",
                        },
                    ],
                },
            )

    async def _generate_content(
        self,
        model: str,
        request: dict[str, Any],
    ) -> GeminiGenerationResult:
        return await self._request(
            model=GeminiGenerationResult,
            method="POST",
            url=f"v1beta/models/{model}:generateContent",
            body=request,
        )

    async def embedding(
        self,
        config: GeminiEmbeddingConfig,
        inputs: Sequence[str],
    ) -> list[list[float]]:
        return list(
            chain(
                *await gather(
                    *[
                        self._create_text_embedding(
                            model=config.model,
                            texts=inputs[index : index + config.batch_size],
                        )
                        for index in range(0, len(inputs), config.batch_size)
                    ]
                )
            )
        )

    async def _create_text_embedding(
        self,
        model: str,
        texts: Sequence[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        response: dict[str, Any] = await self._request(
            model=None,
            method="POST",
            url=f"v1beta/models/{model}:batchEmbedText",
            body={"texts": texts},
        )

        result: list[list[float]] = []
        match response:
            case {"embeddings": [*embeddings]}:
                for embedding in embeddings:
                    match embedding:
                        case {"value": [*vector]}:
                            assert all(isinstance(value, float) for value in vector)  # nosec: B101
                            result.append(vector)

                        case _:
                            raise GeminiException("Invalid Gemini embedding response: %s", response)

            case _:
                raise GeminiException("Invalid Gemini embedding response: %s", response)

        assert len(texts) == len(result)  # nosec: B101
        return result

    async def __aenter__(self) -> None:
        if self._client.is_closed:
            self._client = self._initialize_client()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    @overload
    async def _request[Requested: DataModel](
        self,
        model: None,
        method: str,
        url: str,
        query: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        body: DataModel | dict[str, Any] | None = None,
        follow_redirects: bool | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]: ...

    @overload
    async def _request[Requested: DataModel](
        self,
        model: type[Requested],
        method: str,
        url: str,
        query: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        body: DataModel | dict[str, Any] | None = None,
        follow_redirects: bool | None = None,
        timeout: float | None = None,
    ) -> Requested: ...

    async def _request[Requested: DataModel](  # noqa: PLR0913, PLR0912
        self,
        model: type[Requested] | None,
        method: str,
        url: str,
        query: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        body: DataModel | dict[str, Any] | None = None,
        follow_redirects: bool | None = None,
        timeout: float | None = None,
    ) -> Requested | dict[str, Any]:
        request_headers: dict[str, str]
        if headers:
            request_headers = headers

        else:
            request_headers = {
                "Accept": "application/json",
            }

        body_content: str | None
        match body:
            case None:
                body_content = None

            case body_model if isinstance(body_model, DataModel):
                body_content = body_model.as_json()

            case values:
                body_content = json.dumps(values)

        if body_content:
            request_headers["Content-Type"] = "application/json"

        response: Response
        try:
            response = await self._client.request(
                method=method,
                url=url,
                headers=request_headers,
                params=query,
                content=body_content,
                follow_redirects=follow_redirects or False,
                timeout=timeout,
            )

        except Exception as exc:
            raise GeminiException("Network request failed") from exc

        status: HTTPStatus = HTTPStatus(value=response.status_code)
        if status.is_success:
            try:
                if model := model:
                    return model.from_json(await response.aread())

                else:
                    return json.loads(await response.aread())

            except Exception as exc:
                raise GeminiException("Failed to decode Gemini response", response) from exc

        elif status.is_client_error:
            error_body: bytes = await response.aread()
            raise GeminiException(
                "Gemini request error: %s, %s",
                status,
                error_body.decode("utf-8"),
            )

        else:
            raise GeminiException("Network request failed: %s", response)
