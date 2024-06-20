import json
from asyncio import gather
from collections.abc import Sequence
from http import HTTPStatus
from typing import Any, Literal, Self, cast, final, overload

from httpx import AsyncClient, Response

from draive.gemini.config import GeminiConfig, GeminiEmbeddingConfig
from draive.gemini.errors import GeminiException
from draive.gemini.models import (
    GeminiFunctionsTool,
    GeminiGenerationRequest,
    GeminiGenerationResult,
    GeminiRequestMessage,
)
from draive.parameters import DataModel
from draive.scope import ScopeDependency
from draive.utils import getenv_str, not_missing

__all__ = [
    "GeminiClient",
]


@final
class GeminiClient(ScopeDependency):
    @classmethod
    def prepare(cls) -> Self:
        return cls(
            endpoint=getenv_str("GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com"),
            api_key=getenv_str("GEMINI_API_KEY"),
            timeout=90,
        )

    def __init__(
        self,
        endpoint: str,
        api_key: str | None,
        timeout: float | None = None,
    ) -> None:
        self._client: AsyncClient = AsyncClient(
            base_url=endpoint,
            params={
                "key": api_key,
            },
            timeout=timeout,
        )

    async def generate(  # noqa: PLR0913
        self,
        *,
        config: GeminiConfig,
        messages: list[GeminiRequestMessage],
        instruction: str | None = None,
        response_schema: dict[str, Any] | None = None,
        tools: list[GeminiFunctionsTool] | None = None,
        suggest_tools: bool = False,
        stream: bool = False,
    ) -> GeminiGenerationResult:
        if stream:
            raise NotImplementedError("Gemini streaming is not supported yet")

        else:
            return await self._generate_content(
                model=config.model,
                request={
                    "generationConfig": {
                        "responseMimeType": cast(
                            Literal["text/plain", "application/json"],
                            config.response_format,
                        )
                        if not_missing(config.response_format)
                        else "text/plain",
                        "responseSchema": response_schema,
                        "temperature": config.temperature,
                        "maxOutputTokens": config.max_tokens
                        if not_missing(config.max_tokens)
                        else None,
                        "topP": config.top_p if not_missing(config.top_p) else None,
                        "topK": config.top_k if not_missing(config.top_k) else None,
                        "candidateCount": 1,
                    },
                    "systemInstruction": {
                        "role": "",
                        "parts": ({"text": instruction},),
                    }
                    if instruction
                    else None,
                    "contents": messages,
                    "tools": tools or [],
                    "toolConfig": {
                        "functionCallingConfig": {
                            "mode": ("ANY" if suggest_tools else "AUTO") if tools else "NONE",
                        },
                    },
                },
            )

    async def _generate_content(
        self,
        model: str,
        request: GeminiGenerationRequest,
    ) -> GeminiGenerationResult:
        return await self._request(
            model=GeminiGenerationResult,
            method="POST",
            url=f"v1beta/models/{model}:generateContent",
            body=cast(dict[str, Any], request),  # there is some linter issue requiring to cast
        )

    async def embedding(
        self,
        config: GeminiEmbeddingConfig,
        inputs: Sequence[str],
    ) -> list[list[float]]:
        return list(
            await gather(
                *[
                    self._create_text_embedding(
                        model=config.model,
                        text=text,
                    )
                    for text in inputs
                ]
            )
        )

    async def _create_text_embedding(
        self,
        model: str,
        text: str,
    ) -> list[float]:
        response: dict[str, Any] = await self._request(
            model=None,
            method="POST",
            url=f"v1/models/{model}:embedContent",
            body={
                "content": {"parts": [{"text": text}]},
            },
        )

        match response:
            case {"embedding": {"values": [*vector]}}:
                return vector

            case _:
                raise GeminiException("Invalid Gemini embedding response: %s", response)

    async def dispose(self) -> None:
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
            raise GeminiException("Gemini request error: %s", error_body.decode("utf-8"))

        else:
            raise GeminiException("Network request failed: %s", response)
