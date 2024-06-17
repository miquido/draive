import json
from http import HTTPStatus
from typing import Any, Literal, Self, cast, final

from httpx import AsyncClient, Response

from draive.gemini.config import GeminiConfig
from draive.gemini.errors import GeminiException
from draive.gemini.models import (
    GeminiFunctionsTool,
    GeminiGenerationConfig,
    GeminiGenerationRequest,
    GeminiGenerationResult,
    GeminiMessage,
    GeminiTextMessageContent,
    GeminiToolConfig,
    GeminiToolFunctionCallingConfig,
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
        messages: list[GeminiMessage],
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
                request=GeminiGenerationRequest(
                    config=GeminiGenerationConfig(
                        response_format=cast(
                            Literal["text/plain", "application/json"],
                            config.response_format,
                        )
                        if not_missing(config.response_format)
                        else "text/plain",
                        response_schema=response_schema,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens if not_missing(config.max_tokens) else None,
                        top_p=config.top_p if not_missing(config.top_p) else None,
                        top_k=config.top_k if not_missing(config.top_k) else None,
                        n=1,
                    ),
                    instruction=GeminiMessage(
                        role="",
                        content=[GeminiTextMessageContent(text=instruction)],
                    )
                    if instruction
                    else None,
                    messages=messages,
                    tools=tools or [],
                    tools_config=GeminiToolConfig(
                        function_calling=GeminiToolFunctionCallingConfig(
                            mode=("ANY" if suggest_tools else "AUTO") if tools else "NONE",
                        )
                    ),
                ),
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
            body=request,
        )

    async def dispose(self) -> None:
        await self._client.aclose()

    async def _request[Requested: DataModel](  # noqa: PLR0913
        self,
        model: type[Requested],
        method: str,
        url: str,
        query: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        body: DataModel | dict[str, Any] | None = None,
        follow_redirects: bool | None = None,
        timeout: float | None = None,
    ) -> Requested:
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

        if HTTPStatus(value=response.status_code).is_success:
            try:
                return model.from_json(await response.aread())

            except Exception as exc:
                raise GeminiException("Failed to decode Gemini response", response) from exc

        else:
            raise GeminiException("Network request failed: %s", response)
