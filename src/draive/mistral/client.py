import json
from asyncio import gather
from collections.abc import AsyncIterable, Sequence
from http import HTTPStatus
from itertools import chain
from typing import Any, Literal, Self, cast, final, overload

from httpx import AsyncClient, Response

from draive.mistral.config import MistralChatConfig, MistralEmbeddingConfig
from draive.mistral.errors import MistralException
from draive.mistral.models import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessage,
    EmbeddingResponse,
)
from draive.parameters import DataModel
from draive.scope import ScopeDependency
from draive.utils import getenv_str, not_missing

__all__ = [
    "MistralClient",
]


@final
class MistralClient(ScopeDependency):
    @classmethod
    def prepare(cls) -> Self:
        return cls(
            endpoint=getenv_str("MISTRAL_ENDPOINT", "https://api.mistral.ai"),
            api_key=getenv_str("MISTRAL_API_KEY"),
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
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            timeout=timeout,
        )

    @overload
    async def chat_completion(
        self,
        *,
        config: MistralChatConfig,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
        require_tools: bool | None = False,
        stream: Literal[True],
    ) -> AsyncIterable[ChatCompletionStreamResponse]: ...

    @overload
    async def chat_completion(
        self,
        *,
        config: MistralChatConfig,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
        require_tools: bool | None = False,
    ) -> ChatCompletionResponse: ...

    async def chat_completion(  # noqa: PLR0913
        self,
        *,
        config: MistralChatConfig,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
        require_tools: bool | None = False,
        stream: bool = False,
    ) -> AsyncIterable[ChatCompletionStreamResponse] | ChatCompletionResponse:
        if stream:
            raise NotImplementedError("Mistral streaming is not supported yet")

        else:
            tool_choice: str
            match require_tools:
                case None:
                    tool_choice = "none"

                case True:
                    tool_choice = "any"

                case False:
                    tool_choice = "auto"

            return await self._create_chat_completion(
                messages=messages,
                model=config.model,
                temperature=config.temperature,
                top_p=config.top_p if not_missing(config.top_p) else None,
                max_tokens=config.max_tokens if not_missing(config.max_tokens) else None,
                response_format=cast(dict[str, str], config.response_format),
                seed=config.seed if not_missing(config.seed) else None,
                tools=tools,
                tool_choice=tool_choice if tools else None,
            )

    async def embedding(
        self,
        config: MistralEmbeddingConfig,
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

    async def _create_chat_completion(  # noqa: PLR0913
        self,
        model: str,
        temperature: float,
        top_p: float | None,
        seed: int | None,
        max_tokens: int | None,
        response_format: dict[str, str] | None,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None,
        tool_choice: str | None,
    ) -> ChatCompletionResponse:
        request_body: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }

        if tools:
            request_body["tools"] = tools
        if tool_choice is not None:
            request_body["tool_choice"] = tool_choice
        elif tools:
            request_body["tool_choice"] = "auto"
        if max_tokens:
            request_body["max_tokens"] = max_tokens
        if top_p is not None:
            request_body["top_p"] = top_p
        if seed is not None:
            request_body["random_seed"] = seed
        if response_format is not None:
            request_body["response_format"] = response_format

        return await self._request(
            model=ChatCompletionResponse,
            method="POST",
            url="v1/chat/completions",
            body=request_body,
        )

    async def _create_text_embedding(
        self,
        model: str,
        texts: Sequence[str],
    ) -> list[list[float]]:
        response: EmbeddingResponse = await self._request(
            model=EmbeddingResponse,
            method="POST",
            url="v1/embeddings",
            body={
                "model": model,
                "input": texts,
            },
        )
        return [element.embedding for element in response.data]

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
            raise MistralException("Network request failed") from exc

        status: HTTPStatus = HTTPStatus(value=response.status_code)
        if status.is_success:
            try:
                return model.from_json(await response.aread())

            except Exception as exc:
                raise MistralException("Failed to decode Mistral response %s", response) from exc

        elif status.is_client_error:
            error_body: bytes = await response.aread()
            raise MistralException(
                "Mistral request error: %s %s",
                status,
                error_body.decode("utf-8"),
            )

        else:
            raise MistralException("Network request failed %s", response)
