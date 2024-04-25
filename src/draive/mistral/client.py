import json
from asyncio import gather
from collections.abc import AsyncIterable, Iterable
from itertools import chain
from typing import Any, Literal, Self, cast, final, overload

from httpx import AsyncClient, Response

from draive.helpers import getenv_str, when_missing
from draive.mistral.config import MistralChatConfig, MistralEmbeddingConfig
from draive.mistral.errors import MistralException
from draive.mistral.models import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessage,
    EmbeddingResponse,
)
from draive.scope import ScopeDependency
from draive.types import Model

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
        tools: list[dict[str, object]],
        stream: Literal[True],
    ) -> AsyncIterable[ChatCompletionStreamResponse]: ...

    @overload
    async def chat_completion(
        self,
        *,
        config: MistralChatConfig,
        messages: list[ChatMessage],
        tools: list[dict[str, object]],
        suggest_tools: bool,
        stream: Literal[True],
    ) -> AsyncIterable[ChatCompletionStreamResponse]: ...

    @overload
    async def chat_completion(
        self,
        *,
        config: MistralChatConfig,
        messages: list[ChatMessage],
        tools: list[dict[str, object]],
        suggest_tools: bool = False,
    ) -> ChatCompletionResponse: ...

    async def chat_completion(  # noqa: PLR0913
        self,
        *,
        config: MistralChatConfig,
        messages: list[ChatMessage],
        tools: list[dict[str, object]],
        suggest_tools: bool = False,
        stream: bool = False,
    ) -> AsyncIterable[ChatCompletionStreamResponse] | ChatCompletionResponse:
        if stream:
            raise NotImplementedError("Mistral streaming is not supported yet")
        else:
            return await self._create_chat_completion(
                messages=messages,
                model=config.model,
                temperature=config.temperature,
                top_p=when_missing(config.top_p, default=None),
                max_tokens=when_missing(config.max_tokens, default=None),
                response_format=cast(dict[str, str], config.response_format),
                seed=when_missing(config.seed, default=None),
                tools=tools,
                tool_choice=("any" if suggest_tools else "auto") if tools else None,
            )

    async def embedding(
        self,
        config: MistralEmbeddingConfig,
        inputs: Iterable[str],
    ) -> list[list[float]]:
        inputs_list: list[str] = list(inputs)
        return list(
            chain(
                *await gather(
                    *[
                        self._create_text_embedding(
                            model=config.model,
                            texts=list(inputs_list[index : index + config.batch_size]),
                        )
                        for index in range(0, len(inputs_list), config.batch_size)
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
        tools: list[dict[str, object]],
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
        texts: list[str],
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

    async def _request[Requested: Model](  # noqa: PLR0913
        self,
        model: type[Requested],
        method: str,
        url: str,
        query: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        body: Model | dict[str, Any] | None = None,
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

            case body_model if isinstance(body_model, Model):
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

        if response.status_code in range(200, 299):
            try:
                return model.from_json(await response.aread())
            except Exception as exc:
                raise MistralException("Failed to decode mistral response", response) from exc
        else:
            raise MistralException("Network request failed", response)
