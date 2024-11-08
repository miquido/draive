import json
from asyncio import gather
from collections.abc import AsyncIterator, Sequence
from http import HTTPStatus
from itertools import chain
from types import TracebackType
from typing import Any, ClassVar, Literal, Self, cast, final, overload

from haiway import getenv_str, not_missing
from httpx import AsyncClient, Response

from draive.mistral.config import MistralChatConfig, MistralEmbeddingConfig
from draive.mistral.models import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessage,
    EmbeddingResponse,
)
from draive.mistral.types import MistralException
from draive.parameters import DataModel

__all__ = [
    "MistralClient",
]


@final
class MistralClient:
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
            "MISTRAL_ENDPOINT",
            default="https://api.mistral.ai",
        )
        self._api_key: str | None = api_key or getenv_str("MISTRAL_API_KEY")
        self._timeout: float = timeout or 60
        self._client: AsyncClient = self._initialize_client()

    def _initialize_client(self) -> AsyncClient:
        return AsyncClient(  # nosec: B113 - false positive
            base_url=self._endpoint,
            headers={
                "Authorization": f"Bearer {self._api_key}",
            },
            timeout=self._timeout,
        )

    @overload
    async def chat_completion(
        self,
        *,
        config: MistralChatConfig,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
        tool_choice: Literal["auto", "any", "none"] = "auto",
        stream: Literal[True],
    ) -> AsyncIterator[ChatCompletionStreamResponse]: ...

    @overload
    async def chat_completion(
        self,
        *,
        config: MistralChatConfig,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
        tool_choice: Literal["auto", "any", "none"] = "auto",
    ) -> ChatCompletionResponse: ...

    async def chat_completion(
        self,
        *,
        config: MistralChatConfig,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
        tool_choice: Literal["auto", "any", "none"] = "auto",
        stream: bool = False,
    ) -> AsyncIterator[ChatCompletionStreamResponse] | ChatCompletionResponse:
        if stream:
            raise NotImplementedError("Mistral streaming is not supported yet")

        else:
            if messages[-1]["role"] == "assistant":
                if config.response_format == {"type": "json_object"}:
                    del messages[-1]  # for json mode ignore prefill

                else:
                    messages[-1]["prefix"] = True  # add prefill parameter indicator

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
                stop=config.stop_sequences if not_missing(config.stop_sequences) else None,
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
        stop: list[str] | None,
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
        if stop:
            request_body["stop"] = stop
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
