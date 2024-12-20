import json
from collections.abc import Sequence
from http import HTTPStatus
from types import TracebackType
from typing import Any, ClassVar, Literal, Self, cast, final

from haiway import getenv_str, not_missing
from httpx import AsyncClient, Response

from draive.ollama.config import OllamaChatConfig
from draive.ollama.models import ChatCompletionResponse, ChatMessage
from draive.ollama.types import OllamaException
from draive.parameters import DataModel

__all__ = [
    "OllamaClient",
]


@final
class OllamaClient:
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
        timeout: float | None = None,
    ) -> None:
        self._endpoint: str = endpoint or getenv_str(
            "OLLAMA_ENDPOINT",
            default="http://localhost:11434",
        )
        self._timeout: float = timeout or 60
        self._client: AsyncClient = self._initialize_client()

    def _initialize_client(self) -> AsyncClient:
        return AsyncClient(  # nosec: B113 - false positive
            base_url=self._endpoint,
            timeout=self._timeout,
        )

    async def chat_completion(
        self,
        *,
        config: OllamaChatConfig,
        messages: list[ChatMessage],
    ) -> ChatCompletionResponse:
        return await self._create_chat_completion(
            messages=messages,
            model=config.model,
            temperature=config.temperature,
            top_k=config.top_k if not_missing(config.top_k) else None,
            top_p=config.top_p if not_missing(config.top_p) else None,
            max_tokens=config.max_tokens if not_missing(config.max_tokens) else None,
            response_format=cast(Literal["text", "json"], config.response_format)
            if not_missing(config.response_format)
            else "text",
            seed=config.seed if not_missing(config.seed) else None,
            stop=config.stop_sequences if not_missing(config.stop_sequences) else None,
        )

    async def _create_chat_completion(  # noqa: PLR0913
        self,
        model: str,
        temperature: float,
        top_k: float | None,
        top_p: float | None,
        seed: int | None,
        max_tokens: int | None,
        response_format: Literal["text", "json"],
        messages: list[ChatMessage],
        stop: Sequence[str] | None,
    ) -> ChatCompletionResponse:
        request_body: dict[str, Any] = {
            "model": model,
            "messages": [message.as_dict() for message in messages],
            "options": {
                "temperature": temperature,
            },
            "stream": False,
        }

        if max_tokens:
            request_body["options"]["num_predict"] = max_tokens
        if top_k is not None:
            request_body["options"]["top_k"] = top_k
        if top_p is not None:
            request_body["options"]["top_p"] = top_p
        if seed is not None:
            request_body["options"]["seed"] = seed
        if stop:
            request_body["options"]["stop"] = stop
        if response_format == "json":
            request_body["format"] = "json"

        return await self._request(
            model=ChatCompletionResponse,
            method="POST",
            url="/api/chat",
            body=request_body,
        )

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
            raise OllamaException("Network request failed") from exc

        status: HTTPStatus = HTTPStatus(value=response.status_code)
        if status.is_success:
            try:
                return model.from_json(await response.aread())

            except Exception as exc:
                raise OllamaException("Failed to decode Ollama response", response) from exc

        elif status.is_client_error:
            error_body: bytes = await response.aread()
            raise OllamaException(
                "Ollama request error: %s %s",
                status,
                error_body.decode("utf-8"),
            )

        else:
            raise OllamaException("Network request failed", response)
