from asyncio import Lock, get_running_loop
from collections.abc import AsyncIterable
from typing import Any, Literal, Self, cast, final, overload

from mistralrs import (
    Architecture,
    ChatCompletionChunkResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Runner,
    Which,
)

from draive.mrs.config import MRSChatConfig
from draive.mrs.errors import MRSException
from draive.scope import ScopeDependency
from draive.utils import not_missing

__all__ = [
    "MRSClient",
]


@final
class MRSClient(ScopeDependency):
    @classmethod
    def prepare(cls) -> Self:
        return cls(
            models={
                "Phi-3": Which.Plain(
                    model_id="microsoft/Phi-3-mini-4k-instruct",
                    arch=Architecture.Phi3,
                    tokenizer_json=None,
                    repeat_last_n=64,
                ),
            }
        )

    def __init__(
        self,
        models: dict[
            str,
            Which.Plain
            | Which.Lora
            | Which.XLora
            | Which.GGUF
            | Which.GGML
            | Which.LoraGGML
            | Which.LoraGGUF
            | Which.XLoraGGML
            | Which.XLoraGGUF
            | Which.VisionPlain,
        ],
    ) -> None:
        self._models: dict[
            str,
            Which.Plain
            | Which.Lora
            | Which.XLora
            | Which.GGUF
            | Which.GGML
            | Which.LoraGGML
            | Which.LoraGGUF
            | Which.XLoraGGML
            | Which.XLoraGGUF
            | Which.VisionPlain,
        ] = models
        self._runners: dict[str, Runner] = {}
        self._cache_lock: Lock = Lock()

    @overload
    async def chat_completion(
        self,
        *,
        config: MRSChatConfig,
        messages: list[dict[str, object]],
        stream: Literal[True],
    ) -> AsyncIterable[ChatCompletionChunkResponse]: ...

    @overload
    async def chat_completion(
        self,
        *,
        config: MRSChatConfig,
        messages: list[dict[str, object]],
    ) -> ChatCompletionResponse: ...

    async def chat_completion(
        self,
        *,
        config: MRSChatConfig,
        messages: list[dict[str, object]],
        stream: bool = False,
    ) -> AsyncIterable[ChatCompletionChunkResponse] | ChatCompletionResponse:
        if stream:
            raise NotImplementedError("mistral.rs streaming is not supported yet")

        else:
            return await self._create_chat_completion(
                messages=messages,
                model=config.model,
                temperature=config.temperature,
                top_p=config.top_p if not_missing(config.top_p) else None,
                top_k=config.top_k if not_missing(config.top_k) else None,
                max_tokens=config.max_tokens if not_missing(config.max_tokens) else None,
            )

    async def _create_chat_completion(  # noqa: PLR0913
        self,
        model: str,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
        max_tokens: int | None,
        messages: list[dict[str, object]],
    ) -> ChatCompletionResponse:
        return await get_running_loop().run_in_executor(
            None,
            self._send_chat_completion_request,
            await self._get_runner(model),
            model,
            temperature,
            top_p,
            top_k,
            max_tokens,
            messages,
        )

    async def dispose(self) -> None:
        pass

    async def _get_runner(
        self,
        model_name: str,
    ) -> Runner:
        async with self._cache_lock:
            if current := self._runners.get(model_name):
                return current

            elif model := self._models.get(model_name):
                runner: Runner = await get_running_loop().run_in_executor(
                    None,
                    self._load_runner,
                    model,
                )
                self._runners[model_name] = runner
                return runner

            else:
                raise MRSException(
                    "Requested unsupported model - %s",
                    model_name,
                )

    def _load_runner(
        self,
        model: Which.Plain
        | Which.Lora
        | Which.XLora
        | Which.GGUF
        | Which.GGML
        | Which.LoraGGML
        | Which.LoraGGUF
        | Which.XLoraGGML
        | Which.XLoraGGUF
        | Which.VisionPlain,
    ) -> Runner:
        return Runner(which=model)

    def _send_chat_completion_request(  # noqa: PLR0913
        self,
        runner: Runner,
        model: str,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
        max_tokens: int | None,
        messages: list[dict[Any, Any]],
    ) -> ChatCompletionResponse:
        return cast(
            ChatCompletionResponse,
            runner.send_chat_completion_request(
                request=ChatCompletionRequest(
                    messages=messages,
                    model=model,
                    logit_bias=None,
                    logprobs=False,
                    top_logprobs=None,
                    max_tokens=max_tokens,
                    n_choices=1,
                    presence_penalty=None,
                    frequency_penalty=None,
                    stop_seqs=None,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False,
                    top_k=top_k,
                    grammar=None,
                    grammar_type=None,
                    adapters=None,
                ),
            ),
        )
