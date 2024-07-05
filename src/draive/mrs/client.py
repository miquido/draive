from asyncio import Lock
from collections.abc import AsyncIterable, Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, Final, Literal, Self, cast, final, overload

from mistralrs import (  # type: ignore
    Architecture,
    ChatCompletionChunkResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Runner,
    Which,
)

from draive.mrs.config import MRSChatConfig
from draive.mrs.errors import MRSException
from draive.scope import ScopeDependency, ctx
from draive.utils import not_missing, run_async

__all__ = [
    "MRSClient",
]

MRS_EXECUTOR: Final[Executor | None] = ThreadPoolExecutor(max_workers=1)


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
            return await self._create_chat_stream(
                messages=messages,
                model=config.model,
                temperature=config.temperature,
                top_p=config.top_p if not_missing(config.top_p) else None,
                top_k=config.top_k if not_missing(config.top_k) else None,
                max_tokens=config.max_tokens if not_missing(config.max_tokens) else None,
            )

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
        return await self._send_chat_completion_request(
            runner=await self._get_runner(model),
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            messages=messages,
        )

    async def _create_chat_stream(  # noqa: PLR0913
        self,
        model: str,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
        max_tokens: int | None,
        messages: list[dict[str, object]],
    ) -> AsyncIterable[ChatCompletionChunkResponse]:
        return ctx.stream_sync(
            await self._send_chat_completion_stream_request(
                runner=await self._get_runner(model),
                model=model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                messages=messages,
            ),
            executor=MRS_EXECUTOR,
        )

    async def dispose(self) -> None:
        pass

    async def _get_runner(
        self,
        model_name: str,
        /,
    ) -> Runner:
        async with self._cache_lock:
            if current := self._runners.get(model_name):
                return current

            elif model := self._models.get(model_name):
                runner: Runner = await self._load_runner(model=model)
                self._runners[model_name] = runner
                return runner

            else:
                raise MRSException(
                    "Requested unsupported model - %s",
                    model_name,
                )

    @run_async(executor=MRS_EXECUTOR)
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

    @run_async(executor=MRS_EXECUTOR)
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

    @run_async(executor=MRS_EXECUTOR)
    def _send_chat_completion_stream_request(  # noqa: PLR0913
        self,
        runner: Runner,
        model: str,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
        max_tokens: int | None,
        messages: list[dict[Any, Any]],
    ) -> Iterator[ChatCompletionChunkResponse]:
        return cast(
            Iterator[ChatCompletionChunkResponse],
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
                    stream=True,
                    top_k=top_k,
                    grammar=None,
                    grammar_type=None,
                    adapters=None,
                ),
            ),
        )
