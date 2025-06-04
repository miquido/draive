from collections.abc import Collection, Iterable, Mapping
from types import TracebackType
from typing import Any, Literal, final

from haiway import State

from draive.vllm.api import VLLMAPI
from draive.vllm.embedding import VLLMEmbedding
from draive.vllm.lmm_generation import VLLMLMMGeneration

__all__ = ("VLLM",)


@final
class VLLM(
    VLLMLMMGeneration,
    VLLMEmbedding,
    VLLMAPI,
):
    """
    Access to VLLM services, can be used to prepare various functionalities like lmm.
    """

    __slots__ = ("_features",)

    def __init__(
        self,
        base_url: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        features: Collection[Literal["lmm", "text_embedding"]] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(
            base_url=base_url,
            default_headers=default_headers,
            **extra,
        )
        self._features: frozenset[Literal["lmm", "text_embedding"]] = (
            frozenset(features) if features is not None else frozenset(("lmm", "text_embedding"))
        )

    async def __aenter__(self) -> Iterable[State]:
        await self._initialize_client()
        state: list[State] = []
        if "lmm" in self._features:
            state.append(self.lmm())

        if "text_embedding" in self._features:
            state.append(self.text_embedding())

        return state

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._deinitialize_client()
