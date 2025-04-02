from collections.abc import Iterable, Mapping, Set
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

    __slots__ = ("_disposable_state",)

    def __init__(
        self,
        base_url: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        disposable_state: Set[Literal["lmm", "text_embedding"]] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(
            base_url=base_url,
            default_headers=default_headers,
            **extra,
        )
        self._disposable_state: frozenset[Literal["lmm", "text_embedding"]] = (
            frozenset(disposable_state)
            if disposable_state is not None
            else frozenset(("lmm", "text_embedding"))
        )

    async def __aenter__(self) -> Iterable[State]:
        await self._initialize_client()
        state: list[State] = []
        if "lmm" in self._disposable_state:
            state.append(self.lmm())

        if "text_embedding" in self._disposable_state:
            state.append(self.text_embedding())

        return state

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._deinitialize_client()
