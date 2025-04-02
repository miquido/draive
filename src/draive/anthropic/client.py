from collections.abc import Iterable, Set
from types import TracebackType
from typing import Literal, final, overload

from haiway import State

from draive.anthropic.api import AnthropicAPI
from draive.anthropic.lmm_generation import AnthropicLMMGeneration

__all__ = ("Anthropic",)


@final
class Anthropic(
    AnthropicLMMGeneration,
    AnthropicAPI,
):
    __slots__ = ("_disposable_state",)

    @overload
    def __init__(
        self,
        provider: Literal["anthropic"] = "anthropic",
        /,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        disposable_state: Set[Literal["lmm"]] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        provider: Literal["bedrock"],
        /,
        *,
        base_url: str | None = None,
        timeout: float = 60.0,
        disposable_state: Set[Literal["lmm"]] | None = None,
    ) -> None: ...

    def __init__(
        self,
        provider: Literal["anthropic", "bedrock"] = "anthropic",
        /,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        disposable_state: Set[Literal["lmm"]] | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

        self._disposable_state: frozenset[Literal["lmm"]] = (
            frozenset(disposable_state) if disposable_state is not None else frozenset(("lmm",))
        )

    async def __aenter__(self) -> Iterable[State]:
        await self._initialize_client()
        if "lmm" in self._disposable_state:
            return (self.lmm(),)

        return ()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._deinitialize_client()
