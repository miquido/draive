from collections.abc import Collection, Iterable
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
    __slots__ = ("_features",)

    @overload
    def __init__(
        self,
        provider: Literal["anthropic"] = "anthropic",
        /,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        features: Collection[Literal["lmm"]] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        provider: Literal["bedrock"],
        /,
        *,
        base_url: str | None = None,
        aws_region: str | None = None,
        timeout: float = 60.0,
        features: Collection[Literal["lmm"]] | None = None,
    ) -> None: ...

    def __init__(
        self,
        provider: Literal["anthropic", "bedrock"] = "anthropic",
        /,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        aws_region: str | None = None,
        timeout: float = 60.0,
        features: Collection[Literal["lmm"]] | None = None,
    ) -> None:
        super().__init__(
            provider,
            base_url=base_url,
            api_key=api_key,
            aws_region=aws_region,
            timeout=timeout,
        )

        self._features: frozenset[Literal["lmm"]] = (
            frozenset(features) if features is not None else frozenset(("lmm",))
        )

    async def __aenter__(self) -> Iterable[State]:
        await self._initialize_client()
        if "lmm" in self._features:
            return (self.lmm(),)

        return ()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._deinitialize_client()
