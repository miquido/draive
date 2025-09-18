from collections.abc import Collection, Iterable
from types import TracebackType
from typing import Literal, final, overload

from haiway import State, getenv_str

from draive.anthropic.api import AnthropicAPI
from draive.anthropic.messages import AnthropicMessages
from draive.models import GenerativeModel

__all__ = ("Anthropic",)


@final
class Anthropic(
    AnthropicMessages,
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
        timeout: float | None = None,
        features: Collection[type[GenerativeModel]] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        provider: Literal["bedrock"],
        /,
        *,
        base_url: str | None = None,
        aws_region: str | None = None,
        timeout: float | None = None,
        features: Collection[type[GenerativeModel]] | None = None,
    ) -> None: ...

    def __init__(
        self,
        provider: Literal["anthropic", "bedrock"] = "anthropic",
        /,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        aws_region: str | None = None,
        timeout: float | None = None,
        features: Collection[type[GenerativeModel]] | None = None,
    ) -> None:
        super().__init__(
            provider,
            base_url=base_url,
            api_key=api_key or getenv_str("ANTHROPIC_API_KEY"),
            aws_region=aws_region or getenv_str("AWS_BEDROCK_REGION"),
            timeout=timeout,
        )

        self._features: Collection[type[GenerativeModel]]
        if features is not None:
            self._features = features

        else:
            self._features = (GenerativeModel,)

    async def __aenter__(self) -> Iterable[State]:
        self._client = self._prepare_client()
        await self._client.__aenter__()
        if GenerativeModel in self._features:
            return (self.generative_model(),)

        return ()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._client.__aexit__(
            exc_type,
            exc_val,
            exc_tb,
        )
        del self._client
