from collections.abc import Iterable, Mapping
from types import TracebackType
from typing import Any, final

from haiway import State

from draive.generation import DecisionGeneration
from draive.systemone.api import SystemOneAPI, SystemOneImages
from draive.systemone.decisions import SystemOneDecisions

__all__ = ("SystemOne",)


@final
class SystemOne(
    SystemOneDecisions,
    SystemOneAPI,
):
    """Access to a System One API decision server, prepares ``DecisionGeneration``.

    Works with the hosted TypeSafe API and with self-hosted OpenJev servers,
    which follow the same request shape but differ in image delivery.

    Parameters
    ----------
    server_url : str | None, default=None
        Server address, resolved from ``SYSTEMONE_SERVER_URL`` when omitted and
        defaulting to ``https://api.typesafe.ai``.
    api_key : str | None, default=None
        Bearer token, resolved from ``SYSTEMONE_API_KEY`` or ``TYPESAFE_API_KEY`` when
        omitted. Requests are sent without authorization when none is set.
    images : SystemOneImages, default="assets"
        How images are delivered - ``"assets"`` for the TypeSafe API, which
        accepts any number of png / jpeg images placed within the state,
        ``"images"`` for Ollama, which accepts images shared by all questions
        in a separate list, or ``"inline"`` for OpenJev servers, which accept
        a single image within the state.
    headers : Mapping[str, str] | None, default=None
        Additional headers sent with each request, i.e. TypeSafe feature flags.
    timeout : float, default=30
        Request timeout in seconds.
    **extra : Any
        Additional arguments passed to the underlying ``HTTPXClient``.
    """

    __slots__ = ()

    def __init__(
        self,
        server_url: str | None = None,
        api_key: str | None = None,
        images: SystemOneImages = "assets",
        headers: Mapping[str, str] | None = None,
        timeout: float = 30,
        **extra: Any,
    ) -> None:
        super().__init__(
            server_url=server_url,
            api_key=api_key,
            images=images,
            headers=headers,
            timeout=timeout,
            **extra,
        )

    async def __aenter__(self) -> Iterable[State]:
        await self._initialize_client()
        return (DecisionGeneration(generating=self.decide),)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._deinitialize_client()
