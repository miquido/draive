from typing import Literal, overload

from haiway import getenv_int, getenv_str
from qdrant_client import AsyncQdrantClient

from draive.qdrant.utils import qdrant_operation

__all__ = ("QdrantSession",)


class QdrantSession:
    __slots__ = (
        "_api_key",
        "_client",
        "_grpc_port",
        "_host",
        "_in_memory",
        "_port",
        "_ssl",
        "_timeout",
    )

    @overload
    def __init__(
        self,
        *,
        in_memory: Literal[True],
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        grpc_port: int | None = None,
        api_key: str | None = None,
        ssl: bool = False,
        timeout: int = 5,
        in_memory: Literal[False] = False,
    ) -> None: ...

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        grpc_port: int | None = None,
        api_key: str | None = None,
        ssl: bool = False,
        timeout: int = 5,
        in_memory: bool = False,
    ) -> None:
        # environment is read when the session is created, not when the module is imported
        self._host: str = host if host is not None else getenv_str("QDRANT_HOST", "localhost")
        self._port: int = port if port is not None else getenv_int("QDRANT_PORT", 6333)
        self._grpc_port: int = (
            grpc_port if grpc_port is not None else getenv_int("QDRANT_GRPC_PORT", 6334)
        )
        self._api_key: str | None = api_key if api_key is not None else getenv_str("QDRANT_API_KEY")
        self._ssl: bool = ssl
        self._timeout: int = timeout
        self._in_memory: bool = in_memory
        self._client: AsyncQdrantClient  # initialized later

    @property
    def client(self) -> AsyncQdrantClient:
        assert self._client is not None  # nosec: B101
        return self._client

    async def _open_session(self) -> None:
        assert not hasattr(self, "_client")  # nosec: B101

        if self._in_memory:
            self._client = AsyncQdrantClient(location=":memory:")

        else:
            self._client = AsyncQdrantClient(
                host=self._host,
                port=self._port,
                grpc_port=self._grpc_port,
                https=self._ssl,
                api_key=self._api_key,
                timeout=self._timeout,
                prefer_grpc=True,
            )

    async def _close_session(self) -> None:
        try:
            with qdrant_operation("session closing"):
                await self._client.close(grpc_grace=5)  # 5 sec grace period

        finally:
            del self._client
