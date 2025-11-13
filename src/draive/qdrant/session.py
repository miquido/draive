from typing import Literal, overload

from qdrant_client import AsyncQdrantClient

from draive.qdrant.config import QDRANT_GRPC_PORT, QDRANT_HOST, QDRANT_PORT

__all__ = ("QdrantSession",)


class QdrantSession:
    __slots__ = (
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
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        grpc_port: int = QDRANT_GRPC_PORT,
        ssl: bool = False,
        timeout: int = 5,
        in_memory: Literal[False] = False,
    ) -> None: ...

    def __init__(
        self,
        *,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        grpc_port: int = QDRANT_GRPC_PORT,
        ssl: bool = False,
        timeout: int = 5,
        in_memory: bool = False,
    ) -> None:
        self._host: str = host
        self._port: int = port
        self._grpc_port: int = grpc_port
        self._ssl: bool = ssl
        self._timeout: int = timeout
        self._in_memory: bool = in_memory
        self._client: AsyncQdrantClient | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        assert self._client is not None  # nosec: B101
        return self._client

    async def _open_session(self) -> None:
        if self._client is not None:
            await self._client.close(grpc_grace=5)

        if self._in_memory:
            self._client = AsyncQdrantClient(location=":memory:")

        else:
            self._client = AsyncQdrantClient(
                host=self._host,
                port=self._port,
                grpc_port=self._grpc_port,
                https=self._ssl,
                timeout=self._timeout,
                prefer_grpc=True,
            )

    async def _close_session(self) -> None:
        if self._client is None:
            return

        try:
            await self._client.close(grpc_grace=5)  # 5 sec grace period

        finally:
            self._client = None
