from types import TracebackType
from typing import final

from draive.surreal.connection import SurrealConnection
from draive.surreal.state import Surreal

__all__ = ["SurrealClient"]


@final
class SurrealClient(SurrealConnection):
    async def __aenter__(self) -> Surreal:
        await self._open_connection()
        return Surreal(session_preparing=self.prepare_session)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._close_connection()
