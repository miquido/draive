from typing import Final

from draive import getenv_int, getenv_str

__all__ = [
    "QDRANT_GRPC_PORT",
    "QDRANT_HOST",
    "QDRANT_PORT",
]

QDRANT_HOST: Final[str] = getenv_str(
    "QDRANT_HOST",
    default="localhost",
)
QDRANT_PORT: Final[int] = getenv_int(
    "QDRANT_PORT",
    default=6333,
)
QDRANT_GRPC_PORT: Final[int] = getenv_int(
    "QDRANT_GRPC_PORT",
    default=6334,
)
