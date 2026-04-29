try:
    import surrealdb  # pyright: ignore[reportUnusedImport]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "draive.surreal requires the 'surrealdb' extra. "
        "Install via `pip install draive[surrealdb]`."
    ) from exc

from draive.surreal.client import SurrealClient
from draive.surreal.memory import SurrealConversationMemory
from draive.surreal.state import Surreal
from draive.surreal.templates import SurrealTemplatesRepository
from draive.surreal.types import SurrealBasicValue, SurrealException, SurrealObject, SurrealValue
from draive.surreal.vector import SurrealVectorIndex

__all__ = (
    "Surreal",
    "SurrealBasicValue",
    "SurrealClient",
    "SurrealConversationMemory",
    "SurrealException",
    "SurrealObject",
    "SurrealTemplatesRepository",
    "SurrealValue",
    "SurrealVectorIndex",
)
