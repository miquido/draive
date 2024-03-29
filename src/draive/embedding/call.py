from collections.abc import Iterable

from draive.embedding.state import Embedding
from draive.scope import ctx
from draive.types import Embedded

__all__ = [
    "embed_text",
]


async def embed_text(
    values: Iterable[str],
) -> list[Embedded[str]]:
    return await ctx.state(Embedding).embed_text(values=values)
