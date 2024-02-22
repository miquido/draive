from collections.abc import Iterable
from typing import TypeVar

from draive.embedding.state import Embedding
from draive.scope import ctx
from draive.types import StringConvertible
from draive.types.embedded import Embedded

__all__ = [
    "embed_text",
]


_Embeddable = TypeVar(
    "_Embeddable",
    bound=StringConvertible,
)


async def embed_text(
    values: Iterable[_Embeddable],
) -> list[Embedded[_Embeddable]]:
    return await ctx.state(Embedding).embed_text(values=values)
