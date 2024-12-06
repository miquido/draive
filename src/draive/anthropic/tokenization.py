from pathlib import Path
from typing import Any

from haiway import asynchronous
from tokenizers import Tokenizer

from draive.tokenization import Tokenization

__all__ = [
    "anthropic_tokenizer",
]


async def anthropic_tokenizer() -> Tokenization:
    tokenizer: Tokenizer = await _tokenizer()

    def anthropic_tokenize_text(
        text: str,
        **extra: Any,
    ) -> list[int]:
        return tokenizer.encode(text).ids  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    return Tokenization(tokenize_text=anthropic_tokenize_text)


@asynchronous
def _tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(Path(__file__).parent / "tokenizer.json"))  # pyright: ignore
