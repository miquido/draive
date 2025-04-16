from collections.abc import Sequence
from typing import Any

from haiway import State, ctx

from draive.tokenization.types import TextTokenizing

__all__ = ("Tokenization",)


class Tokenization(State):
    @classmethod
    def tokenize(
        cls,
        text: str,
        **extra: Any,
    ) -> Sequence[int]:
        """
        Tokenize input text using current Tokenization state.
        """

        return ctx.state(cls).text_tokenizing(
            text,
            **extra,
        )

    @classmethod
    def count_tokens(
        cls,
        text: str,
        **extra: Any,
    ) -> int:
        """
        Count input text tokens using current Tokenization state.
        """

        return len(
            ctx.state(cls).text_tokenizing(
                text,
                **extra,
            )
        )

    text_tokenizing: TextTokenizing
