from functools import cache

from tiktoken import Encoding, encoding_for_model

from draive.openai.config import OpenAIChatConfig
from draive.scope import ctx

__all__ = [
    "openai_count_text_tokens",
]


def openai_count_text_tokens(
    text: str,
) -> int:
    return len(
        _encoding(
            model_name=ctx.state(OpenAIChatConfig).model,
        ).encode(text=text)
    )


@cache
def _encoding(model_name: str) -> Encoding:
    return encoding_for_model(model_name=model_name)
