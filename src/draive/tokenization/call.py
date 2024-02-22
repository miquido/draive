from draive.scope import ctx
from draive.tokenization.state import Tokenization


def count_text_tokens(
    text: str,
) -> int:
    return ctx.state(Tokenization).count_text_tokens(text=text)
