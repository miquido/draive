from pathlib import Path
from typing import Any

from draive.mistral.config import MistralChatConfig
from draive.scope import ctx
from draive.sentencepiece import sentencepiece_tokenize_text
from draive.utils import cache

__all__ = [
    "mistral_tokenize_text",
]


def mistral_tokenize_text(
    text: str,
    **extra: Any,
) -> list[int]:
    return sentencepiece_tokenize_text(
        text=text,
        model_path=_model_path(ctx.state(MistralChatConfig).model),
        **extra,
    )


@cache(limit=4)
def _model_path(model_name: str) -> str:
    model_file: str = _mapping.get(model_name, "mistral_v3_tokenizer.model")
    return str(Path(__file__).parent / "tokens" / model_file)


_mapping: dict[str, str] = {
    "open-mistral-7b": "mistral_v1_tokenizer.model",
    "open-mixtral-8x7b": "mistral_v1_tokenizer.model",
    "mistral-embed": "mistral_v1_tokenizer.model",
    "mistral-small": "mistral_v2_tokenizer.model",
    "mistral-large": "mistral_v2_tokenizer.model",
    "open-mixtral-8x22b": "mistral_v3_tokenizer.model",
    "codestral-22b": "mistral_v3_tokenizer.model",
}
