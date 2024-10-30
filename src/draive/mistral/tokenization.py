from pathlib import Path

from draive.sentencepiece import sentencepiece_processor, sentencepiece_tokenizer
from draive.tokenization import Tokenization

__all__ = [
    "mistral_text_tokenizer",
]


def mistral_text_tokenizer(
    model_name: str,
    /,
) -> Tokenization:
    return Tokenization(
        tokenize_text=sentencepiece_tokenizer(
            sentencepiece_processor(model_path=_model_path(model_name))
        )
    )


def _model_path(
    model_name: str,
    /,
) -> str:
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
