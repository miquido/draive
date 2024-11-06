from pathlib import Path

from draive.sentencepiece import sentencepiece_processor, sentencepiece_tokenizer
from draive.tokenization import Tokenization

__all__ = [
    "gemini_tokenizer",
]


def gemini_tokenizer(
    model_name: str,
    /,
) -> Tokenization:
    return Tokenization(
        tokenize_text=sentencepiece_tokenizer(
            sentencepiece_processor(model_path=_model_path(model_name))
        )
    )


def _model_path(model_name: str) -> str:
    model_file: str = _mapping.get(model_name, "gemini_tokenizer.model")
    return str(Path(__file__).parent / "tokens" / model_file)


_mapping: dict[str, str] = {
    "gemini-1.5-flash": "gemini_tokenizer.model",
    "gemini-1.5-pro": "gemini_tokenizer.model",
}
