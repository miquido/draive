from pathlib import Path

from draive.sentencepiece import sentencepiece_processor, sentencepiece_text_tokenizer
from draive.tokenization import TextTokenizing

__all__ = [
    "gemini_text_tokenizer",
]


def gemini_text_tokenizer(
    model_name: str,
    /,
) -> TextTokenizing:
    return sentencepiece_text_tokenizer(sentencepiece_processor(model_path=_model_path(model_name)))


def _model_path(model_name: str) -> str:
    model_file: str = _mapping.get(model_name, "gemini_tokenizer.model")
    return str(Path(__file__).parent / "tokens" / model_file)


_mapping: dict[str, str] = {
    "gemini-1.5-flash": "gemini_tokenizer.model",
    "gemini-1.5-pro": "gemini_tokenizer.model",
}
