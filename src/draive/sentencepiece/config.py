from haiway import MISSING, Missing

from draive.parameters import DataModel

__all__ = [
    "SentencePieceConfig",
]


class SentencePieceConfig(DataModel):
    model_path: str | Missing = MISSING
