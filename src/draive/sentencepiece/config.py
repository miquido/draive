from draive.parameters import DataModel
from draive.utils import MISSING, Missing

__all__ = [
    "SentencePieceConfig",
]


class SentencePieceConfig(DataModel):
    model_path: str | Missing = MISSING
