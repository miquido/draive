from typing_extensions import deprecated

from draive.parameters import DataModel
from draive.utils import MISSING, Missing

__all__ = [
    "MRSChatConfig",
]


@deprecated("mistralrs support will be removed")
class MRSChatConfig(DataModel):
    model: str
    temperature: float = 0.75
    top_p: float | Missing = MISSING
    top_k: int | Missing = MISSING
    max_tokens: int = 2048
    stop_sequences: list[str] | Missing = MISSING
