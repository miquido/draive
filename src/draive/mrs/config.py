from draive.parameters import DataModel
from draive.utils import MISSING, Missing

__all__ = [
    "MRSChatConfig",
]


class MRSChatConfig(DataModel):
    model: str = "Phi-3"
    temperature: float = 0.75
    top_p: float | Missing = MISSING
    top_k: int | Missing = MISSING
    max_tokens: int = 2048
