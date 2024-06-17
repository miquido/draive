from typing import Literal

from draive.parameters import DataModel
from draive.utils import MISSING, Missing

__all__ = [
    "MRSChatConfig",
]


class MRSChatConfig(DataModel):
    model: Literal["Phi-3"] | str = "Phi-3"
    temperature: float = 0.0
    top_p: float | Missing = MISSING
    top_k: int | Missing = MISSING
    max_tokens: int | Missing = MISSING
