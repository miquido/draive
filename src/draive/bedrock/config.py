from haiway import MISSING, Missing

from draive.parameters import DataModel

__all__ = [
    "BedrockChatConfig",
]


class BedrockChatConfig(DataModel):
    model: str
    temperature: float = 0.75
    top_p: float | Missing = MISSING
    max_tokens: int = 2048
    timeout: float | Missing = MISSING
    stop_sequences: list[str] | Missing = MISSING
