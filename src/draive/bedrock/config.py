from haiway import MISSING, Missing, State

__all__ = [
    "BedrockChatConfig",
]


class BedrockChatConfig(State):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    max_tokens: int | Missing = MISSING
    timeout: float | Missing = MISSING
    stop_sequences: list[str] | Missing = MISSING
