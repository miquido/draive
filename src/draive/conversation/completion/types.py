from typing import Any, Protocol, runtime_checkable

from draive.conversation.state import ConversationMemory
from draive.conversation.types import ConversationOutputStream
from draive.models import ModelInstructions
from draive.multimodal import Multimodal
from draive.tools import Toolbox

__all__ = ("ConversationCompleting",)


@runtime_checkable
class ConversationCompleting(Protocol):
    def __call__(
        self,
        *,
        instructions: ModelInstructions,
        toolbox: Toolbox,
        memory: ConversationMemory,
        message: Multimodal,
        **extra: Any,
    ) -> ConversationOutputStream: ...
