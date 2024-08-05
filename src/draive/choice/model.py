from typing import Any, Self
from uuid import uuid4

from draive.parameters import State
from draive.types import Multimodal, MultimodalContent

__all__ = [
    "ChoiceOption",
]


class ChoiceOption(State):
    @classmethod
    def of(
        cls,
        content: Multimodal,
        /,
        identifier: str | None = None,
        **meta: Any,
    ) -> Self:
        assert identifier is None or len(identifier) > 0, "Identifier can't be empty"  # nosec: B101

        return cls(
            identifier=identifier or uuid4().hex,
            content=MultimodalContent.of(content),
            meta=meta,
        )

    identifier: str
    content: MultimodalContent
    meta: dict[str, Any]
