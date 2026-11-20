from collections.abc import Sequence
from typing import Any

from draive.models import DecisionAnswer, DecisionModel, DecisionQuestion
from draive.multimodal import Multimodal

__all__ = ("generate_decisions",)


async def generate_decisions(
    *,
    input: Multimodal,  # noqa: A002
    questions: Sequence[DecisionQuestion],
    **extra: Any,
) -> Sequence[DecisionAnswer]:
    return await DecisionModel.decide(
        input=input,
        questions=questions,
        **extra,
    )
