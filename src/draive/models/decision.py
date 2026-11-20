from collections.abc import Sequence
from typing import Any, final, overload

from haiway import State, statemethod

from draive.models.types import (
    DecisionAnswer,
    DecisionGenerating,
    DecisionQuestion,
)
from draive.multimodal import Multimodal

__all__ = ("DecisionModel",)


@final
class DecisionModel(State):
    """Typed facade for model based decisions about a prepared input.

    Delegates execution to a configured ``DecisionGenerating`` callable, i.e. a
    provider backend deciding natively. Completion based decisions are available
    as an explicit option, see ``generate_completion_decisions``.

    Parameters
    ----------
    generating : DecisionGenerating
        Async callable implementing backend specific decision behavior.
    """

    @overload
    @classmethod
    async def decide(
        cls,
        *,
        input: Multimodal,
        questions: Sequence[DecisionQuestion],
        **extra: Any,
    ) -> Sequence[DecisionAnswer]: ...

    @overload
    async def decide(
        self,
        *,
        input: Multimodal,
        questions: Sequence[DecisionQuestion],
        **extra: Any,
    ) -> Sequence[DecisionAnswer]: ...

    @statemethod
    async def decide(
        self,
        *,
        input: Multimodal,  # noqa: A002
        questions: Sequence[DecisionQuestion],
        **extra: Any,
    ) -> Sequence[DecisionAnswer]:
        """Answer decision questions about the input.

        Parameters
        ----------
        input : Multimodal
            Content the decisions are made about.
        questions : Sequence[DecisionQuestion]
            Non-empty questions with unique identifiers.
        **extra : Any
            Additional provider options forwarded to the backend.

        Returns
        -------
        Sequence[DecisionAnswer]
            Answers in the order of the questions.
        """
        assert questions, "Decision requires at least one question"  # nosec: B101
        assert len({question.identifier for question in questions}) == len(  # nosec: B101
            questions
        ), "Decision question identifiers have to be unique"

        return await self._generating(
            input=input,
            questions=questions,
            **extra,
        )

    _generating: DecisionGenerating

    def __init__(
        self,
        generating: DecisionGenerating,
    ) -> None:
        super().__init__(_generating=generating)
