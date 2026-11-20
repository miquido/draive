from collections.abc import Sequence
from typing import Any, overload

from haiway import State, ctx, statemethod

from draive.generation.decision.default import generate_decisions
from draive.models import (
    DecisionAnswer,
    DecisionCheck,
    DecisionCheckAnswer,
    DecisionChoice,
    DecisionChoiceAnswer,
    DecisionGenerating,
    DecisionOption,
    DecisionQuestion,
    DecisionRating,
    DecisionRatingAnswer,
    DecisionSelection,
    DecisionSelectionAnswer,
)
from draive.multimodal import Multimodal, Template, TemplatesRepository

__all__ = ("DecisionGeneration",)


class DecisionGeneration(State):
    @overload
    @classmethod
    async def decide(
        cls,
        *,
        input: Template | Multimodal,
        questions: Sequence[DecisionQuestion],
        **extra: Any,
    ) -> Sequence[DecisionAnswer]: ...

    @overload
    async def decide(
        self,
        *,
        input: Template | Multimodal,
        questions: Sequence[DecisionQuestion],
        **extra: Any,
    ) -> Sequence[DecisionAnswer]: ...

    @statemethod
    async def decide(
        self,
        *,
        input: Template | Multimodal,  # noqa: A002
        questions: Sequence[DecisionQuestion],
        **extra: Any,
    ) -> Sequence[DecisionAnswer]:
        """Answer decision questions about a shared input.

        Each question is answered independently of the others. Implementations
        answer every question in the order of the questions. Choice and rating
        answers carry the decided option of their question, with its meta
        merged into the answer meta. Implementations put their specific values,
        i.e. calibrated probabilities, into the answer meta as well.

        Parameters
        ----------
        input : Template | Multimodal
            Content the decisions are made about, or a template resolved
            before deciding.
        questions : Sequence[DecisionQuestion]
            Questions with unique identifiers.
        **extra : Any
            Additional provider options forwarded to the backend.

        Returns
        -------
        Sequence[DecisionAnswer]
            Answers in the order of their questions, each carrying the
            identifier of the answered question.

        Raises
        ------
        AssertionError
            When questions are empty or their identifiers are not unique.
        GenerationOutputMissing
            When the default implementation could not resolve a valid answer.
        ModelException
            When the backend failed, i.e. ``ModelInputInvalid`` for input it
            does not support or ``ModelRateLimit`` when rate limited.
        ExceptionGroup
            When several questions failed together within the default
            implementation, grouping their underlying failures.
        """
        assert questions and len({question.identifier for question in questions}) == len(questions)  # nosec: B101
        async with ctx.scope("decision_generation"):
            ctx.record_info(
                attributes={"decision.questions": [question.identifier for question in questions]},
            )

            if isinstance(input, Template):
                ctx.record_info(
                    attributes={"input.template": input.identifier},
                )
                input = await TemplatesRepository.resolve(input)  # noqa: A001

            return await self._generating(
                input=input,
                questions=questions,
                **extra,
            )

    @overload
    @classmethod
    async def check(
        cls,
        *,
        input: Template | Multimodal,
        criteria: str,
        **extra: Any,
    ) -> DecisionCheckAnswer: ...

    @overload
    async def check(
        self,
        *,
        input: Template | Multimodal,
        criteria: str,
        **extra: Any,
    ) -> DecisionCheckAnswer: ...

    @statemethod
    async def check(
        self,
        *,
        input: Template | Multimodal,  # noqa: A002
        criteria: str,
        **extra: Any,
    ) -> DecisionCheckAnswer:
        """Answer a single yes / no question about the input.

        Parameters
        ----------
        input : Template | Multimodal
            Content the decision is made about, or a template resolved
            before deciding.
        criteria : str
            Criteria evaluated against the input.
        **extra : Any
            Additional provider options forwarded to the backend.

        Returns
        -------
        DecisionCheckAnswer
            Decided outcome.

        Raises
        ------
        GenerationOutputMissing
            When the backend did not answer the question.
        """
        answers: Sequence[DecisionAnswer] = await self.decide(
            input=input,
            questions=(
                DecisionCheck(
                    identifier="check",
                    criteria=criteria,
                ),
            ),
            **extra,
        )
        assert isinstance(answers[0], DecisionCheckAnswer)  # nosec: B101
        return answers[0]

    @overload
    @classmethod
    async def choose(
        cls,
        *,
        input: Template | Multimodal,
        criteria: str,
        options: Sequence[DecisionOption],
        **extra: Any,
    ) -> DecisionChoiceAnswer: ...

    @overload
    async def choose(
        self,
        *,
        input: Template | Multimodal,
        criteria: str,
        options: Sequence[DecisionOption],
        **extra: Any,
    ) -> DecisionChoiceAnswer: ...

    @statemethod
    async def choose(
        self,
        *,
        input: Template | Multimodal,  # noqa: A002
        criteria: str,
        options: Sequence[DecisionOption],
        **extra: Any,
    ) -> DecisionChoiceAnswer:
        """Choose a single option matching the input.

        Parameters
        ----------
        input : Template | Multimodal
            Content the decision is made about, or a template resolved
            before deciding.
        criteria : str
            Criteria evaluated against the input.
        options : Sequence[DecisionOption]
            At least two options with unique identifiers.
        **extra : Any
            Additional provider options forwarded to the backend.

        Returns
        -------
        DecisionChoiceAnswer
            Decided option, its meta merged into the answer meta.

        Raises
        ------
        GenerationOutputMissing
            When the backend did not answer with one of the options.
        """
        answers: Sequence[DecisionAnswer] = await self.decide(
            input=input,
            questions=(
                DecisionChoice(
                    identifier="choice",
                    criteria=criteria,
                    options=options,
                ),
            ),
            **extra,
        )

        assert isinstance(answers[0], DecisionChoiceAnswer)  # nosec: B101
        return answers[0]

    @overload
    @classmethod
    async def select(
        cls,
        *,
        input: Template | Multimodal,
        criteria: str,
        options: Sequence[DecisionOption],
        **extra: Any,
    ) -> DecisionSelectionAnswer: ...

    @overload
    async def select(
        self,
        *,
        input: Template | Multimodal,
        criteria: str,
        options: Sequence[DecisionOption],
        **extra: Any,
    ) -> DecisionSelectionAnswer: ...

    @statemethod
    async def select(
        self,
        *,
        input: Template | Multimodal,  # noqa: A002
        criteria: str,
        options: Sequence[DecisionOption],
        **extra: Any,
    ) -> DecisionSelectionAnswer:
        """Select every option matching the input.

        Parameters
        ----------
        input : Template | Multimodal
            Content the decision is made about, or a template resolved
            before deciding.
        criteria : str
            Criteria evaluated against the input.
        options : Sequence[DecisionOption]
            At least one option with a unique identifier, each of them is
            selected independently of the others.
        **extra : Any
            Additional provider options forwarded to the backend.

        Returns
        -------
        DecisionSelectionAnswer
            Selected options, empty when none of them applies.

        Raises
        ------
        GenerationOutputMissing
            When the backend did not answer with the given options.
        """
        answers: Sequence[DecisionAnswer] = await self.decide(
            input=input,
            questions=(
                DecisionSelection(
                    identifier="selection",
                    criteria=criteria,
                    options=options,
                ),
            ),
            **extra,
        )
        assert isinstance(answers[0], DecisionSelectionAnswer)  # nosec: B101
        return answers[0]

    @overload
    @classmethod
    async def rate(
        cls,
        *,
        input: Template | Multimodal,
        criteria: str,
        levels: Sequence[DecisionOption],
        **extra: Any,
    ) -> DecisionRatingAnswer: ...

    @overload
    async def rate(
        self,
        *,
        input: Template | Multimodal,
        criteria: str,
        levels: Sequence[DecisionOption],
        **extra: Any,
    ) -> DecisionRatingAnswer: ...

    @statemethod
    async def rate(
        self,
        *,
        input: Template | Multimodal,  # noqa: A002
        criteria: str,
        levels: Sequence[DecisionOption],
        **extra: Any,
    ) -> DecisionRatingAnswer:
        """Rate the input on an ordered scale of levels.

        Parameters
        ----------
        input : Template | Multimodal
            Content the decision is made about, or a template resolved
            before deciding.
        criteria : str
            Criteria evaluated against the input.
        levels : Sequence[DecisionOption]
            At least two levels with unique identifiers, ordered from the
            lowest to the highest.
        **extra : Any
            Additional provider options forwarded to the backend.

        Returns
        -------
        DecisionRatingAnswer
            Decided level, its meta merged into the answer meta.

        Raises
        ------
        GenerationOutputMissing
            When the backend did not answer with one of the levels.
        """
        answers: Sequence[DecisionAnswer] = await self.decide(
            input=input,
            questions=(
                DecisionRating(
                    identifier="rating",
                    criteria=criteria,
                    levels=levels,
                ),
            ),
            **extra,
        )
        answer: DecisionAnswer = answers[0]
        assert isinstance(answer, DecisionRatingAnswer)  # nosec: B101
        return answer

    _generating: DecisionGenerating = generate_decisions

    def __init__(
        self,
        generating: DecisionGenerating = generate_decisions,
    ) -> None:
        super().__init__(_generating=generating)
