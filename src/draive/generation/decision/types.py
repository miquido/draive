from collections.abc import Sequence
from typing import Annotated, Any, Protocol, final, runtime_checkable

from haiway import Meta, State, Verifier

from draive.multimodal import Multimodal

__all__ = (
    "DecisionAnswer",
    "DecisionCheck",
    "DecisionCheckAnswer",
    "DecisionChoice",
    "DecisionChoiceAnswer",
    "DecisionGenerating",
    "DecisionOption",
    "DecisionQuestion",
    "DecisionRating",
    "DecisionRatingAnswer",
    "DecisionSelection",
    "DecisionSelectionAnswer",
)


def _verified_identifier(
    value: str,
) -> str:
    if not value or value != value.strip():
        raise ValueError(
            f"Decision identifier '{value}' can't be empty or surrounded by whitespace"
        )

    return value


def _verified_criteria(
    value: str,
) -> str:
    if not value.strip():
        raise ValueError("Decision criteria can't be empty")

    return value


@final
class DecisionCheck(State, serializable=True):
    """Binary (yes / no) decision question.

    Attributes
    ----------
    identifier : str
        Identifier of the question, its answer is reported under it.
    criteria : str
        Criteria evaluated against the decision input.
    """

    identifier: Annotated[str, Verifier(_verified_identifier)]
    criteria: Annotated[str, Verifier(_verified_criteria)]


@final
class DecisionOption(State, serializable=True):
    """Option of a ``DecisionChoice`` or a level of a ``DecisionRating`` question.

    Attributes
    ----------
    identifier : str
        Identifier of the option, unique within its question.
    description : str | None
        Optional description of when the option applies, omitted when the
        identifier is self-explanatory.
    meta : Meta
        Caller defined values associated with the option, merged into the
        meta of the answer deciding it. They are never sent to the deciding
        backend.
    """

    identifier: Annotated[str, Verifier(_verified_identifier)]
    description: str | None = None
    meta: Meta = Meta.empty


def _verified_options(
    value: Sequence[DecisionOption],
) -> Sequence[DecisionOption]:
    if len(value) < 2:  # noqa: PLR2004
        raise ValueError("Decision requires at least two options")

    if len({option.identifier for option in value}) != len(value):
        raise ValueError("Decision option identifiers have to be unique")

    return value


@final
class DecisionChoice(State, serializable=True):
    """Single-select decision question over mutually exclusive options.

    Attributes
    ----------
    identifier : str
        Identifier of the question, its answer is reported under it.
    criteria : str
        Criteria evaluated against the decision input.
    options : Sequence[DecisionOption]
        At least two options with unique identifiers.
    """

    identifier: Annotated[str, Verifier(_verified_identifier)]
    criteria: Annotated[str, Verifier(_verified_criteria)]
    options: Annotated[Sequence[DecisionOption], Verifier(_verified_options)]


@final
class DecisionRating(State, serializable=True):
    """Rating decision question over ordered levels.

    Attributes
    ----------
    identifier : str
        Identifier of the question, its answer is reported under it.
    criteria : str
        Criteria evaluated against the decision input.
    levels : Sequence[DecisionOption]
        At least two levels with unique identifiers, ordered from the lowest
        to the highest.
    """

    identifier: Annotated[str, Verifier(_verified_identifier)]
    criteria: Annotated[str, Verifier(_verified_criteria)]
    levels: Annotated[Sequence[DecisionOption], Verifier(_verified_options)]


def _verified_selection_options(
    value: Sequence[DecisionOption],
) -> Sequence[DecisionOption]:
    if not value:
        raise ValueError("Decision selection requires at least one option")

    if len({option.identifier for option in value}) != len(value):
        raise ValueError("Decision option identifiers have to be unique")

    return value


@final
class DecisionSelection(State, serializable=True):
    """Multi-select decision question over independent options.

    Attributes
    ----------
    identifier : str
        Identifier of the question, its answer is reported under it.
    criteria : str
        Criteria evaluated against the decision input.
    options : Sequence[DecisionOption]
        At least one option with a unique identifier, each of them is
        selected independently of the others.
    """

    identifier: Annotated[str, Verifier(_verified_identifier)]
    criteria: Annotated[str, Verifier(_verified_criteria)]
    options: Annotated[Sequence[DecisionOption], Verifier(_verified_selection_options)]


DecisionQuestion = DecisionCheck | DecisionChoice | DecisionSelection | DecisionRating
"""Any supported decision question."""


@final
class DecisionCheckAnswer(State, serializable=True):
    """Answer to a ``DecisionCheck`` question.

    Attributes
    ----------
    identifier : str
        Identifier of the answered question.
    value : bool
        Decided outcome.
    meta : Meta
        Backend specific values, i.e. calibrated probabilities.
    """

    identifier: str
    value: bool
    meta: Meta = Meta.empty


@final
class DecisionChoiceAnswer(State, serializable=True):
    """Answer to a ``DecisionChoice`` question.

    Attributes
    ----------
    identifier : str
        Identifier of the answered question.
    choice : DecisionOption
        Decided option.
    meta : Meta
        Meta of the decided option merged with backend specific values,
        i.e. calibrated probabilities.
    """

    identifier: str
    choice: DecisionOption
    meta: Meta = Meta.empty


@final
class DecisionRatingAnswer(State, serializable=True):
    """Answer to a ``DecisionRating`` question.

    Attributes
    ----------
    identifier : str
        Identifier of the answered question.
    level : DecisionOption
        Decided level.
    meta : Meta
        Meta of the decided level merged with backend specific values,
        i.e. calibrated probabilities.
    """

    identifier: str
    level: DecisionOption
    meta: Meta = Meta.empty


@final
class DecisionSelectionAnswer(State, serializable=True):
    """Answer to a ``DecisionSelection`` question.

    Attributes
    ----------
    identifier : str
        Identifier of the answered question.
    selected : Sequence[DecisionOption]
        Selected options in the order of the question options, empty when
        none of them applies. Each option carries its own meta.
    meta : Meta
        Backend specific values, i.e. calibrated probabilities.
    """

    identifier: str
    selected: Sequence[DecisionOption]
    meta: Meta = Meta.empty


DecisionAnswer = (
    DecisionCheckAnswer | DecisionChoiceAnswer | DecisionSelectionAnswer | DecisionRatingAnswer
)
"""Any supported decision answer."""


@runtime_checkable
class DecisionGenerating(Protocol):
    async def __call__(
        self,
        *,
        input: Multimodal,  # noqa: A002
        questions: Sequence[DecisionQuestion],
        **extra: Any,
    ) -> Sequence[DecisionAnswer]: ...
