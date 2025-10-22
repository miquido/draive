from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import FORMAT_INSTRUCTION, extract_evaluation_result
from draive.multimodal import Multimodal, MultimodalContent
from draive.stages import Stage

__all__ = ("helpfulness_evaluator",)


INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the USER_QUERY and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a helpfulness metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is helpfulness - the extent to which the EVALUATED content addresses the user's needs, questions, or requests effectively. Helpful content should be relevant to the user's query, provide useful information or solutions, be actionable when appropriate, and demonstrate understanding of what the user is trying to achieve. The content should genuinely assist the user in accomplishing their goal or answering their question.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a helpfulness score using exact name of one of the following values:
- "poor" is very low helpfulness, the content fails to address the user's query or provides irrelevant, unhelpful information.
- "fair" is low helpfulness, the content partially addresses the query but lacks important details or actionable information.
- "good" is moderate helpfulness, the content addresses most of the user's needs but could be more complete or actionable.
- "excellent" is high helpfulness, the content effectively addresses the user's query with relevant, useful information and minor gaps.
- "perfect" is very high helpfulness, the content fully addresses the user's needs with comprehensive, actionable, and highly relevant information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501


@evaluator(name="helpfulness")
async def helpfulness_evaluator(
    evaluated: Multimodal,
    /,
    *,
    user_query: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate how well content addresses user needs and questions.

    This evaluator assesses the helpfulness of content by examining how effectively
    it addresses the user's query, provides useful information, and assists the user
    in accomplishing their goals.

    Parameters
    ----------
    evaluated : Multimodal
        The content to evaluate for helpfulness
    user_query : Multimodal
        The user's original query or request
    guidelines : str | None, optional
        Additional guidelines for helpfulness evaluation, by default None

    Returns
    -------
    EvaluationScore
        Helpfulness score with categorical rating and explanation

    Raises
    ------
    ValueError
        When the evaluator fails to parse the result
    """
    if not evaluated:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Input was empty!"},
        )

    if not user_query:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "User query was empty!"},
        )

    return extract_evaluation_result(
        await Stage.completion(
            MultimodalContent.of(
                "<USER_QUERY>",
                user_query,
                "</USER_QUERY>\n<EVALUATED>",
                evaluated,
                "</EVALUATED>",
            ),
            instructions=INSTRUCTION.format(
                guidelines=f"\n<GUIDELINES>\n{guidelines}\n</GUIDELINES>\n" if guidelines else "",
            ),
        ).execute()
    )
