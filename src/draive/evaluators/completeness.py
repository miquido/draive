from typing import cast

from draive.evaluation import EvaluationScore, EvaluationScoreValue, evaluator
from draive.multimodal import Multimodal, MultimodalContent
from draive.stages import Stage

__all__ = ("completeness_evaluator",)


INSTRUCTION: str = """\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the USER_QUERY and the EVALUATED content by carefully examining them, then rate\
 the EVALUATED content using solely a completeness metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is completeness - the extent to which the EVALUATED content fully\
 addresses and answers all aspects of the USER_QUERY. Complete content should address\
 all parts of multi-part questions, provide comprehensive responses to complex queries,\
 and not leave important aspects of the user's request unanswered.
</EVALUATION_CRITERIA>
{guidelines}
<RATING>
Assign a completeness score using exact name of one of the following values:
- "poor" is very low completeness, the content addresses very few aspects of the\
 user's query, leaving most questions unanswered.
- "fair" is low completeness, the content addresses some aspects of the user's query\
 but leaves several important parts unanswered or incomplete.
- "good" is moderate completeness, the content addresses most aspects of the user's\
 query but may miss some details or minor components.
- "excellent" is high completeness, the content addresses nearly all aspects of the\
 user's query with only minor gaps or omissions.
- "perfect" is very high completeness, the content fully and comprehensively addresses\
 all aspects of the user's query without any significant omissions.
Use the "none" value for content that cannot be rated at all.
</RATING>

<FORMAT>
The final result containing only the rating value, HAVE to be put inside a `RESULT`\
 xml tag within the result i.e. `<RESULT>good</RESULT>`.
</FORMAT>
"""


@evaluator(name="completeness")
async def completeness_evaluator(
    evaluated: Multimodal,
    /,
    *,
    user_query: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate how completely content addresses all aspects of a user query.

    This evaluator assesses whether the content fully answers all parts of
    the user's question or request, ensuring no important aspects are left
    unanswered or incomplete.

    Parameters
    ----------
    evaluated : Multimodal
        The content to evaluate for completeness
    user_query : Multimodal
        The user's original query or request
    guidelines : str | None, optional
        Additional guidelines for completeness evaluation, by default None

    Returns
    -------
    EvaluationScore
        Completeness score with categorical rating and explanation

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

    completion: MultimodalContent = await Stage.completion(
        MultimodalContent.of(
            "<USER_QUERY>",
            user_query,
            "</USER_QUERY>\n<EVALUATED>",
            evaluated,
            "</EVALUATED>",
        ),
        instructions=INSTRUCTION.format(
            guidelines=f"\n<GUIDELINES>\n{guidelines}\n</GUIDELINES>\n"
            if guidelines is not None
            else ""
        ),
    ).execute()

    if result := completion.tag("RESULT"):
        return EvaluationScore.of(
            cast(EvaluationScoreValue, result.content.to_str().lower()),
            meta={"comment": completion.to_str()},
        )

    else:
        raise ValueError(f"Invalid evaluator result:\n{completion}")
