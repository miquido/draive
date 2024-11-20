from typing import cast

from draive.evaluation import EvaluationScore, EvaluationScoreValue, evaluator
from draive.multimodal import Multimodal, MultimodalContent, MultimodalTagElement
from draive.steps import steps_completion

__all__ = [
    "coverage_evaluator",
]


INSTRUCTION: str = """\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate\
 the EVALUATED content using solely a coverage metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is coverage - the extent to which the EVALUATED content includes all\
 the key points from the REFERENCE content.
EVALUATED content with good coverage includes all the important information from\
 the REFERENCE content without omitting critical points.
</EVALUATION_CRITERIA>

<RATING>
Assign a coverage score using exact name of one of the following values:
- "poor" is very low coverage - the content misses most key points from the reference content.
- "fair" is low coverage - the content includes some key points but omits several important ones.
- "good" is moderate coverage - the content covers most key points but misses a few important details.
- "excellent" is high coverage - the content includes nearly all key points with minor omissions.
- "perfect" is very high coverage - the content comprehensively covers all key points from the reference content.
Use the "none" value for content that cannot be rated at all.
</RATING>

<FORMAT>
The final result containing only the numerical score value HAVE to be put inside a `RESULT`\
 xml tag within the result i.e. `<RESULT>good</RESULT>`.
</FORMAT>
"""  # noqa: E501


@evaluator(name="coverage")
async def coverage_evaluator(
    evaluated: Multimodal,
    /,
    *,
    reference: Multimodal,
) -> EvaluationScore:
    if not evaluated:
        return EvaluationScore(
            value=0,
            comment="Input was empty!",
        )

    if not reference:
        return EvaluationScore(
            value=0,
            comment="Reference was empty!",
        )

    completion: MultimodalContent = await steps_completion(
        MultimodalContent.of(
            "<REFERENCE>",
            reference,
            "</REFERENCE>\n<EVALUATED>",
            evaluated,
            "</EVALUATED>",
        ),
        instruction=INSTRUCTION,
    )

    if result := MultimodalTagElement.parse_first(
        completion,
        tag="RESULT",
    ):
        return EvaluationScore.of(
            cast(EvaluationScoreValue, result.content.as_string()),
            comment=completion.as_string(),
        )

    else:
        raise ValueError("Invalid evaluator result")
