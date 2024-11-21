from typing import cast

from draive.evaluation import EvaluationScore, EvaluationScoreValue, evaluator
from draive.multimodal import Multimodal, MultimodalContent, MultimodalTagElement
from draive.steps import steps_completion

__all__ = [
    "consistency_evaluator",
]


INSTRUCTION: str = """\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate\
 the EVALUATED content using solely a consistency metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is consistency - a factual alignment between the REFERENCE and the EVALUATED content.
A factually consistent content contains only elements that are entailed by the REFERENCE content.
</EVALUATION_CRITERIA>

<RATING>
Assign a consistency score using exact name of one of the following values:
- "poor" is very low consistency, the content contains multiple hallucinated facts\
 or significant misalignments with the reference content.
- "fair" is low consistency, the content has several instances of information not supported by\
 the reference content.
- "good" is moderate consistency, the content is mostly consistent but contains a few unsupported\
 statements.
- "excellent" is high consistency, the content is largely consistent with minor discrepancies.
- "perfect" is very high consistency, the content is fully consistent with the reference content,\
 containing only supported information.
Use the "none" value for content that cannot be rated at all.
</RATING>

<FORMAT>
The final result containing only the numerical score value HAVE to be put inside a `RESULT`\
 xml tag within the result i.e. `<RESULT>good</RESULT>`.
</FORMAT>
"""  # noqa: E501


@evaluator(name="consistency")
async def consistency_evaluator(
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
