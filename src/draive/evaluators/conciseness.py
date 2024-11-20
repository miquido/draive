from typing import cast

from draive.evaluation import EvaluationScore, EvaluationScoreValue, evaluator
from draive.multimodal import Multimodal, MultimodalContent, MultimodalTagElement
from draive.steps import steps_completion

__all__ = [
    "conciseness_evaluator",
]


INSTRUCTION: str = """\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate\
 the EVALUATED content using solely a conciseness metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is conciseness - a extent to which the EVALUATED content is brief and to the\
 point while still covering all key information.
A concise content avoids unnecessary details and repetition, also avoiding being overly verbose\
 or include irrelevant information.
</EVALUATION_CRITERIA>

<RATING>
Assign a conciseness score using exact name of one of the following values:
- "poor" is very low conciseness, the content is excessively verbose with much irrelevant\
 information.
- "fair" is low conciseness, the content contains unnecessary details and some irrelevant\
 information.
- "good" is moderate conciseness, the content is somewhat concise but could be more focused.
- "excellent" is high conciseness, the content is mostly concise with minimal unnecessary\
 information.
- "perfect" is very high conciseness, the content is highly concise, containing only essential\
 information.
Use the "none" value for content that cannot be rated at all.
</RATING>

<FORMAT>
The final result containing only the numerical score value HAVE to be put inside a `RESULT`\
 xml tag within the result i.e. `<RESULT>good</RESULT>`.
</FORMAT>
"""


@evaluator(name="conciseness")
async def conciseness_evaluator(
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
