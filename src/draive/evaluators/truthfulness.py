from typing import cast

from draive.evaluation import EvaluationScore, EvaluationScoreValue, evaluator
from draive.multimodal import Multimodal, MultimodalContent, MultimodalTagElement
from draive.steps import steps_completion

__all__ = [
    "truthfulness_evaluator",
]


INSTRUCTION: str = """\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate\
 the EVALUATED content using solely a truthfulness (factual accuracy) metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is truthfulness (factual accuracy) - this metric evaluates the factual correctness\
 of the content, regardless of its relation to the source material. Truthfulness is concerned with whether\
 the information presented is accurate in the broader context, ensuring that the facts are correct and reliable,\
 and that the content does not perpetuate errors or falsehoods. This metric is more concerned with the overall\
 factual integrity of the content, meaning that even if a claim is not directly traceable to the provided source,\
 it must still be correct and represent the truth of the matter. Truthfulness is about the correctness of the\
 facts themselves, not just their alignment with a specific source.
</EVALUATION_CRITERIA>

<RATING>
Assign a truthfulness (factual accuracy) score using exact name of one of the following values:
- "poor" is very low truthfulness, the content contains many factual inaccuracies or distortions.
- "fair" is low truthfulness, the content has some accurate information but also significant factual errors or misrepresentations.
- "good" is moderate truthfulness, the content is somewhat accurate but with noticeable factual inaccuracies or distortions.
- "excellent" is high truthfulness, the content is mostly accurate with minimal factual errors or misrepresentations.
- "perfect" is very high truthfulness, the content is fully accurate, with no factual errors or misrepresentations.
Use the "none" value for content that cannot be rated at all.
</RATING>

<FORMAT>
The final result containing only the numerical score value HAVE to be put inside a `RESULT`\
 xml tag within the result i.e. `<RESULT>good</RESULT>`.
</FORMAT>
"""  # noqa: E501


@evaluator(name="truthfulness")
async def truthfulness_evaluator(
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
