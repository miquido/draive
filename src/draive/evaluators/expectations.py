from typing import cast

from draive.evaluation import EvaluationScore, EvaluationScoreValue, evaluator
from draive.multimodal import Multimodal, MultimodalContent, MultimodalTagElement
from draive.stages import Stage

__all__ = ("expectations_evaluator",)


INSTRUCTION: str = """\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Understand the EXPECTATIONS for the EVALUATED content by carefully examining them, then rate\
 the EVALUATED content using solely the metric of expectations fulfillment  according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is defined by fulfilling expectations and criteria defined within EXPECTATIONS.
</EVALUATION_CRITERIA>
{guidelines}
<RATING>
Assign an expectation fulfillment score using exact name of one of the following values:
- "poor" is very low expectation fulfilment - the content misses most key points from the expectation.
- "fair" is low expectation fulfilment - the content includes some key points but omits several important ones.
- "good" is moderate expectation fulfilment - the content covers most key points but misses a few important details.
- "excellent" is high expectation fulfilment - the content includes nearly all key points with minor omissions.
- "perfect" is very high expectation fulfilment - the content comprehensively covers all key points from the expectations.
Use the "none" value for content that cannot be rated at all.
</RATING>

<FORMAT>
The final result containing only the rating value, HAVE to be put inside a `RESULT`\
 xml tag within the result i.e. `<RESULT>good</RESULT>`.
</FORMAT>
"""  # noqa: E501


@evaluator(name="expectations")
async def expectations_evaluator(
    evaluated: Multimodal,
    /,
    *,
    expectations: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    if not evaluated:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Input was empty!"},
        )

    if not expectations:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Expectations was empty!"},
        )

    completion: MultimodalContent = await Stage.completion(
        MultimodalContent.of(
            "<EVALUATED>",
            evaluated,
            "</EVALUATED>\n<EXPECTATIONS>",
            expectations,
            "</EXPECTATIONS>",
        ),
        instruction=INSTRUCTION.format(
            guidelines=f"\n<GUIDELINES>\n{guidelines}\n</GUIDELINES>\n"
            if guidelines is not None
            else ""
        ),
    ).execute()

    if result := MultimodalTagElement.parse_first(
        "RESULT",
        content=completion,
    ):
        return EvaluationScore.of(
            cast(EvaluationScoreValue, result.content.to_str()),
            meta={"comment": completion.to_str()},
        )

    else:
        raise ValueError(f"Invalid evaluator result:\n{completion}")
