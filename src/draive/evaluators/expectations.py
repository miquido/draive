from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import FORMAT_INSTRUCTION, extract_evaluation_result
from draive.multimodal import Multimodal, MultimodalContent
from draive.stages import Stage

__all__ = ("expectations_evaluator",)


INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Understand the EXPECTATIONS for the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely the metric of expectations fulfillment according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is defined by fulfilling expectations and criteria defined within EXPECTATIONS.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign an expectation fulfillment score using exact name of one of the following values:
- "poor" is very low expectation fulfilment - the content misses most key points from the expectation.
- "fair" is low expectation fulfilment - the content includes some key points but omits several important ones.
- "good" is moderate expectation fulfilment - the content covers most key points but misses a few important details.
- "excellent" is high expectation fulfilment - the content includes nearly all key points with minor omissions.
- "perfect" is very high expectation fulfilment - the content comprehensively covers all key points from the expectations.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
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

    return extract_evaluation_result(
        await Stage.completion(
            MultimodalContent.of(
                "<EVALUATED>",
                evaluated,
                "</EVALUATED>\n<EXPECTATIONS>",
                expectations,
                "</EXPECTATIONS>",
            ),
            instructions=INSTRUCTION.format(
                guidelines=f"\n<GUIDELINES>\n{guidelines}\n</GUIDELINES>\n" if guidelines else "",
            ),
        ).execute()
    )
