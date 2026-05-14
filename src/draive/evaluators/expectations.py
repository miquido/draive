from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import (
    FORMAT_INSTRUCTION,
    extract_evaluation_result,
    model_context_multimodal,
)
from draive.models import ModelContext, ModelInput
from draive.multimodal import Multimodal, MultimodalContent
from draive.steps import Step

__all__ = (
    "expectations_context_evaluator",
    "expectations_evaluator",
)


@evaluator(name="expectations")
async def expectations_evaluator(
    evaluated: Multimodal,
    /,
    *,
    expectations: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate expectations.

    Parameters
    ----------
    evaluated : Multimodal
        Evaluator input parameter.
    expectations : Multimodal
        Evaluator input parameter.
    guidelines : str | None
        Evaluator input parameter.

    Returns
    -------
    EvaluationScore
        Evaluation result.
    """
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
        await Step.generating_completion(
            instructions=CONTENT_INSTRUCTION.format(
                guidelines=f"\n<GUIDELINES>\n{guidelines}\n</GUIDELINES>\n" if guidelines else "",
            ),
        ).run(
            (
                ModelInput.of(
                    MultimodalContent.of(
                        "<EVALUATED>",
                        evaluated,
                        "</EVALUATED>\n<EXPECTATIONS>",
                        expectations,
                        "</EXPECTATIONS>",
                    ),
                ),
            )
        )
    )


@evaluator(name="expectations_context")
async def expectations_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    expectations: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate expectations using model context.

    Parameters
    ----------
    evaluated : ModelContext
        Evaluator input parameter.
    expectations : Multimodal
        Evaluator input parameter.
    guidelines : str | None
        Evaluator input parameter.

    Returns
    -------
    EvaluationScore
        Evaluation result.
    """
    if not evaluated:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Input context was empty!"},
        )

    if not expectations:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Expectations was empty!"},
        )

    return extract_evaluation_result(
        await Step.generating_completion(
            instructions=CONTEXT_INSTRUCTION.format(
                guidelines=f"\n<GUIDELINES>\n{guidelines}\n</GUIDELINES>\n" if guidelines else "",
            ),
        ).run(
            (
                ModelInput.of(
                    MultimodalContent.of(
                        "<EVALUATED>",
                        model_context_multimodal(evaluated),
                        "\n</EVALUATED>\n<EXPECTATIONS>",
                        expectations,
                        "</EXPECTATIONS>",
                    ),
                ),
            )
        )
    )


CONTENT_INSTRUCTION: str = f"""\
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

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline alongside the defined EXPECTATIONS. Focus on model-produced results in output elements and assess how well they fulfil the specified expectations across the entire conversation.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is expectations fulfilment of model results in context.
Assess whether model outputs satisfy the criteria and requirements defined in EXPECTATIONS, considering the full scope of the conversation.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign an expectation fulfillment score using exact name of one of the following values:
- "poor" is very low expectation fulfilment, model outputs miss most key points from the expectations.
- "fair" is low expectation fulfilment, model outputs address some expectations but omit several important ones.
- "good" is moderate expectation fulfilment, model outputs cover most expectations but miss a few important details.
- "excellent" is high expectation fulfilment, model outputs satisfy nearly all expectations with minor omissions.
- "perfect" is very high expectation fulfilment, model outputs comprehensively satisfy all expectations.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
