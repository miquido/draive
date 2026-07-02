from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import (
    FORMAT_INSTRUCTION,
    extract_evaluation_result,
    is_empty_content,
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
    if is_empty_content(evaluated):
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Input was empty!"},
        )

    if is_empty_content(expectations):
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
                        "<EXPECTATIONS>",
                        expectations,
                        "</EXPECTATIONS>\n<EVALUATED>",
                        evaluated,
                        "</EVALUATED>",
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

    if is_empty_content(expectations):
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Expectations was empty!"},
        )

    evaluated_content: MultimodalContent = model_context_multimodal(evaluated)

    if is_empty_content(evaluated_content):
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Input context was empty!"},
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
                        "<EXPECTATIONS>",
                        expectations,
                        "</EXPECTATIONS>\n<EVALUATED>",
                        evaluated_content,
                        "</EVALUATED>",
                    ),
                ),
            )
        )
    )


CONTENT_INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the EXPECTATIONS and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely an expectations fulfilment metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is expectations fulfilment - the extent to which the EVALUATED content satisfies the criteria, requirements, and points defined in EXPECTATIONS.
Before scoring, enumerate every distinct expectation, criterion, or constraint stated in EXPECTATIONS as a numbered checklist, and mark each as fully met, partially met, or unmet in the EVALUATED content. Score from this checklist, not from overall impression: brevity that still satisfies every stated expectation is full fulfilment.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Anchor the score to the fraction of stated expectations fully met, counting a partially-met one as half.
Assign an expectation fulfilment score using exact name of one of the following values:
- "poor" - misses most key points from the expectations.
- "fair" - addresses some key points, but omits several important ones.
- "good" - covers most key points, but misses a few important details.
- "excellent" - satisfies nearly all key points, with minor omissions.
- "perfect" - comprehensively satisfies all key points from the expectations.
Use the "none" value only for content that cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline alongside the defined EXPECTATIONS. Focus on model-produced results in output elements, then rate them using solely an expectations fulfilment metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is expectations fulfilment of model results in context.
Assess whether model outputs satisfy the criteria and requirements defined in EXPECTATIONS, considering the full scope of the conversation.
Before scoring, enumerate every distinct expectation, criterion, or constraint stated in EXPECTATIONS as a numbered checklist, and mark each as fully met, partially met, or unmet by the model outputs. Score from this checklist, not from overall impression: brevity that still satisfies every stated expectation is full fulfilment.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Anchor the score to the fraction of stated expectations fully met, counting a partially-met one as half.
Assign an expectation fulfilment score using exact name of one of the following values:
- "poor" - outputs miss most key points from the expectations.
- "fair" - outputs address some expectations, but omit several important ones.
- "good" - outputs cover most expectations, but miss a few important details.
- "excellent" - outputs satisfy nearly all expectations, with minor omissions.
- "perfect" - outputs comprehensively satisfy all expectations.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
