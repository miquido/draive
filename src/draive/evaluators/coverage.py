from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import (
    FORMAT_INSTRUCTION,
    extract_evaluation_result,
    model_context_multimodal,
)
from draive.models import ModelContext, ModelInput
from draive.multimodal import Multimodal, MultimodalContent
from draive.steps import Step


@evaluator(name="coverage")
async def coverage_evaluator(
    evaluated: Multimodal,
    /,
    *,
    reference: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate coverage.

    Parameters
    ----------
    evaluated : Multimodal
        Evaluator input parameter.
    reference : Multimodal
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

    if not reference:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Reference was empty!"},
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
                        "<REFERENCE>",
                        reference,
                        "</REFERENCE>\n<EVALUATED>",
                        evaluated,
                        "</EVALUATED>",
                    ),
                ),
            )
        )
    )


@evaluator(name="coverage_context")
async def coverage_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    reference: Multimodal | None = None,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate coverage using model context.

    Parameters
    ----------
    evaluated : ModelContext
        Evaluator input parameter.
    reference : Multimodal | None
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

    evaluated_content: MultimodalContent = model_context_multimodal(evaluated)

    input_content: MultimodalContent = MultimodalContent.of(
        "<EVALUATED>",
        evaluated_content,
        "</EVALUATED>",
    )

    if reference:
        input_content = MultimodalContent.of(
            "<REFERENCE>",
            reference,
            "</REFERENCE>\n<EVALUATED>",
            evaluated_content,
            "</EVALUATED>",
        )

    return extract_evaluation_result(
        await Step.generating_completion(
            instructions=CONTEXT_INSTRUCTION.format(
                guidelines=f"\n<GUIDELINES>\n{guidelines}\n</GUIDELINES>\n" if guidelines else "",
            ),
        ).run((ModelInput.of(input_content),))
    )


CONTENT_INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a coverage metric according to the EVALUATION_CRITERIA.
Before scoring, extract the key points from the REFERENCE as a numbered checklist of distinct facts, claims, or arguments — focus on points a knowledgeable reader would consider essential, not every minor detail. For each item, mark it as fully covered, partially covered, or absent in the EVALUATED content. Score from this checklist. Do not penalize the EVALUATED content for omitting trivia or for paraphrasing differently — coverage is about substance, not wording.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is coverage - the extent to which the EVALUATED content includes all the key points from the REFERENCE content.
EVALUATED content with good coverage includes all the important information from the REFERENCE content without omitting critical points.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Anchor the score to the fraction of essential checklist items fully covered; count a partially-covered item as half. Different wording that conveys the same information is full coverage.
Assign a coverage score using exact name of one of the following values:
- "poor" - fewer than about a quarter of essential points are covered; most of the reference is absent.
- "fair" - roughly a quarter to half of essential points are covered; several important ones missing or mentioned only in passing.
- "good" - roughly half to three-quarters of essential points covered; a few important details missing.
- "excellent" - nearly all essential points covered substantively; only minor or peripheral items missing.
- "perfect" - every essential point from the reference is represented in the evaluated content, regardless of paraphrasing.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements and assess whether they cover all key points from the source material.
When REFERENCE is explicitly provided, evaluate model outputs for coverage of its key points; otherwise assess whether outputs address all key points raised by user inputs across the context.
Before scoring, extract the essential points from the source material as a numbered checklist, then mark each as fully covered, partially covered, or absent in the model outputs. Score from this checklist. Do not penalize for omitting trivia or for paraphrasing differently.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is coverage of model results in context.
Assess whether model outputs include all important information from the source material (reference or user-provided context) without omitting critical points that were expected to be addressed.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Anchor the score to the fraction of essential checklist items fully covered; count a partially-covered item as half. Different wording that conveys the same information is full coverage.
Assign a coverage score using exact name of one of the following values:
- "poor" - fewer than about a quarter of essential points are covered.
- "fair" - roughly a quarter to half of essential points covered; several important ones missing.
- "good" - roughly half to three-quarters of essential points covered; a few important details missing.
- "excellent" - nearly all essential points covered substantively; only minor or peripheral items missing.
- "perfect" - every essential point is represented in the model outputs, regardless of paraphrasing.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
