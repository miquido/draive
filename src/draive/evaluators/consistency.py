from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import (
    FORMAT_INSTRUCTION,
    extract_evaluation_result,
    model_context_multimodal,
)
from draive.models import ModelContext, ModelInput
from draive.multimodal import Multimodal, MultimodalContent
from draive.steps import Step


@evaluator(name="consistency")
async def consistency_evaluator(
    evaluated: Multimodal,
    /,
    *,
    reference: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate consistency.

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


@evaluator(name="consistency_context")
async def consistency_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    reference: Multimodal | None = None,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate consistency using model context.

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
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a consistency metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is consistency - a factual alignment between the REFERENCE and the EVALUATED content.
A factually consistent content contains only elements that are entailed by the REFERENCE content.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a consistency score using exact name of one of the following values:
- "poor" is very low consistency, the content contains multiple hallucinated facts or significant misalignments with the reference content.
- "fair" is low consistency, the content has several instances of information not supported by the reference content.
- "good" is moderate consistency, the content is mostly consistent but contains a few unsupported statements.
- "excellent" is high consistency, the content is largely consistent with minor discrepancies.
- "perfect" is very high consistency, the content is fully consistent with the reference content, containing only supported information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements and assess whether they are factually consistent with established information.
When REFERENCE is explicitly provided, evaluate model outputs for consistency against it; otherwise assess whether outputs are internally consistent with facts and information established earlier in the context.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is consistency of model results in context.
Assess whether model outputs contain only factual elements entailed by the reference material or the established context, without introducing contradictions, unsupported claims, or information that conflicts with what was previously stated.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a consistency score using exact name of one of the following values:
- "poor" is very low consistency, model outputs contain multiple contradictions or hallucinated facts not supported by the context.
- "fair" is low consistency, model outputs have several instances of information inconsistent with the context.
- "good" is moderate consistency, model outputs are mostly consistent but contain a few unsupported statements.
- "excellent" is high consistency, model outputs are largely consistent with minor discrepancies.
- "perfect" is very high consistency, model outputs are fully consistent, containing only supported information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
