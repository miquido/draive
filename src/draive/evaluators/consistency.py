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
    if is_empty_content(evaluated):
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Input was empty!"},
        )

    if is_empty_content(reference):
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

    if is_empty_content(evaluated_content):
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Input context was empty!"},
        )

    instruction: str
    input_content: MultimodalContent
    if reference and not is_empty_content(reference):
        instruction = CONTEXT_REFERENCE_INSTRUCTION
        input_content = MultimodalContent.of(
            "<REFERENCE>",
            reference,
            "</REFERENCE>\n<EVALUATED>",
            evaluated_content,
            "</EVALUATED>",
        )

    else:
        instruction = CONTEXT_INSTRUCTION
        input_content = MultimodalContent.of(
            "<EVALUATED>",
            evaluated_content,
            "</EVALUATED>",
        )

    return extract_evaluation_result(
        await Step.generating_completion(
            instructions=instruction.format(
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
Evaluated metric is consistency - the factual alignment between the REFERENCE and the EVALUATED content. Factually consistent content contains only elements that are entailed by the REFERENCE content, without contradictions or unsupported additions.
Do not treat absence of claims as perfect consistency: content that makes no assessable claims cannot be rated (see "none").
Judge only consistency against the REFERENCE: do not reward or penalize style, coverage, or fluency on their own.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a consistency score using exact name of one of the following values:
- "poor" - multiple hallucinated facts or significant misalignments with the reference.
- "fair" - several instances of information not supported by the reference.
- "good" - mostly consistent, but with a few unsupported statements.
- "excellent" - largely consistent, with minor discrepancies.
- "perfect" - fully consistent with the reference, containing only supported information.
Use the "none" value only for content that cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, placeholders, or content with no intelligible, assessable claims.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_REFERENCE_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a consistency metric against the REFERENCE according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is consistency of model results in context.
Assess whether model outputs contain only factual elements entailed by the REFERENCE, without introducing contradictions, unsupported claims, or information that conflicts with it.
Do not treat silence or absence of claims as perfect consistency: outputs that make no assessable claims cannot be rated (see "none").
Judge only consistency against the REFERENCE: do not reward or penalize style, coverage, or fluency on their own.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a consistency score using exact name of one of the following values:
- "poor" - multiple contradictions or hallucinated facts not supported by the reference.
- "fair" - several instances of information inconsistent with the reference.
- "good" - mostly consistent, but with a few unsupported statements.
- "excellent" - largely consistent, with minor discrepancies.
- "perfect" - fully consistent, containing only supported information.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, placeholders, or no intelligible, assessable claims.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a consistency metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is consistency of model results in context.
Assess whether model outputs contain only factual elements entailed by the established context, without introducing contradictions, unsupported claims, or information that conflicts with what was previously stated.
Do not treat silence or absence of claims as perfect consistency: outputs that make no assessable claims cannot be rated (see "none").
Judge only consistency against the context: do not reward or penalize style, coverage, or fluency on their own.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a consistency score using exact name of one of the following values:
- "poor" - multiple contradictions or hallucinated facts not supported by the context.
- "fair" - several instances of information inconsistent with the context.
- "good" - mostly consistent, but with a few unsupported statements.
- "excellent" - largely consistent, with minor discrepancies.
- "perfect" - fully consistent, containing only supported information.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, placeholders, or no intelligible, assessable claims.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
