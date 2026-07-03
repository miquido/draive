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


@evaluator(name="coherence")
async def coherence_evaluator(
    evaluated: Multimodal,
    /,
    *,
    reference: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate coherence.

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
                    )
                ),
            )
        )
    )


@evaluator(name="coherence_context")
async def coherence_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    reference: Multimodal | None = None,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate coherence using model context.

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
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a coherence metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is coherence - the collective structural quality of the EVALUATED content, aligned with the DUC (Document Understanding Conference) quality question of structure and coherence: the content should be well-structured and well-organized.
The content should not be a heap of related information but should build from part to part into a coherent body of information about the topic, with logical connections and smooth transitions. Treat the REFERENCE as supplemental material describing the expected well-structured progression, and use it as a benchmark for structural alignment.
Judge only coherence: do not reward or penalize factual accuracy or coverage on their own.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a coherence score using exact name of one of the following values:
- "poor" - chaotic; lacking logical connections between parts.
- "fair" - some connections are visible, but the overall structure is weak.
- "good" - a noticeable structure, but with some shortcomings.
- "excellent" - well-organized, with minor imperfections.
- "perfect" - exemplarily structured, with smooth transitions between ideas.
Use the "none" value only for content that cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_REFERENCE_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a coherence metric against the REFERENCE according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is coherence of model results in context.
Assess whether model outputs form a logically connected and well-structured progression from available contextual information (especially prior inputs and prior outputs). Treat the REFERENCE as a benchmark for the expected well-structured progression. Outputs should not be chaotic, contradictory, or disconnected from what was established in the context timeline.
Judge only coherence: do not reward or penalize factual accuracy or coverage on their own.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a coherence score using exact name of one of the following values:
- "poor" - outputs are chaotic or largely disconnected from context.
- "fair" - some context alignment exists but structure is weak.
- "good" - outputs are mostly coherent with a few structural gaps.
- "excellent" - outputs are well-organized and context-aligned, with minor imperfections.
- "perfect" - outputs are consistently well-structured and fully coherent within context.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a coherence metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is coherence of model results in context.
Assess whether model outputs form a logically connected and well-structured progression from available contextual information (especially prior inputs and prior outputs). Outputs should not be chaotic, contradictory, or disconnected from what was established in the context timeline.
Judge only coherence: do not reward or penalize factual accuracy or coverage on their own.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a coherence score using exact name of one of the following values:
- "poor" - outputs are chaotic or largely disconnected from context.
- "fair" - some context alignment exists but structure is weak.
- "good" - outputs are mostly coherent with a few structural gaps.
- "excellent" - outputs are well-organized and context-aligned, with minor imperfections.
- "perfect" - outputs are consistently well-structured and fully coherent within context.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
