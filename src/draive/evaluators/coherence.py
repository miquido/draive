from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import (
    FORMAT_INSTRUCTION,
    extract_evaluation_result,
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
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a coherence metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is coherence - a collective quality of the content.
We align this dimension with the DUC (Document Understanding Conference) quality question of structure and coherence, whereby the content should be well-structured and well-organized.
EVALUATED content should not just be a heap of related information, but should build from part to part into a coherent body of information about the topic.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a coherence score using exact name of one of the following values:
- "poor" is very low coherence, the content is chaotic, lacking logical connections between parts.
- "fair" is low coherence, some connections are visible, but the overall structure is weak.
- "good" is moderate coherence, the content has a noticeable structure, but with some shortcomings.
- "excellent" is high coherence, the content is well-organized with minor imperfections.
- "perfect" is very high coherence, the content is exemplarily structured, with smooth transitions between ideas.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the CONTEXT timeline. Focus on model-produced results in output elements and judge whether they are coherent with information available in the context itself.
When present, treat REFERENCE as supplemental material that can help validate alignment, but do not make REFERENCE mandatory for scoring.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is coherence of model results in context.
Assess whether outputs form a logically connected and well-structured progression from available contextual information (especially prior inputs and prior outputs).
Outputs should not be chaotic, contradictory, or disconnected from what was established in the context timeline.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a coherence score using exact name of one of the following values:
- "poor" is very low coherence, outputs are chaotic or largely disconnected from context.
- "fair" is low coherence, some context alignment exists but structure is weak.
- "good" is moderate coherence, outputs are mostly coherent with a few structural gaps.
- "excellent" is high coherence, outputs are well-organized and context-aligned with minor imperfections.
- "perfect" is very high coherence, outputs are consistently well-structured and fully coherent within context.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
