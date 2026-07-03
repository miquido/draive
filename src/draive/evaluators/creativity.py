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


@evaluator(name="creativity")
async def creativity_evaluator(
    evaluated: Multimodal,
    /,
    *,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate creativity.

    Parameters
    ----------
    evaluated : Multimodal
        Evaluator input parameter.
    guidelines : str | None
        Evaluator input parameter.

    Returns
    -------
    EvaluationScore
        Evaluation result.

    Raises
    ------
    ValueError
        If the model produced an invalid or malformed evaluation result.
    """
    if is_empty_content(evaluated):
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Input was empty!"},
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
                        "</EVALUATED>",
                    ),
                ),
            )
        )
    )


@evaluator(name="creativity_context")
async def creativity_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate creativity using model context.

    Parameters
    ----------
    evaluated : ModelContext
        Evaluator input parameter.
    guidelines : str | None
        Evaluator input parameter.

    Returns
    -------
    EvaluationScore
        Evaluation result.

    Raises
    ------
    ValueError
        If the model produced an invalid or malformed evaluation result.
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
Carefully examine the EVALUATED content, then rate it using solely a creativity metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is creativity - the degree of originality, novelty, and innovative thinking demonstrated in the EVALUATED content.
Creative content shows original ideas, unique perspectives, imaginative approaches, novel combinations of concepts, or inventive solutions, whether artistic, conceptual, or problem-solving. Consider uniqueness of ideas, originality of expression, and the ability to think beyond conventional patterns.
Judge only creativity: do not reward or penalize factual accuracy, fluency, or length on their own.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Judge originality relative to a competent baseline response to the same prompt, not against literary or commercial masterpieces. "perfect" does not require genius - it requires a clear, non-obvious creative choice carried through the piece.
Assign a creativity score using exact name of one of the following values:
- "poor" - entirely conventional, formulaic, or trope-driven; no original choice visible.
- "fair" - mostly conventional with at most one mildly original element; predictable from the prompt alone.
- "good" - some genuinely creative elements alongside conventional structure; one or two non-obvious choices.
- "excellent" - clearly original framing, perspective, or combination that distinguishes the content from a generic answer.
- "perfect" - built around a strong creative choice (unusual angle, fresh metaphor, unexpected combination) and carried through; need not be unprecedented to qualify.
Use the "none" value only for content that cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a creativity metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is creativity of model results in context.
Assess the degree of originality, novelty, and innovative thinking demonstrated in model outputs - original ideas, unique perspectives, imaginative approaches, and the ability to think beyond conventional patterns within the conversational context.
Judge only creativity: do not reward or penalize factual accuracy, fluency, or length on their own.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Judge originality relative to a competent baseline response to the same request, not against literary or commercial masterpieces. "perfect" does not require genius - it requires a clear, non-obvious creative choice carried through.
Assign a creativity score using exact name of one of the following values:
- "poor" - entirely conventional, formulaic, or trope-driven outputs; no original choice visible.
- "fair" - mostly conventional outputs with at most one mildly original element; predictable from the request alone.
- "good" - outputs show some genuinely creative elements alongside conventional structure; one or two non-obvious choices.
- "excellent" - outputs use a clearly original framing, perspective, or combination that distinguishes them from a generic answer.
- "perfect" - outputs are built around a strong creative choice and carry it through; need not be unprecedented to qualify.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
