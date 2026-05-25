from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import (
    FORMAT_INSTRUCTION,
    extract_evaluation_result,
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
    Evaluate the originality and creative thinking in content.

    This evaluator assesses the degree of creativity, originality, and
    innovative thinking demonstrated in content, including novel ideas,
    unique perspectives, and imaginative approaches.

    Parameters
    ----------
    evaluated : Multimodal
        The content to evaluate for creativity
    guidelines : str | None, optional
        Additional guidelines for creativity evaluation, by default None

    Returns
    -------
    EvaluationScore
        Creativity score with categorical rating and explanation

    Raises
    ------
    ValueError
        When the evaluator fails to parse the result
    """
    if not evaluated:
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
                        "<CONTENT>",
                        evaluated,
                        "</CONTENT>",
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
    """
    if not evaluated:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Input context was empty!"},
        )

    evaluated_content: MultimodalContent = model_context_multimodal(evaluated)

    return extract_evaluation_result(
        await Step.generating_completion(
            instructions=CONTEXT_INSTRUCTION.format(
                guidelines=f"\n<GUIDELINES>\n{guidelines}\n</GUIDELINES>\n" if guidelines else "",
            ),
        ).run(
            (
                ModelInput.of(
                    MultimodalContent.of(
                        "<CONTENT>",
                        evaluated_content,
                        "</CONTENT>",
                    ),
                ),
            )
        )
    )


CONTENT_INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Carefully examine provided CONTENT, then rate it using solely a creativity metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is creativity - the degree of originality, novelty, and innovative thinking demonstrated in the content. Creative content should show original ideas, unique perspectives, imaginative approaches, novel combinations of concepts, or innovative solutions. This includes artistic creativity, problem-solving creativity, conceptual originality, and the ability to think outside conventional patterns. Consider uniqueness of ideas, originality of expression, innovative approaches, and imaginative elements.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Judge originality relative to a competent baseline response to the same prompt, not against literary or commercial masterpieces. "Perfect" does not require genius — it requires a clear, non-obvious creative choice carried through the piece.
Assign a creativity score using exact name of one of the following values:
- "poor" - entirely conventional, formulaic, or trope-driven; no original choice visible.
- "fair" - mostly conventional with at most one mildly original element; the response would be predictable from the prompt alone.
- "good" - some genuinely creative elements mixed with conventional structure; one or two non-obvious choices.
- "excellent" - clearly original framing, perspective, or combination; the response distinguishes itself from a generic answer.
- "perfect" - the piece is built around a strong creative choice (unusual angle, fresh metaphor, unexpected combination) and carries it through; it does not need to be unprecedented to qualify.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the CONTENT conversation timeline. Focus on model-produced results in output elements and rate their degree of creativity and originality.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is creativity of model results in context.
Assess the degree of originality, novelty, and innovative thinking demonstrated in model outputs, including original ideas, unique perspectives, imaginative approaches, and the ability to think beyond conventional patterns within the conversational context.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Judge originality relative to a competent baseline response to the same request, not against literary masterpieces. "Perfect" does not require genius — it requires a clear, non-obvious creative choice carried through.
Assign a creativity score using exact name of one of the following values:
- "poor" - entirely conventional, formulaic, or trope-driven model outputs; no original choice visible.
- "fair" - mostly conventional outputs with at most one mildly original element.
- "good" - outputs show some genuinely creative elements alongside conventional structure.
- "excellent" - outputs use a clearly original framing, perspective, or combination that distinguishes them from a generic answer.
- "perfect" - outputs are built around a strong creative choice and carry it through; need not be unprecedented to qualify.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
