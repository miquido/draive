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
Assign a creativity score using exact name of one of the following values:
- "poor" is very low creativity, the content is entirely conventional, generic, or lacks any original or novel elements.
- "fair" is low creativity, the content shows minimal originality with mostly conventional ideas and approaches.
- "good" is moderate creativity, the content demonstrates some original thinking and creative elements mixed with conventional approaches.
- "excellent" is high creativity, the content shows significant originality, innovative thinking, and creative approaches with minor conventional elements.
- "perfect" is very high creativity, the content is highly original, innovative, and demonstrates exceptional creative thinking throughout.
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
Assign a creativity score using exact name of one of the following values:
- "poor" is very low creativity, model outputs are entirely conventional, generic, or lack any original or novel elements.
- "fair" is low creativity, model outputs show minimal originality with mostly conventional ideas and approaches.
- "good" is moderate creativity, model outputs demonstrate some original thinking and creative elements mixed with conventional approaches.
- "excellent" is high creativity, model outputs show significant originality and innovative thinking with minor conventional elements.
- "perfect" is very high creativity, model outputs are highly original, innovative, and demonstrate exceptional creative thinking throughout.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
