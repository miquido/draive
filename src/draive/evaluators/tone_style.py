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
    "tone_style_context_evaluator",
    "tone_style_evaluator",
)


@evaluator(name="tone_style")
async def tone_style_evaluator(
    evaluated: Multimodal,
    /,
    *,
    expected_tone_style: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate how well content matches expected tone and style.

    This evaluator assesses the tone, style, and voice of content to determine
    how well it matches the expected tone and style characteristics. It considers
    formality level, emotional tone, professionalism, and overall alignment with
    the specified tone/style requirements.

    Parameters
    ----------
    evaluated : Multimodal
        The content to evaluate for tone and style
    expected_tone_style : Multimodal
        The expected tone and style characteristics to match against
    guidelines : str | None, optional
        Additional guidelines for tone/style evaluation, by default None

    Returns
    -------
    EvaluationScore
        Tone and style score with categorical rating and explanation

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

    if not expected_tone_style:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Expected tone/style was empty!"},
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
                        "<EXPECTED_TONE_STYLE>",
                        expected_tone_style,
                        "</EXPECTED_TONE_STYLE>\n<EVALUATED>",
                        evaluated,
                        "</EVALUATED>",
                    ),
                ),
            )
        )
    )


@evaluator(name="tone_style_context")
async def tone_style_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    expected_tone_style: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate tone style using model context.

    Parameters
    ----------
    evaluated : ModelContext
        Evaluator input parameter.
    expected_tone_style : Multimodal
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

    if not expected_tone_style:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Expected tone/style was empty!"},
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
                        "<EXPECTED_TONE_STYLE>",
                        expected_tone_style,
                        "</EXPECTED_TONE_STYLE>\n<EVALUATED>",
                        model_context_multimodal(evaluated),
                        "</EVALUATED>",
                    ),
                ),
            )
        )
    )


CONTENT_INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the EXPECTED_TONE_STYLE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a tone and style metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is tone and style - how well the EVALUATED content matches the EXPECTED_TONE_STYLE in terms of writing tone, style, and voice. This includes assessing the level of formality, emotional tone, professionalism, clarity of voice, consistency of style, and overall alignment with the expected tone and style characteristics. Consider factors like formality level (formal/informal), emotional tone (positive/negative/neutral), politeness, professionalism, and brand voice consistency.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a tone and style score using exact name of one of the following values:
- "poor" is very low tone/style match, the content has tone and style that significantly conflicts with the expected tone/style requirements.
- "fair" is low tone/style match, the content has some alignment with expected tone/style but contains several mismatches or inappropriate elements.
- "good" is moderate tone/style match, the content generally aligns with expected tone/style with some minor inconsistencies or deviations.
- "excellent" is high tone/style match, the content closely matches the expected tone/style with only minimal deviations.
- "perfect" is very high tone/style match, the content perfectly aligns with the expected tone, style, and voice characteristics.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Compare the EXPECTED_TONE_STYLE specification against model-produced results in the EVALUATED conversation timeline. Focus on model outputs and assess how consistently they match the expected tone and style across the entire conversation.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is tone and style alignment of model results in context.
Assess how consistently model outputs match the EXPECTED_TONE_STYLE across the conversation, including formality level, emotional tone, professionalism, clarity of voice, consistency of style, and overall alignment with the expected tone and style characteristics.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a tone and style score using exact name of one of the following values:
- "poor" is very low tone/style match, model outputs have tone and style that significantly conflicts with the expected tone/style requirements.
- "fair" is low tone/style match, model outputs have some alignment with expected tone/style but contain several mismatches or inappropriate elements.
- "good" is moderate tone/style match, model outputs generally align with expected tone/style with some minor inconsistencies or deviations.
- "excellent" is high tone/style match, model outputs closely match the expected tone/style with only minimal deviations.
- "perfect" is very high tone/style match, model outputs perfectly align with the expected tone, style, and voice characteristics throughout.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
