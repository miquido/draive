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
    Evaluate tone and style.

    Parameters
    ----------
    evaluated : Multimodal
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
    if is_empty_content(evaluated):
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Input was empty!"},
        )

    if is_empty_content(expected_tone_style):
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
    Evaluate tone and style using model context.

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

    if is_empty_content(expected_tone_style):
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Expected tone/style was empty!"},
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
                        "<EXPECTED_TONE_STYLE>",
                        expected_tone_style,
                        "</EXPECTED_TONE_STYLE>\n<EVALUATED>",
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
Compare the EXPECTED_TONE_STYLE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a tone and style metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is tone and style - how well the EVALUATED content matches the EXPECTED_TONE_STYLE in writing tone, style, and voice. This covers formality level (formal/informal), emotional tone (positive/negative/neutral), politeness, professionalism, clarity of voice, consistency of style, and brand-voice alignment.
Judge only tone and style: do not reward or penalize factual accuracy, coverage, or length on their own.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Before scoring, identify the two or three load-bearing dimensions of EXPECTED_TONE_STYLE (e.g. formality, warmth, register, brand voice, rhythm) and judge the match on those dimensions specifically. Treat an isolated misstep as a deviation, not a failure. "perfect" does not require the content to read as if written by the same author as the spec - it requires consistent alignment on the load-bearing dimensions.
Assign a tone and style score using exact name of one of the following values:
- "poor" - reads in a tone fundamentally at odds with the spec (e.g. wry where reverent was asked, casual where formal was asked); the mismatch is felt throughout.
- "fair" - gestures at the requested voice, but breaks into the wrong register repeatedly; a reader would notice the mismatch.
- "good" - broadly matches the requested voice, with a few clear misses (one wrong word choice, one sentence in the wrong register, one missing characteristic flourish).
- "excellent" - matches the requested voice on every load-bearing dimension, with one or two subtle deviations a careful reader might notice.
- "perfect" - consistent alignment on every load-bearing dimension; no register breaks; no jarring word choices.
Use the "none" value only for content that cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Compare the EXPECTED_TONE_STYLE specification against the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a tone and style metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is tone and style alignment of model results in context.
Assess how consistently model outputs match the EXPECTED_TONE_STYLE across the conversation, including formality level, emotional tone, politeness, professionalism, clarity of voice, consistency of style, and brand-voice alignment.
Judge only tone and style: do not reward or penalize factual accuracy, coverage, or length on their own.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Before scoring, identify the two or three load-bearing dimensions of EXPECTED_TONE_STYLE (e.g. formality, warmth, register, brand voice, rhythm) and judge the match on those dimensions specifically. Treat an isolated misstep as a deviation, not a failure.
Assign a tone and style score using exact name of one of the following values:
- "poor" - outputs read in a tone fundamentally at odds with the spec; the mismatch is felt throughout the conversation.
- "fair" - outputs gesture at the requested voice, but break into the wrong register repeatedly.
- "good" - outputs broadly match the requested voice, with a few clear misses across the conversation.
- "excellent" - outputs match the requested voice on every load-bearing dimension, with one or two subtle deviations.
- "perfect" - consistent alignment on every load-bearing dimension across the conversation; no register breaks.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
