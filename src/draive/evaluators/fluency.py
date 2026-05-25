from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import (
    FORMAT_INSTRUCTION,
    extract_evaluation_result,
    model_context_multimodal,
)
from draive.models import ModelContext, ModelInput
from draive.multimodal import Multimodal, MultimodalContent
from draive.steps import Step


@evaluator(name="fluency")
async def fluency_evaluator(
    evaluated: Multimodal,
    /,
    *,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate fluency.

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


@evaluator(name="fluency_context")
async def fluency_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate fluency using model context.

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
Carefully examine provided CONTENT, then rate it using solely a fluency metric according to the EVALUATION_CRITERIA.
Do not default to a high score for content that is grammatically valid but stiff, awkward, or unnatural. Stilted phrasing, register mismatches, run-ons, and clunky word order are all fluency defects, not stylistic choices. Count concrete issues (awkward phrasings, agreement errors, dropped articles, register breaks) before scoring.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is fluency - the quality of the content in terms of grammar, spelling, punctuation, content choice, and overall structure.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Anchor the score to concrete language-quality signals, not surface plausibility.
Assign a fluency score using exact name of one of the following values:
- "poor" - multiple grammar/spelling/agreement errors per paragraph; ESL-quality awkwardness; or text that is hard to parse on first read.
- "fair" - frequent stilted constructions, run-ons, dropped articles, or register breaks; comprehensible but visibly non-native or unedited.
- "good" - mostly readable native-speaker prose with several clunky phrasings, occasional awkward word order, or one minor grammar slip; passable but would not survive a copy-edit.
- "excellent" - reads as natural native-speaker prose with one or two minor awkward spots that a copy-editor would smooth.
- "perfect" - no awkward phrasings, no grammar issues, no register breaks anywhere; reads as if professionally edited.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the CONTENT conversation timeline. Focus on model-produced results in output elements and rate their language fluency.
Do not default to a high score for outputs that are grammatically valid but stiff, awkward, or unnatural. Stilted phrasing, register mismatches, run-ons, and clunky word order are all fluency defects.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is fluency of model results in context.
Assess the quality of model outputs in terms of grammar, spelling, punctuation, word choice, and overall linguistic naturalness across the conversation.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Anchor the score to concrete language-quality signals, not surface plausibility.
Assign a fluency score using exact name of one of the following values:
- "poor" - multiple grammar or agreement errors per output; ESL-quality awkwardness; outputs that are hard to parse on first read.
- "fair" - frequent stilted constructions, run-ons, dropped articles, or register breaks; comprehensible but visibly unedited.
- "good" - mostly readable native-speaker prose with several clunky phrasings or one minor grammar slip; passable but unpolished.
- "excellent" - natural native-speaker prose with one or two minor awkward spots that a copy-editor would smooth.
- "perfect" - no awkward phrasings, no grammar issues, no register breaks anywhere; reads as if professionally edited.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
