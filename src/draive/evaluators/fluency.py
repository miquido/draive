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
Carefully examine the EVALUATED content, then rate it using solely a fluency metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is fluency - the linguistic quality of the EVALUATED content in terms of grammar, spelling, punctuation, word choice, and natural sentence structure.
Grammatical validity alone is not fluency: stilted phrasing, register mismatches, run-ons, and clunky word order are fluency defects, not stylistic choices. Count concrete issues (awkward phrasings, agreement errors, dropped articles, register breaks) before scoring rather than relying on surface plausibility.
Judge only fluency: do not reward or penalize factual accuracy, relevance, or content choices on their own.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Anchor the score to concrete language-quality signals, not surface plausibility.
Assign a fluency score using exact name of one of the following values:
- "poor" - multiple grammar, spelling, or agreement errors per paragraph; ESL-quality awkwardness; or text that is hard to parse on first read.
- "fair" - frequent stilted constructions, run-ons, dropped articles, or register breaks; comprehensible but visibly non-native or unedited.
- "good" - mostly readable native-speaker prose with several clunky phrasings, occasional awkward word order, or one minor grammar slip; passable but would not survive a copy-edit.
- "excellent" - natural native-speaker prose with one or two minor awkward spots a copy-editor would smooth.
- "perfect" - no awkward phrasings, grammar issues, or register breaks anywhere; reads as if professionally edited.
Use the "none" value only for content that cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a fluency metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is fluency of model results in context.
Assess the linguistic quality of model outputs in terms of grammar, spelling, punctuation, word choice, and natural sentence structure across the conversation.
Grammatical validity alone is not fluency: stilted phrasing, register mismatches, run-ons, and clunky word order are fluency defects. Count concrete issues before scoring rather than relying on surface plausibility.
Judge only fluency: do not reward or penalize factual accuracy, relevance, or content choices on their own.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Anchor the score to concrete language-quality signals, not surface plausibility.
Assign a fluency score using exact name of one of the following values:
- "poor" - multiple grammar or agreement errors per output; ESL-quality awkwardness; outputs that are hard to parse on first read.
- "fair" - frequent stilted constructions, run-ons, dropped articles, or register breaks; comprehensible but visibly unedited.
- "good" - mostly readable native-speaker prose with several clunky phrasings or one minor grammar slip; passable but unpolished.
- "excellent" - natural native-speaker prose with one or two minor awkward spots a copy-editor would smooth.
- "perfect" - no awkward phrasings, grammar issues, or register breaks anywhere; reads as if professionally edited.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
