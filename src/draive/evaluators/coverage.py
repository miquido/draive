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


@evaluator(name="coverage")
async def coverage_evaluator(
    evaluated: Multimodal,
    /,
    *,
    reference: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate coverage.

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


@evaluator(name="coverage_context")
async def coverage_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    reference: Multimodal | None = None,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate coverage using model context.

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
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a coverage metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is coverage - the extent to which the EVALUATED content includes all the key points from the REFERENCE content, without omitting critical points.
Before scoring, extract the key points from the REFERENCE as a numbered checklist of distinct facts, claims, or arguments a knowledgeable reader would consider essential (not every minor detail), and mark each as fully covered, partially covered, or absent in the EVALUATED content. Score from this checklist. Different wording that conveys the same information is full coverage; do not penalize omitting trivia or paraphrasing differently.
Brevity is not itself a coverage failure: when the EVALUATED content is a much shorter summary or brief than the REFERENCE, it is expected to compress or drop secondary statistics, methodology, subgroup detail, and caveats, and that compression alone must not push the score down.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Anchor the score to the fraction of essential checklist items fully covered, counting a partially-covered item as half. For a much shorter summary or brief, do not let the raw fraction over a long checklist set the score by itself: first check whether the primary conclusion(s) and any decisive caveat are conveyed. A brief that conveys those while only omitting secondary statistics, subgroup detail, or methodology should score "good" or better regardless of how many secondary items it skips - that is a floor, not a ceiling; if it also captures those secondary specifics in condensed form, score "excellent" or "perfect". Reserve "fair" or lower for a brief that misrepresents, distorts, or omits the primary conclusion(s) themselves. This leniency does not apply when the REFERENCE is a flat list or enumeration of distinct, comparably important items (e.g. named people, entries, records) - there each missing named item is a plain omission and the raw fraction covered drives the score.
Assign a coverage score using exact name of one of the following values:
- "poor" - fewer than about a quarter of essential points are covered; most of the reference is absent.
- "fair" - roughly a quarter to half of essential points covered; several important ones missing or mentioned only in passing.
- "good" - roughly half to three-quarters of essential points covered; a few important details missing.
- "excellent" - nearly all essential points covered substantively; only minor or peripheral items missing.
- "perfect" - every essential point from the reference is represented, regardless of paraphrasing.
Use the "none" value only for content that cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_REFERENCE_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a coverage metric against the REFERENCE according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is coverage of model results in context.
Assess whether model outputs include all important information from the REFERENCE without omitting critical points that were expected to be addressed.
Before scoring, extract the essential points from the REFERENCE as a numbered checklist, then mark each as fully covered, partially covered, or absent in the model outputs. Score from this checklist. Different wording that conveys the same information is full coverage; do not penalize omitting trivia or paraphrasing differently.
Brevity is not itself a coverage failure: when the model outputs are a much shorter summary or brief than the REFERENCE, they are expected to compress or drop secondary statistics, methodology, subgroup detail, and caveats, and that compression alone must not push the score down.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Anchor the score to the fraction of essential checklist items fully covered, counting a partially-covered item as half. For much shorter outputs, first check whether the primary conclusion(s) and any decisive caveat are conveyed: outputs that convey those while omitting only secondary detail should score "good" or better, unless the primary conclusion(s) themselves are misrepresented or omitted. This leniency does not apply when the REFERENCE is a flat enumeration of distinct, comparably important items - there each missing item is a plain omission and the raw fraction covered drives the score.
Assign a coverage score using exact name of one of the following values:
- "poor" - fewer than about a quarter of essential points are covered.
- "fair" - roughly a quarter to half of essential points covered; several important ones missing.
- "good" - roughly half to three-quarters of essential points covered; a few important details missing.
- "excellent" - nearly all essential points covered substantively; only minor or peripheral items missing.
- "perfect" - every essential point is represented in the model outputs, regardless of paraphrasing.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a coverage metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is coverage of model results in context.
Assess whether model outputs address all key points raised by user inputs across the context, without omitting critical points that were expected to be addressed.
Before scoring, extract the essential points raised by the user inputs as a numbered checklist, then mark each as fully covered, partially covered, or absent in the model outputs. Score from this checklist. Different wording that conveys the same information is full coverage; do not penalize omitting trivia or paraphrasing differently.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Anchor the score to the fraction of essential checklist items fully covered, counting a partially-covered item as half.
Assign a coverage score using exact name of one of the following values:
- "poor" - fewer than about a quarter of essential points are covered.
- "fair" - roughly a quarter to half of essential points covered; several important ones missing.
- "good" - roughly half to three-quarters of essential points covered; a few important details missing.
- "excellent" - nearly all essential points covered substantively; only minor or peripheral items missing.
- "perfect" - every essential point is represented in the model outputs, regardless of paraphrasing.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
