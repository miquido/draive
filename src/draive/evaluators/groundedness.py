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


@evaluator(name="groundedness")
async def groundedness_evaluator(
    evaluated: Multimodal,
    /,
    *,
    reference: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate groundedness.

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


@evaluator(name="groundedness_context")
async def groundedness_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    reference: Multimodal | None = None,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate groundedness using model context.

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
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a groundedness metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is groundedness - the extent to which the EVALUATED content is anchored in and traceable to the REFERENCE source material. Grounded content faithfully reflects the source without introducing extraneous information, unsupported claims, or interpretations that stray from it; every detail and conclusion should be rooted in the provided information.
Judge only groundedness: do not reward or penalize coverage, style, or fluency on their own. Correctly-paraphrased content that adds no unsupported claims is fully grounded, even when it omits parts of the reference.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a groundedness score using exact name of one of the following values:
- "poor" - mostly ungrounded, with many unsupported claims.
- "fair" - some accurate information, but also significant ungrounded content.
- "good" - somewhat grounded, but with noticeable ungrounded elements.
- "excellent" - mostly grounded, with minimal unverified or unsupported claims.
- "perfect" - fully grounded, accurately reflecting the source information.
Use the "none" value only for content that cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_REFERENCE_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a groundedness metric against the REFERENCE according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is groundedness of model results in context.
Assess the extent to which model outputs are anchored in and traceable to the provided REFERENCE, faithfully representing it without introducing extraneous claims, unsupported interpretations, or hallucinated details.
Judge only groundedness: do not reward or penalize coverage, style, or fluency on their own. Correctly-paraphrased outputs that add no unsupported claims are fully grounded, even when they omit parts of the reference.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a groundedness score using exact name of one of the following values:
- "poor" - outputs are mostly ungrounded, with many unsupported or fabricated claims.
- "fair" - some grounded information, but also significant ungrounded content.
- "good" - somewhat grounded, but with noticeable ungrounded elements.
- "excellent" - mostly grounded, with minimal unverified or unsupported claims.
- "perfect" - fully grounded, accurately reflecting the reference information.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a groundedness metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is groundedness of model results in context.
Assess the extent to which model outputs are anchored in and traceable to information established within the conversation context (especially prior inputs and prior outputs), faithfully representing it without introducing extraneous claims, unsupported interpretations, or hallucinated details.
Judge only groundedness: do not reward or penalize coverage, style, or fluency on their own. Correctly-paraphrased outputs that add no unsupported claims are fully grounded.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a groundedness score using exact name of one of the following values:
- "poor" - outputs are mostly ungrounded, with many unsupported or fabricated claims.
- "fair" - some grounded information, but also significant ungrounded content.
- "good" - somewhat grounded, but with noticeable ungrounded elements.
- "excellent" - mostly grounded, with minimal unverified or unsupported claims.
- "perfect" - fully grounded, accurately reflecting the contextual information.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
