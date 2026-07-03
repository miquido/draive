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


@evaluator(name="readability")
async def readability_evaluator(
    evaluated: Multimodal,
    /,
    *,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate readability.

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


@evaluator(name="readability_context")
async def readability_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate readability using model context.

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
Carefully examine the EVALUATED content, then rate it using solely a readability metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is readability - the ease with which a reader can understand the EVALUATED content.
Readable content uses clear, concise language, is well-structured, and avoids convoluted sentences, unexplained jargon, and disorganized presentation. Judge readability for the content's own apparent audience; dense specialist jargon aimed at a general or lay audience is a readability defect.
Judge only readability: do not reward or penalize factual accuracy or relevance on their own.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a readability score using exact name of one of the following values:
- "poor" - extremely difficult to understand; complex language and convoluted structure throughout.
- "fair" - challenging to read; frequent complex sentences, unclear language, or disorganized parts.
- "good" - somewhat clear but with some areas that are difficult to understand.
- "excellent" - mostly clear and easy to read, with minor instances of complexity.
- "perfect" - highly clear, concise, and easy to understand throughout.
Use the "none" value only for content that cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a readability metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is readability of model results in context.
Assess the ease with which a reader can understand model outputs across the conversation, considering clarity of language, logical structure, appropriate formatting, and avoidance of convoluted or jargon-heavy passages. Judge readability for the outputs' own apparent audience.
Judge only readability: do not reward or penalize factual accuracy or relevance on their own.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a readability score using exact name of one of the following values:
- "poor" - outputs are extremely difficult to understand; complex language and convoluted structure throughout.
- "fair" - outputs are challenging to read; frequent complex sentences, unclear language, or disorganized parts.
- "good" - outputs are somewhat clear but with some areas that are difficult to understand.
- "excellent" - outputs are mostly clear and easy to read, with minor instances of complexity.
- "perfect" - outputs are highly clear, concise, and easy to understand throughout.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
