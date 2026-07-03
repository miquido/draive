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


@evaluator(name="relevance")
async def relevance_evaluator(
    evaluated: Multimodal,
    /,
    *,
    reference: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate relevance.

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


@evaluator(name="relevance_context")
async def relevance_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    reference: Multimodal | None = None,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate relevance using model context.

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
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a relevance metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is relevance - the selection of important parts from the REFERENCE content. The EVALUATED content should include only important information from the REFERENCE, avoiding redundancies and excess information.
Judge only relevance: assess topical pertinence and the absence of redundant or off-topic material. Do not penalize brevity, limited depth, or incomplete coverage - a short but fully on-topic response is highly relevant.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a relevance score using exact name of one of the following values:
- "poor" - mostly irrelevant or redundant information.
- "fair" - some important points, but significant irrelevant parts.
- "good" - covers most important points, but includes some unnecessary information.
- "excellent" - focuses on important information, with minor inclusions of less relevant content.
- "perfect" - precisely captures only the most important information from the reference.
Use the "none" value only for content that cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_REFERENCE_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a relevance metric against the REFERENCE according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is relevance of model results in context.
Assess whether model outputs include only information pertinent to the REFERENCE, avoiding unnecessary digressions, redundancies, and content that does not serve what the REFERENCE establishes as important.
Judge only relevance: assess topical pertinence and the absence of redundant or off-topic material. Do not penalize brevity, limited depth, or incomplete coverage - a short but fully on-topic response is highly relevant.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a relevance score using exact name of one of the following values:
- "poor" - mostly irrelevant or redundant information relative to the reference.
- "fair" - some important points, but significant irrelevant parts.
- "good" - covers most important points, but includes some unnecessary information.
- "excellent" - focuses on important information, with minor inclusions of less relevant content.
- "perfect" - precisely captures only the most important information from the reference.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a relevance metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is relevance of model results in context.
Assess whether model outputs include only pertinent information addressing the user's queries, avoiding unnecessary digressions, redundancies, and content that does not serve the user's intent established in the conversation.
Judge only relevance: assess topical pertinence and the absence of redundant or off-topic material. Do not penalize brevity, limited depth, or incomplete coverage - a short but fully on-topic response is highly relevant.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a relevance score using exact name of one of the following values:
- "poor" - mostly irrelevant or redundant information relative to user queries.
- "fair" - some important points, but significant irrelevant parts.
- "good" - covers most important points, but includes some unnecessary information.
- "excellent" - focuses on important information, with minor inclusions of less relevant content.
- "perfect" - precisely addresses the user's queries with only the most pertinent information.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
