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
Evaluated metric is relevance - selection of important parts from the REFERENCE content.
The EVALUATED content should include only important information from the REFERENCE avoiding redundancies and excess information.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a relevance score using exact name of one of the following values:
- "poor" is very low relevance, the content contains mostly irrelevant or redundant information.
- "fair" is low relevance, the content includes some important points but has significant irrelevant parts.
- "good" is moderate relevance, the content covers most important points but includes some unnecessary information.
- "excellent" is high relevance, the content focuses on important information with minor inclusions of less relevant content.
- "perfect" is very high relevance, the content precisely captures only the most important information from the reference.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_REFERENCE_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements and assess their relevance against the REFERENCE.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is relevance of model results in context.
Assess whether model outputs include only information pertinent to the REFERENCE, avoiding unnecessary digressions, redundancies, and content that does not serve what the REFERENCE establishes as important.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a relevance score using exact name of one of the following values:
- "poor" is very low relevance, model outputs contain mostly irrelevant or redundant information relative to the reference.
- "fair" is low relevance, model outputs include some important points but have significant irrelevant parts.
- "good" is moderate relevance, model outputs cover most important points but include some unnecessary information.
- "excellent" is high relevance, model outputs focus on important information with minor inclusions of less relevant content.
- "perfect" is very high relevance, model outputs precisely capture only the most important information from the reference.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements and assess their relevance to the user's queries and intent.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is relevance of model results in context.
Assess whether model outputs include only pertinent information addressing the user's queries, avoiding unnecessary digressions, redundancies, and content that does not serve the user's intent established in the conversation.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a relevance score using exact name of one of the following values:
- "poor" is very low relevance, model outputs contain mostly irrelevant or redundant information relative to user queries.
- "fair" is low relevance, model outputs include some important points but have significant irrelevant parts.
- "good" is moderate relevance, model outputs cover most important points but include some unnecessary information.
- "excellent" is high relevance, model outputs focus on important information with minor inclusions of less relevant content.
- "perfect" is very high relevance, model outputs precisely address the user's queries with only the most pertinent information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
