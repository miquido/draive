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


@evaluator(name="helpfulness")
async def helpfulness_evaluator(
    evaluated: Multimodal,
    /,
    *,
    user_query: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate helpfulness.

    Parameters
    ----------
    evaluated : Multimodal
        Evaluator input parameter.
    user_query : Multimodal
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

    if is_empty_content(user_query):
        return EvaluationScore.of(
            0.0,
            meta={"comment": "User query was empty!"},
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
                        "<USER_QUERY>",
                        user_query,
                        "</USER_QUERY>\n<EVALUATED>",
                        evaluated,
                        "</EVALUATED>",
                    ),
                ),
            )
        )
    )


@evaluator(name="helpfulness_context")
async def helpfulness_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    user_query: Multimodal | None = None,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate helpfulness using model context.

    Parameters
    ----------
    evaluated : ModelContext
        Evaluator input parameter.
    user_query : Multimodal | None
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
    if user_query and not is_empty_content(user_query):
        instruction = CONTEXT_QUERY_INSTRUCTION
        input_content = MultimodalContent.of(
            "<USER_QUERY>",
            user_query,
            "</USER_QUERY>\n<EVALUATED>",
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
Compare the USER_QUERY and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a helpfulness metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is helpfulness - the extent to which the EVALUATED content addresses the user's needs, questions, or requests effectively. Helpful content is relevant to the USER_QUERY, provides useful information or solutions, is actionable when appropriate, and demonstrates understanding of what the user is trying to achieve.
Judge helpfulness by substance, not length or confidence: a fluent response that provides no usable information is not helpful.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a helpfulness score using exact name of one of the following values:
- "poor" - fails to address the user's query, or provides irrelevant, unhelpful information.
- "fair" - partially addresses the query, but lacks important details or actionable information.
- "good" - addresses most of the user's needs, but could be more complete or actionable.
- "excellent" - effectively addresses the query with relevant, useful information and only minor gaps.
- "perfect" - fully addresses the user's needs with comprehensive, actionable, and highly relevant information.
Use the "none" value only for content that cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_QUERY_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a helpfulness metric against the provided USER_QUERY according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is helpfulness of model results in context.
Assess the extent to which model outputs effectively address the provided USER_QUERY, providing relevant, useful, and actionable information that genuinely assists the user in accomplishing their goals.
Judge helpfulness by substance, not length or confidence: a fluent response that provides no usable information is not helpful.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a helpfulness score using exact name of one of the following values:
- "poor" - outputs fail to address the user's queries, or provide irrelevant, unhelpful information.
- "fair" - outputs partially address the queries, but lack important details or actionable information.
- "good" - outputs address most of the user's needs, but could be more complete or actionable.
- "excellent" - outputs effectively address the queries with relevant, useful information and only minor gaps.
- "perfect" - outputs fully address the user's needs with comprehensive, actionable, and highly relevant information.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a helpfulness metric according to the EVALUATION_CRITERIA, judging against the user's intent inferred from the context.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is helpfulness of model results in context.
Assess the extent to which model outputs effectively address the user's needs, questions, or requests inferred from the conversation, providing relevant, useful, and actionable information that genuinely assists the user in accomplishing their goals.
Judge helpfulness by substance, not length or confidence: a fluent response that provides no usable information is not helpful.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a helpfulness score using exact name of one of the following values:
- "poor" - outputs fail to address the user's queries, or provide irrelevant, unhelpful information.
- "fair" - outputs partially address the queries, but lack important details or actionable information.
- "good" - outputs address most of the user's needs, but could be more complete or actionable.
- "excellent" - outputs effectively address the queries with relevant, useful information and only minor gaps.
- "perfect" - outputs fully address the user's needs with comprehensive, actionable, and highly relevant information.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
