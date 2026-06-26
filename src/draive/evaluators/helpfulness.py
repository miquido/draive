from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import (
    FORMAT_INSTRUCTION,
    extract_evaluation_result,
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
    Evaluate how well content addresses user needs and questions.

    This evaluator assesses the helpfulness of content by examining how effectively
    it addresses the user's query, provides useful information, and assists the user
    in accomplishing their goals.

    Parameters
    ----------
    evaluated : Multimodal
        The content to evaluate for helpfulness
    user_query : Multimodal
        The user's original query or request
    guidelines : str | None, optional
        Additional guidelines for helpfulness evaluation, by default None

    Returns
    -------
    EvaluationScore
        Helpfulness score with categorical rating and explanation

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

    if not user_query:
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

    instruction: str
    input_content: MultimodalContent
    if user_query:
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
Evaluated metric is helpfulness - the extent to which the EVALUATED content addresses the user's needs, questions, or requests effectively. Helpful content should be relevant to the user's query, provide useful information or solutions, be actionable when appropriate, and demonstrate understanding of what the user is trying to achieve. The content should genuinely assist the user in accomplishing their goal or answering their question.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a helpfulness score using exact name of one of the following values:
- "poor" is very low helpfulness, the content fails to address the user's query or provides irrelevant, unhelpful information.
- "fair" is low helpfulness, the content partially addresses the query but lacks important details or actionable information.
- "good" is moderate helpfulness, the content addresses most of the user's needs but could be more complete or actionable.
- "excellent" is high helpfulness, the content effectively addresses the user's query with relevant, useful information and minor gaps.
- "perfect" is very high helpfulness, the content fully addresses the user's needs with comprehensive, actionable, and highly relevant information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_QUERY_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements and assess how effectively they address the user's needs, using the provided USER_QUERY as the primary query to evaluate helpfulness against.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is helpfulness of model results in context.
Assess the extent to which model outputs effectively address the provided USER_QUERY — providing relevant, useful, and actionable information that genuinely assists the user in accomplishing their goals.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a helpfulness score using exact name of one of the following values:
- "poor" is very low helpfulness, model outputs fail to address the user's queries or provide irrelevant, unhelpful information.
- "fair" is low helpfulness, model outputs partially address the queries but lack important details or actionable information.
- "good" is moderate helpfulness, model outputs address most of the user's needs but could be more complete or actionable.
- "excellent" is high helpfulness, model outputs effectively address the user's queries with relevant, useful information and minor gaps.
- "perfect" is very high helpfulness, model outputs fully address the user's needs with comprehensive, actionable, and highly relevant information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements and assess how well they serve the user's intent inferred from the context.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is helpfulness of model results in context.
Assess the extent to which model outputs effectively address the user's needs, questions, or requests inferred from the conversation — providing relevant, useful, and actionable information that genuinely assists the user in accomplishing their goals.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a helpfulness score using exact name of one of the following values:
- "poor" is very low helpfulness, model outputs fail to address the user's queries or provide irrelevant, unhelpful information.
- "fair" is low helpfulness, model outputs partially address the queries but lack important details or actionable information.
- "good" is moderate helpfulness, model outputs address most of the user's needs but could be more complete or actionable.
- "excellent" is high helpfulness, model outputs effectively address the user's queries with relevant, useful information and minor gaps.
- "perfect" is very high helpfulness, model outputs fully address the user's needs with comprehensive, actionable, and highly relevant information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
