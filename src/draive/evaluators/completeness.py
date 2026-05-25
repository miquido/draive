from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import (
    FORMAT_INSTRUCTION,
    extract_evaluation_result,
    model_context_multimodal,
)
from draive.models import ModelContext, ModelInput
from draive.multimodal import Multimodal, MultimodalContent
from draive.steps import Step


@evaluator(name="completeness")
async def completeness_evaluator(
    evaluated: Multimodal,
    /,
    *,
    user_query: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate how completely content addresses all aspects of a user query.

    This evaluator assesses whether the content fully answers all parts of
    the user's question or request, ensuring no important aspects are left
    unanswered or incomplete.

    Parameters
    ----------
    evaluated : Multimodal
        The content to evaluate for completeness
    user_query : Multimodal
        The user's original query or request
    guidelines : str | None, optional
        Additional guidelines for completeness evaluation, by default None

    Returns
    -------
    EvaluationScore
        Completeness score with categorical rating and explanation

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


@evaluator(name="completeness_context")
async def completeness_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    user_query: Multimodal | None = None,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate completeness using model context.

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

    input_content: MultimodalContent = MultimodalContent.of(
        "<EVALUATED>",
        evaluated_content,
        "</EVALUATED>",
    )

    if user_query:
        input_content = MultimodalContent.of(
            "<USER_QUERY>",
            user_query,
            "</USER_QUERY>\n<EVALUATED>",
            evaluated_content,
            "</EVALUATED>",
        )

    return extract_evaluation_result(
        await Step.generating_completion(
            instructions=CONTEXT_INSTRUCTION.format(
                guidelines=f"\n<GUIDELINES>\n{guidelines}\n</GUIDELINES>\n" if guidelines else "",
            ),
        ).run((ModelInput.of(input_content),))
    )


CONTENT_INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the USER_QUERY and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a completeness metric according to the EVALUATION_CRITERIA.
Before scoring, enumerate every distinct sub-question, requirement, or explicit constraint in the USER_QUERY as a numbered checklist, and mark each item as fully addressed, partially addressed, or unaddressed in the EVALUATED content. Base the rating on this checklist; do not score on overall impression or surface fluency. A confidently written response that skips a sub-question is incomplete.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is completeness - the extent to which the EVALUATED content fully addresses and answers all aspects of the USER_QUERY. Complete content should address all parts of multi-part questions, provide comprehensive responses to complex queries, and not leave important aspects of the user's request unanswered.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Anchor the score to the fraction of distinct checklist items fully addressed; count a partially-addressed item as half. Be strict: tone or thoroughness on covered items does not compensate for skipped items.
Assign a completeness score using exact name of one of the following values:
- "poor" - fewer than about a quarter of the items are fully addressed; most of the query is unanswered.
- "fair" - roughly a quarter to half of the items are addressed; several important parts are missing or only superficially touched.
- "good" - roughly half to three-quarters of the items are addressed; a few important sub-questions remain missing or shallow.
- "excellent" - nearly all items addressed substantively; only minor or peripheral elements are missing.
- "perfect" - every distinct sub-question, requirement, and constraint in the query receives a substantive, on-topic answer; nothing material is omitted.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements and assess whether they completely address all aspects of user queries or requests present in the context.
When USER_QUERY is explicitly provided, use it as the primary query to evaluate completeness against; otherwise infer the user's full intent and scope from the context itself.
Before scoring, enumerate every distinct sub-question, requirement, or explicit constraint requested as a numbered checklist, and mark each item as fully addressed, partially addressed, or unaddressed by the model outputs. Base the rating on this checklist; do not score on overall impression or surface fluency.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is completeness of model results in context.
Assess whether model outputs fully address all parts of the user's request, including multi-part questions, specific sub-tasks, and implicit requirements expressed in the conversation, leaving no important aspects unanswered.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Anchor the score to the fraction of distinct checklist items fully addressed; count a partially-addressed item as half. Be strict: tone or thoroughness on covered items does not compensate for skipped items.
Assign a completeness score using exact name of one of the following values:
- "poor" - fewer than about a quarter of items fully addressed; most of the request is unanswered.
- "fair" - roughly a quarter to half of items addressed; several important parts missing or superficially touched.
- "good" - roughly half to three-quarters of items addressed; a few important items missing or shallow.
- "excellent" - nearly all items addressed substantively; only minor or peripheral elements missing.
- "perfect" - every distinct sub-question, sub-task, and constraint receives a substantive, on-topic answer; nothing material is omitted.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
