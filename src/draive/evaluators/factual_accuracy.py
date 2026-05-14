from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import (
    FORMAT_INSTRUCTION,
    extract_evaluation_result,
    model_context_multimodal,
)
from draive.models import ModelContext, ModelInput
from draive.multimodal import Multimodal, MultimodalContent
from draive.steps import Step


@evaluator(name="factual_accuracy")
async def factual_accuracy_evaluator(
    evaluated: Multimodal,
    /,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate factual correctness of content based on established knowledge.

    This evaluator assesses the factual accuracy of content by examining
    claims, data, statements, and assertions against well-established facts,
    scientific consensus, and generally accepted knowledge without requiring
    reference material.

    Parameters
    ----------
    evaluated : Multimodal
        The content to evaluate for factual accuracy
    guidelines : str | None, optional
        Additional guidelines for factual accuracy evaluation, by default None

    Returns
    -------
    EvaluationScore
        Factual accuracy score with categorical rating and explanation

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


@evaluator(name="factual_accuracy_context")
async def factual_accuracy_context_evaluator(
    evaluated: ModelContext,
    /,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate factual accuracy using model context.

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
Carefully examine provided CONTENT, then rate it using solely a factual accuracy metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is factual accuracy - the extent to which the content contains factually correct information based on established knowledge, verifiable facts, and reliable sources. This evaluates the correctness of claims, data, statements, and assertions made in the content, regardless of any reference material. The evaluation should consider well-established facts, scientific consensus, historical accuracy, and generally accepted knowledge.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a factual accuracy score using exact name of one of the following values:
- "poor" is very low factual accuracy, the content contains many significant factual errors or false information.
- "fair" is low factual accuracy, the content has several factual inaccuracies or questionable claims mixed with some correct information.
- "good" is moderate factual accuracy, the content is mostly factually correct but contains some minor inaccuracies or unverified claims.
- "excellent" is high factual accuracy, the content is largely factually correct with minimal or very minor factual issues.
- "perfect" is very high factual accuracy, the content is completely factually accurate with all information being correct and verifiable.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the CONTENT conversation timeline. Focus on model-produced results in output elements and assess their factual accuracy based on established knowledge.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is factual accuracy of model results in context.
Assess the correctness of claims, data, statements, and assertions made in model outputs, evaluating them against well-established facts, scientific consensus, and generally accepted knowledge regardless of any reference material.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a factual accuracy score using exact name of one of the following values:
- "poor" is very low factual accuracy, model outputs contain many significant factual errors or false information.
- "fair" is low factual accuracy, model outputs have several factual inaccuracies or questionable claims mixed with some correct information.
- "good" is moderate factual accuracy, model outputs are mostly factually correct but contain some minor inaccuracies or unverified claims.
- "excellent" is high factual accuracy, model outputs are largely factually correct with minimal or very minor factual issues.
- "perfect" is very high factual accuracy, model outputs are completely factually accurate with all information being correct and verifiable.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
