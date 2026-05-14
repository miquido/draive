from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import (
    FORMAT_INSTRUCTION,
    extract_evaluation_result,
    model_context_multimodal,
)
from draive.models import ModelContext, ModelInput
from draive.multimodal import Multimodal, MultimodalContent
from draive.steps import Step


@evaluator(name="truthfulness")
async def truthfulness_evaluator(
    evaluated: Multimodal,
    /,
    *,
    reference: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate truthfulness.

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
    if not evaluated:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Input was empty!"},
        )

    if not reference:
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


@evaluator(name="truthfulness_context")
async def truthfulness_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    reference: Multimodal | None = None,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate truthfulness using model context.

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

    input_content: MultimodalContent = MultimodalContent.of(
        "<EVALUATED>",
        evaluated_content,
        "</EVALUATED>",
    )

    if reference:
        input_content = MultimodalContent.of(
            "<REFERENCE>",
            reference,
            "</REFERENCE>\n<EVALUATED>",
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
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a truthfulness (factual accuracy) metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is truthfulness (factual accuracy) - this metric evaluates the factual correctness of the content, regardless of its relation to the source material. Truthfulness is concerned with whether the information presented is accurate in the broader context, ensuring that the facts are correct and reliable, and that the content does not perpetuate errors or falsehoods. This metric is more concerned with the overall factual integrity of the content, meaning that even if a claim is not directly traceable to the provided source, it must still be correct and represent the truth of the matter. Truthfulness is about the correctness of the facts themselves, not just their alignment with a specific source.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a truthfulness (factual accuracy) score using exact name of one of the following values:
- "poor" is very low truthfulness, the content contains many factual inaccuracies or distortions.
- "fair" is low truthfulness, the content has some accurate information but also significant factual errors or misrepresentations.
- "good" is moderate truthfulness, the content is somewhat accurate but with noticeable factual inaccuracies or distortions.
- "excellent" is high truthfulness, the content is mostly accurate with minimal factual errors or misrepresentations.
- "perfect" is very high truthfulness, the content is fully accurate, with no factual errors or misrepresentations.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements and assess their factual truthfulness.
When REFERENCE is explicitly provided, evaluate model outputs for truthfulness against it; otherwise assess whether model outputs present factually correct and reliable information based on established knowledge.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is truthfulness of model results in context.
Assess the factual correctness of information presented in model outputs, ensuring that facts are correct and reliable and that outputs do not perpetuate errors or falsehoods. Truthfulness concerns the correctness of facts themselves — even claims not directly traceable to a reference must still be accurate.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a truthfulness score using exact name of one of the following values:
- "poor" is very low truthfulness, model outputs contain many factual inaccuracies or distortions.
- "fair" is low truthfulness, model outputs have some accurate information but also significant factual errors or misrepresentations.
- "good" is moderate truthfulness, model outputs are somewhat accurate but with noticeable factual inaccuracies or distortions.
- "excellent" is high truthfulness, model outputs are mostly accurate with minimal factual errors or misrepresentations.
- "perfect" is very high truthfulness, model outputs are fully accurate with no factual errors or misrepresentations.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
