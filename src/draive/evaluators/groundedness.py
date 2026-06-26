from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import (
    FORMAT_INSTRUCTION,
    extract_evaluation_result,
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

    instruction: str
    input_content: MultimodalContent
    if reference:
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
Evaluated metric is groundedness - this metric assesses the extent to which the evaluated content directly ties back to and is anchored in the source data. Grounded content should demonstrate a clear and traceable connection to the provided source material, ensuring that the information presented is not only accurate but also faithfully represents the original context. This metric focuses on how well the content reflects the source material without introducing extraneous information, unsupported claims, or interpretations that stray from the source. Groundedness is about maintaining fidelity to the original data, ensuring that every detail and conclusion is rooted in the provided information.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a groundedness score using exact name of one of the following values:
- "poor" is very low groundedness, the content is mostly ungrounded with many unsupported claims.
- "fair" is low groundedness, the content contains some accurate information but also significant ungrounded content.
- "good" is moderate groundedness, the content is somewhat grounded but with noticeable ungrounded elements.
- "excellent" is high groundedness, the content is mostly grounded with minimal unverified or unsupported claims.
- "perfect" is very high groundedness, the content is fully grounded, accurately reflecting the source information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_REFERENCE_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements and assess how well they are grounded in the REFERENCE.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is groundedness of model results in context.
Assess the extent to which model outputs directly tie back to and are anchored in the provided REFERENCE, ensuring information is not only accurate but faithfully represents the reference without introducing extraneous claims, unsupported interpretations, or hallucinated details.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a groundedness score using exact name of one of the following values:
- "poor" is very low groundedness, model outputs are mostly ungrounded with many unsupported or fabricated claims.
- "fair" is low groundedness, model outputs contain some grounded information but also significant ungrounded content.
- "good" is moderate groundedness, model outputs are somewhat grounded but with noticeable ungrounded elements.
- "excellent" is high groundedness, model outputs are mostly grounded with minimal unverified or unsupported claims.
- "perfect" is very high groundedness, model outputs are fully grounded, accurately reflecting the reference information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements and assess whether they are anchored in and traceable to information established within the conversation context.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is groundedness of model results in context.
Assess the extent to which model outputs directly tie back to and are anchored in information established within the conversation context (especially prior inputs and prior outputs), ensuring information is not only accurate but faithfully represents the context without introducing extraneous claims, unsupported interpretations, or hallucinated details.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a groundedness score using exact name of one of the following values:
- "poor" is very low groundedness, model outputs are mostly ungrounded with many unsupported or fabricated claims.
- "fair" is low groundedness, model outputs contain some grounded information but also significant ungrounded content.
- "good" is moderate groundedness, model outputs are somewhat grounded but with noticeable ungrounded elements.
- "excellent" is high groundedness, model outputs are mostly grounded with minimal unverified or unsupported claims.
- "perfect" is very high groundedness, model outputs are fully grounded, accurately reflecting the contextual information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
