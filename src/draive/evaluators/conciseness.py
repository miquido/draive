from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import (
    FORMAT_INSTRUCTION,
    extract_evaluation_result,
    model_context_multimodal,
)
from draive.models import ModelContext, ModelInput
from draive.multimodal import Multimodal, MultimodalContent
from draive.steps import Step


@evaluator(name="conciseness")
async def conciseness_evaluator(
    evaluated: Multimodal,
    /,
    *,
    reference: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate conciseness.

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


@evaluator(name="conciseness_context")
async def conciseness_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    reference: Multimodal | None = None,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate conciseness using model context.

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
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a conciseness metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is conciseness — the extent to which the EVALUATED content is brief and to the point while still covering all key information.
Concise content avoids unnecessary details and repetition, while not being overly verbose or including irrelevant information.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a conciseness score using exact name of one of the following values:
- "poor" is very low conciseness, the content is excessively verbose with much irrelevant information.
- "fair" is low conciseness, the content contains unnecessary details and some irrelevant information.
- "good" is moderate conciseness, the content is somewhat concise but could be more focused.
- "excellent" is high conciseness, the content is mostly concise with minimal unnecessary information.
- "perfect" is very high conciseness, the content is highly concise, containing only essential information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements and assess how concise and focused they are.
When REFERENCE is explicitly provided, use it as the conciseness benchmark; otherwise assess model outputs for unnecessary verbosity relative to what the user queries required.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is conciseness of model results in context.
Assess whether model outputs are brief and to the point while covering all key information, avoiding unnecessary details, repetition, and verbose elaboration beyond what the conversational context requires.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a conciseness score using exact name of one of the following values:
- "poor" is very low conciseness, model outputs are excessively verbose with much irrelevant information.
- "fair" is low conciseness, model outputs contain unnecessary details and some irrelevant information.
- "good" is moderate conciseness, model outputs are somewhat concise but could be more focused.
- "excellent" is high conciseness, model outputs are mostly concise with minimal unnecessary information.
- "perfect" is very high conciseness, model outputs are highly concise, containing only essential information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
