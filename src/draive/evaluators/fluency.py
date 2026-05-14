from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import (
    FORMAT_INSTRUCTION,
    extract_evaluation_result,
    model_context_multimodal,
)
from draive.models import ModelContext, ModelInput
from draive.multimodal import Multimodal, MultimodalContent
from draive.steps import Step


@evaluator(name="fluency")
async def fluency_evaluator(
    evaluated: Multimodal,
    /,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate fluency.

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


@evaluator(name="fluency_context")
async def fluency_context_evaluator(
    evaluated: ModelContext,
    /,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate fluency using model context.

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
Carefully examine provided CONTENT, then rate it using solely a fluency metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is fluency - the quality of the content in terms of grammar, spelling, punctuation, content choice, and overall structure.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a fluency score using exact name of one of the following values:
- "poor" is very low fluency, the content has many errors that make it hard to understand or look unnatural.
- "fair" is low fluency, the content partially reads well but contains noticeable issues affecting clarity or naturalness.
- "good" is moderate fluency, the content has some errors that affect clarity or smoothness, but the main points remain comprehensible.
- "excellent" is high fluency, the content reads well with only minor issues that rarely impact understanding.
- "perfect" is very high fluency - the content has few or no errors and is easy to read and follow.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the CONTENT conversation timeline. Focus on model-produced results in output elements and rate their language fluency.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is fluency of model results in context.
Assess the quality of model outputs in terms of grammar, spelling, punctuation, word choice, and overall linguistic naturalness across the conversation.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a fluency score using exact name of one of the following values:
- "poor" is very low fluency, model outputs have many errors that make them hard to understand or appear unnatural.
- "fair" is low fluency, model outputs partially read well but contain noticeable issues affecting clarity or naturalness.
- "good" is moderate fluency, model outputs have some errors that affect clarity or smoothness, but main points remain comprehensible.
- "excellent" is high fluency, model outputs read well with only minor issues that rarely impact understanding.
- "perfect" is very high fluency, model outputs have few or no errors and are easy to read and follow.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
