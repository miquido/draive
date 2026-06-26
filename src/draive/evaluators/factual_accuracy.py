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
    *,
    reference: Multimodal | None = None,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate factual correctness of content based on the best available evidence.

    This evaluator assesses the factual accuracy of content by examining
    claims, data, statements, and assertions. When a reference is provided it is
    used as the primary source of ground truth; otherwise claims are judged
    against well-established facts, scientific consensus, and generally accepted
    knowledge, falling back to internal consistency for claims that cannot be
    externally verified.

    Parameters
    ----------
    evaluated : Multimodal
        The content to evaluate for factual accuracy
    reference : Multimodal | None, optional
        Optional authoritative source used as the primary ground truth for the
        claims it covers; when absent, accuracy is judged against established
        knowledge, by default None
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

    instruction: str
    input_content: MultimodalContent
    if reference:
        instruction = CONTENT_REFERENCE_INSTRUCTION
        input_content = MultimodalContent.of(
            "<REFERENCE>",
            reference,
            "</REFERENCE>\n<CONTENT>",
            evaluated,
            "</CONTENT>",
        )

    else:
        instruction = CONTENT_INSTRUCTION
        input_content = MultimodalContent.of(
            "<CONTENT>",
            evaluated,
            "</CONTENT>",
        )

    return extract_evaluation_result(
        await Step.generating_completion(
            instructions=instruction.format(
                guidelines=f"\n<GUIDELINES>\n{guidelines}\n</GUIDELINES>\n" if guidelines else "",
            ),
        ).run((ModelInput.of(input_content),))
    )


@evaluator(name="factual_accuracy_context")
async def factual_accuracy_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    reference: Multimodal | None = None,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate factual accuracy using model context.

    Parameters
    ----------
    evaluated : ModelContext
        Evaluator input parameter.
    reference : Multimodal | None
        Optional authoritative source used as the primary ground truth for the
        claims it covers; when absent, accuracy is judged against established
        knowledge.
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
            "</REFERENCE>\n<CONTENT>",
            evaluated_content,
            "</CONTENT>",
        )

    else:
        instruction = CONTEXT_INSTRUCTION
        input_content = MultimodalContent.of(
            "<CONTENT>",
            evaluated_content,
            "</CONTENT>",
        )

    return extract_evaluation_result(
        await Step.generating_completion(
            instructions=instruction.format(
                guidelines=f"\n<GUIDELINES>\n{guidelines}\n</GUIDELINES>\n" if guidelines else "",
            ),
        ).run((ModelInput.of(input_content),))
    )


CONTENT_REFERENCE_INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Carefully examine provided CONTENT, then rate it using solely a factual accuracy metric according to the EVALUATION_CRITERIA.
Judge factual accuracy using the best available evidence:
1. Use the provided REFERENCE as the primary source of ground truth for the claims it covers, while still flagging anything that contradicts established knowledge.
2. When claims are inherently subjective, first-person, or cannot be externally verified (self-descriptions, opinions, private experiences, internal/proprietary figures, forward-looking statements), do NOT treat unverifiability as inaccuracy. Instead assess internal consistency: whether the content is self-consistent and free of claims that contradict the reference or established knowledge. Plausible, internally consistent, unverifiable content is highly accurate - not inaccurate. This applies ONLY to coherent, meaningful claims; unintelligible input, random characters or bytes, encoded noise, or content with no assessable claims is NOT "plausible unverifiable content" and must be rated "none".
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is factual accuracy - the extent to which the content contains factually correct information judged against the provided REFERENCE as the primary ground truth, while still flagging anything that contradicts established knowledge. This evaluates the correctness of claims, data, statements, and assertions made in the content. Claims that cannot be externally verified are judged on internal consistency and the absence of statements that contradict the reference or established knowledge, rather than penalised for being unverifiable.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a factual accuracy score using exact name of one of the following values:
- "poor" is very low factual accuracy, the content contains many significant factual errors or false information.
- "fair" is low factual accuracy, the content has several factual inaccuracies or questionable claims mixed with some correct information.
- "good" is moderate factual accuracy, the content is mostly factually correct but contains some minor inaccuracies or unverified claims.
- "excellent" is high factual accuracy, the content is largely factually correct with minimal or very minor factual issues.
- "perfect" is very high factual accuracy, the content is completely factually accurate with all information being correct and verifiable.
Assign "none" for input that genuinely cannot be rated: empty content, gibberish, random characters or bytes, encoded noise, or any input that contains no intelligible, assessable claims. Do NOT assign "none" merely because claims cannot be externally verified - score genuine but unverifiable claims on their internal consistency and the absence of knowledge-contradicting claims. Unintelligible noise is never "highly accurate"; it is "none".
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTENT_INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Carefully examine provided CONTENT, then rate it using solely a factual accuracy metric according to the EVALUATION_CRITERIA.
Judge factual accuracy using the best available evidence:
1. Judge claims against established world knowledge, verifiable facts, scientific consensus, and historical accuracy.
2. When claims are inherently subjective, first-person, or cannot be externally verified (self-descriptions, opinions, private experiences, internal/proprietary figures, forward-looking statements), do NOT treat unverifiability as inaccuracy. Instead assess internal consistency: whether the content is self-consistent and free of claims that contradict established knowledge. Plausible, internally consistent, unverifiable content is highly accurate - not inaccurate. This applies ONLY to coherent, meaningful claims; unintelligible input, random characters or bytes, encoded noise, or content with no assessable claims is NOT "plausible unverifiable content" and must be rated "none".
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is factual accuracy - the extent to which the content contains factually correct information judged against established world knowledge, verifiable facts, and reliable sources. This evaluates the correctness of claims, data, statements, and assertions made in the content. Claims that cannot be externally verified are judged on internal consistency and the absence of statements that contradict established knowledge, rather than penalised for being unverifiable.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a factual accuracy score using exact name of one of the following values:
- "poor" is very low factual accuracy, the content contains many significant factual errors or false information.
- "fair" is low factual accuracy, the content has several factual inaccuracies or questionable claims mixed with some correct information.
- "good" is moderate factual accuracy, the content is mostly factually correct but contains some minor inaccuracies or unverified claims.
- "excellent" is high factual accuracy, the content is largely factually correct with minimal or very minor factual issues.
- "perfect" is very high factual accuracy, the content is completely factually accurate with all information being correct and verifiable.
Assign "none" for input that genuinely cannot be rated: empty content, gibberish, random characters or bytes, encoded noise, or any input that contains no intelligible, assessable claims. Do NOT assign "none" merely because claims cannot be externally verified - score genuine but unverifiable claims on their internal consistency and the absence of knowledge-contradicting claims. Unintelligible noise is never "highly accurate"; it is "none".
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_REFERENCE_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the CONTENT conversation timeline. Focus on model-produced results in output elements and assess their factual accuracy according to the EVALUATION_CRITERIA.
Judge factual accuracy using the best available evidence:
1. Use the provided REFERENCE as the primary source of ground truth for the claims it covers, while still flagging anything that contradicts established knowledge.
2. When claims are inherently subjective, first-person, or cannot be externally verified, do NOT treat unverifiability as inaccuracy. Instead assess internal consistency: whether the outputs are self-consistent and free of claims that contradict the reference or established knowledge. Plausible, internally consistent, unverifiable content is highly accurate - not inaccurate. This applies ONLY to coherent, meaningful claims; unintelligible input, random characters or bytes, encoded noise, or content with no assessable claims is NOT "plausible unverifiable content" and must be rated "none".
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is factual accuracy of model results in context.
Assess the correctness of claims, data, statements, and assertions made in model outputs against the provided REFERENCE as the primary ground truth, while still flagging anything that contradicts established knowledge. Claims that cannot be externally verified are judged on internal consistency and the absence of statements that contradict the reference or established knowledge, rather than penalised for being unverifiable.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a factual accuracy score using exact name of one of the following values:
- "poor" is very low factual accuracy, model outputs contain many significant factual errors or false information.
- "fair" is low factual accuracy, model outputs have several factual inaccuracies or questionable claims mixed with some correct information.
- "good" is moderate factual accuracy, model outputs are mostly factually correct but contain some minor inaccuracies or unverified claims.
- "excellent" is high factual accuracy, model outputs are largely factually correct with minimal or very minor factual issues.
- "perfect" is very high factual accuracy, model outputs are completely factually accurate with all information being correct and verifiable.
Assign "none" for input that genuinely cannot be rated: empty content, gibberish, random characters or bytes, encoded noise, or any input that contains no intelligible, assessable claims. Do NOT assign "none" merely because claims cannot be externally verified - score genuine but unverifiable claims on their internal consistency and the absence of knowledge-contradicting claims. Unintelligible noise is never "highly accurate"; it is "none".
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the CONTENT conversation timeline. Focus on model-produced results in output elements and assess their factual accuracy according to the EVALUATION_CRITERIA.
Judge factual accuracy using the best available evidence:
1. Judge claims against established world knowledge, verifiable facts, scientific consensus, and historical accuracy.
2. When claims are inherently subjective, first-person, or cannot be externally verified, do NOT treat unverifiability as inaccuracy. Instead assess internal consistency: whether the outputs are self-consistent and free of claims that contradict established knowledge. Plausible, internally consistent, unverifiable content is highly accurate - not inaccurate. This applies ONLY to coherent, meaningful claims; unintelligible input, random characters or bytes, encoded noise, or content with no assessable claims is NOT "plausible unverifiable content" and must be rated "none".
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is factual accuracy of model results in context.
Assess the correctness of claims, data, statements, and assertions made in model outputs against well-established facts, scientific consensus, and generally accepted knowledge. Claims that cannot be externally verified are judged on internal consistency and the absence of statements that contradict established knowledge, rather than penalised for being unverifiable.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a factual accuracy score using exact name of one of the following values:
- "poor" is very low factual accuracy, model outputs contain many significant factual errors or false information.
- "fair" is low factual accuracy, model outputs have several factual inaccuracies or questionable claims mixed with some correct information.
- "good" is moderate factual accuracy, model outputs are mostly factually correct but contain some minor inaccuracies or unverified claims.
- "excellent" is high factual accuracy, model outputs are largely factually correct with minimal or very minor factual issues.
- "perfect" is very high factual accuracy, model outputs are completely factually accurate with all information being correct and verifiable.
Assign "none" for input that genuinely cannot be rated: empty content, gibberish, random characters or bytes, encoded noise, or any input that contains no intelligible, assessable claims. Do NOT assign "none" merely because claims cannot be externally verified - score genuine but unverifiable claims on their internal consistency and the absence of knowledge-contradicting claims. Unintelligible noise is never "highly accurate"; it is "none".
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
