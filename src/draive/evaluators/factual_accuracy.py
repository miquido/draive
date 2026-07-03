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


@evaluator(name="factual_accuracy")
async def factual_accuracy_evaluator(
    evaluated: Multimodal,
    /,
    *,
    reference: Multimodal | None = None,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate factual accuracy.

    Parameters
    ----------
    evaluated : Multimodal
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
    if is_empty_content(evaluated):
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Input was empty!"},
        )

    instruction: str
    input_content: MultimodalContent
    if reference and not is_empty_content(reference):
        instruction = CONTENT_REFERENCE_INSTRUCTION
        input_content = MultimodalContent.of(
            "<REFERENCE>",
            reference,
            "</REFERENCE>\n<EVALUATED>",
            evaluated,
            "</EVALUATED>",
        )

    else:
        instruction = CONTENT_INSTRUCTION
        input_content = MultimodalContent.of(
            "<EVALUATED>",
            evaluated,
            "</EVALUATED>",
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


CONTENT_REFERENCE_INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a factual accuracy metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is factual accuracy - the extent to which the EVALUATED content contains factually correct claims, data, statements, and assertions, judged against the provided REFERENCE as the primary ground truth while still flagging anything that contradicts established knowledge.
Judge factual accuracy using the best available evidence:
1. Use the REFERENCE as the primary source of ground truth for the claims it covers, while still flagging anything that contradicts established knowledge.
2. When claims are inherently subjective, first-person, or cannot be externally verified (self-descriptions, opinions, private experiences, internal/proprietary figures, forward-looking statements), do NOT treat unverifiability as inaccuracy; instead assess internal consistency - whether the content is self-consistent and free of claims that contradict the reference or established knowledge. Plausible, internally consistent, unverifiable content is highly accurate, not inaccurate. This applies ONLY to coherent, meaningful claims; unintelligible input, random characters or bytes, encoded noise, or content with no assessable claims is NOT "plausible unverifiable content" and must be rated "none".
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a factual accuracy score using exact name of one of the following values:
- "poor" - many significant factual errors or false information.
- "fair" - several factual inaccuracies or questionable claims mixed with some correct information.
- "good" - mostly factually correct, but with some minor inaccuracies or unverified claims.
- "excellent" - largely factually correct, with minimal or very minor factual issues.
- "perfect" - completely factually accurate, with all information correct and verifiable.
Use the "none" value only for input that genuinely cannot be rated: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable claims. Do NOT assign "none" merely because claims cannot be externally verified - score genuine but unverifiable claims on their internal consistency and the absence of knowledge-contradicting claims. Unintelligible noise is never "highly accurate"; it is "none".
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTENT_INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED content, then rate it using solely a factual accuracy metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is factual accuracy - the extent to which the EVALUATED content contains factually correct claims, data, statements, and assertions, judged against established world knowledge, verifiable facts, scientific consensus, and reliable sources.
Judge factual accuracy using the best available evidence:
1. Judge claims against established world knowledge, verifiable facts, scientific consensus, and historical accuracy.
2. When claims are inherently subjective, first-person, or cannot be externally verified (self-descriptions, opinions, private experiences, internal/proprietary figures, forward-looking statements), do NOT treat unverifiability as inaccuracy; instead assess internal consistency - whether the content is self-consistent and free of claims that contradict established knowledge. Plausible, internally consistent, unverifiable content is highly accurate, not inaccurate. This applies ONLY to coherent, meaningful claims; unintelligible input, random characters or bytes, encoded noise, or content with no assessable claims is NOT "plausible unverifiable content" and must be rated "none".
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a factual accuracy score using exact name of one of the following values:
- "poor" - many significant factual errors or false information.
- "fair" - several factual inaccuracies or questionable claims mixed with some correct information.
- "good" - mostly factually correct, but with some minor inaccuracies or unverified claims.
- "excellent" - largely factually correct, with minimal or very minor factual issues.
- "perfect" - completely factually accurate, with all information correct and verifiable.
Use the "none" value only for input that genuinely cannot be rated: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable claims. Do NOT assign "none" merely because claims cannot be externally verified - score genuine but unverifiable claims on their internal consistency and the absence of knowledge-contradicting claims. Unintelligible noise is never "highly accurate"; it is "none".
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_REFERENCE_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a factual accuracy metric against the REFERENCE according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is factual accuracy of model results in context.
Assess the correctness of claims, data, statements, and assertions made in model outputs, judged against the provided REFERENCE as the primary ground truth while still flagging anything that contradicts established knowledge.
Judge factual accuracy using the best available evidence:
1. Use the REFERENCE as the primary source of ground truth for the claims it covers, while still flagging anything that contradicts established knowledge.
2. When claims are inherently subjective, first-person, or cannot be externally verified, do NOT treat unverifiability as inaccuracy; instead assess internal consistency - whether the outputs are self-consistent and free of claims that contradict the reference or established knowledge. Plausible, internally consistent, unverifiable content is highly accurate, not inaccurate. This applies ONLY to coherent, meaningful claims; unintelligible input, random characters or bytes, encoded noise, or content with no assessable claims is NOT "plausible unverifiable content" and must be rated "none".
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a factual accuracy score using exact name of one of the following values:
- "poor" - outputs contain many significant factual errors or false information.
- "fair" - outputs have several factual inaccuracies or questionable claims mixed with some correct information.
- "good" - outputs are mostly factually correct, but with some minor inaccuracies or unverified claims.
- "excellent" - outputs are largely factually correct, with minimal or very minor factual issues.
- "perfect" - outputs are completely factually accurate, with all information correct and verifiable.
Use the "none" value only for input that genuinely cannot be rated: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable claims. Do NOT assign "none" merely because claims cannot be externally verified - score genuine but unverifiable claims on their internal consistency and the absence of knowledge-contradicting claims. Unintelligible noise is never "highly accurate"; it is "none".
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a factual accuracy metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is factual accuracy of model results in context.
Assess the correctness of claims, data, statements, and assertions made in model outputs, judged against well-established facts, scientific consensus, and generally accepted knowledge.
Judge factual accuracy using the best available evidence:
1. Judge claims against established world knowledge, verifiable facts, scientific consensus, and historical accuracy.
2. When claims are inherently subjective, first-person, or cannot be externally verified, do NOT treat unverifiability as inaccuracy; instead assess internal consistency - whether the outputs are self-consistent and free of claims that contradict established knowledge. Plausible, internally consistent, unverifiable content is highly accurate, not inaccurate. This applies ONLY to coherent, meaningful claims; unintelligible input, random characters or bytes, encoded noise, or content with no assessable claims is NOT "plausible unverifiable content" and must be rated "none".
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a factual accuracy score using exact name of one of the following values:
- "poor" - outputs contain many significant factual errors or false information.
- "fair" - outputs have several factual inaccuracies or questionable claims mixed with some correct information.
- "good" - outputs are mostly factually correct, but with some minor inaccuracies or unverified claims.
- "excellent" - outputs are largely factually correct, with minimal or very minor factual issues.
- "perfect" - outputs are completely factually accurate, with all information correct and verifiable.
Use the "none" value only for input that genuinely cannot be rated: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable claims. Do NOT assign "none" merely because claims cannot be externally verified - score genuine but unverifiable claims on their internal consistency and the absence of knowledge-contradicting claims. Unintelligible noise is never "highly accurate"; it is "none".
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
