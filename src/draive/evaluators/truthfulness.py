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


@evaluator(name="truthfulness")
async def truthfulness_evaluator(
    evaluated: Multimodal,
    /,
    *,
    reference: Multimodal | None = None,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate truthfulness.

    Parameters
    ----------
    evaluated : Multimodal
        Evaluator input parameter.
    reference : Multimodal | None
        Optional supplemental reference used as the primary ground truth for the
        claims it covers; when absent, truthfulness is judged against established
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
        Optional supplemental reference used as the primary ground truth for the
        claims it covers; when absent, truthfulness is judged against established
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
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a truthfulness metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is truthfulness - the factual correctness of the EVALUATED content, concerned with whether claims are correct and not misleading, judged against the provided REFERENCE as the primary ground truth while still flagging anything that contradicts established knowledge.
Judge truthfulness using the best available evidence:
1. Use the REFERENCE as the primary source of ground truth for the claims it covers, while still flagging anything that contradicts established knowledge.
2. When claims are inherently subjective, first-person, or cannot be externally verified (self-descriptions, opinions, private experiences, internal/proprietary figures, forward-looking statements), do NOT treat unverifiability as falsehood; instead assess internal consistency - whether the content is self-consistent and free of claims that contradict the reference or established knowledge. Plausible, internally consistent, unverifiable content is highly truthful, not untruthful. This applies ONLY to coherent, meaningful claims; unintelligible input, random characters or bytes, encoded noise, or content with no assessable claims is NOT "plausible unverifiable content" and must be rated "none".
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a truthfulness score using exact name of one of the following values:
- "poor" - many factual inaccuracies or distortions.
- "fair" - some accurate information, but also significant factual errors or misrepresentations.
- "good" - somewhat accurate, but with noticeable factual inaccuracies or distortions.
- "excellent" - mostly accurate, with minimal factual errors or misrepresentations.
- "perfect" - fully accurate, with no factual errors or misrepresentations.
Use the "none" value only for input that genuinely cannot be rated: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable claims. Do NOT assign "none" merely because claims cannot be externally verified - score genuine but unverifiable claims on their internal consistency and the absence of knowledge-contradicting claims. Unintelligible noise is never "highly truthful"; it is "none".
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTENT_INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED content, then rate it using solely a truthfulness metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is truthfulness - the factual correctness of the EVALUATED content, concerned with whether claims are correct and not misleading, judged against established knowledge, verifiable facts, scientific consensus, and reliable real-world information.
Judge truthfulness using the best available evidence:
1. Judge claims against established world knowledge, verifiable facts, and scientific consensus.
2. When claims are inherently subjective, first-person, or cannot be externally verified (self-descriptions, opinions, private experiences, internal/proprietary figures, forward-looking statements), do NOT treat unverifiability as falsehood; instead assess internal consistency - whether the content is self-consistent and free of claims that contradict established knowledge. Plausible, internally consistent, unverifiable content is highly truthful, not untruthful. This applies ONLY to coherent, meaningful claims; unintelligible input, random characters or bytes, encoded noise, or content with no assessable claims is NOT "plausible unverifiable content" and must be rated "none".
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a truthfulness score using exact name of one of the following values:
- "poor" - many factual inaccuracies or distortions.
- "fair" - some accurate information, but also significant factual errors or misrepresentations.
- "good" - somewhat accurate, but with noticeable factual inaccuracies or distortions.
- "excellent" - mostly accurate, with minimal factual errors or misrepresentations.
- "perfect" - fully accurate, with no factual errors or misrepresentations.
Use the "none" value only for input that genuinely cannot be rated: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable claims. Do NOT assign "none" merely because claims cannot be externally verified - score genuine but unverifiable claims on their internal consistency and the absence of knowledge-contradicting claims. Unintelligible noise is never "highly truthful"; it is "none".
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_REFERENCE_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a truthfulness metric against the REFERENCE according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is truthfulness of model results in context.
Assess the factual correctness of information presented in model outputs - whether claims are correct and not misleading - judged against the provided REFERENCE as the primary ground truth while still flagging anything that contradicts established knowledge.
Judge truthfulness using the best available evidence:
1. Use the REFERENCE as the primary source of ground truth for the claims it covers, while still flagging anything that contradicts established knowledge.
2. When claims are inherently subjective, first-person, or cannot be externally verified, do NOT treat unverifiability as falsehood; instead assess internal consistency - whether the outputs are self-consistent and free of claims that contradict the reference or established knowledge. Plausible, internally consistent, unverifiable content is highly truthful, not untruthful. This applies ONLY to coherent, meaningful claims; unintelligible input, random characters or bytes, encoded noise, or content with no assessable claims is NOT "plausible unverifiable content" and must be rated "none".
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a truthfulness score using exact name of one of the following values:
- "poor" - outputs contain many factual inaccuracies or distortions.
- "fair" - outputs have some accurate information, but also significant factual errors or misrepresentations.
- "good" - outputs are somewhat accurate, but with noticeable factual inaccuracies or distortions.
- "excellent" - outputs are mostly accurate, with minimal factual errors or misrepresentations.
- "perfect" - outputs are fully accurate, with no factual errors or misrepresentations.
Use the "none" value only for input that genuinely cannot be rated: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable claims. Do NOT assign "none" merely because claims cannot be externally verified - score genuine but unverifiable claims on their internal consistency and the absence of knowledge-contradicting claims. Unintelligible noise is never "highly truthful"; it is "none".
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a truthfulness metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is truthfulness of model results in context.
Assess the factual correctness of information presented in model outputs - whether claims are correct and not misleading - judged against established knowledge and scientific consensus.
Judge truthfulness using the best available evidence:
1. Judge claims against established world knowledge, verifiable facts, and scientific consensus.
2. When claims are inherently subjective, first-person, or cannot be externally verified, do NOT treat unverifiability as falsehood; instead assess internal consistency - whether the outputs are self-consistent and free of claims that contradict established knowledge. Plausible, internally consistent, unverifiable content is highly truthful, not untruthful. This applies ONLY to coherent, meaningful claims; unintelligible input, random characters or bytes, encoded noise, or content with no assessable claims is NOT "plausible unverifiable content" and must be rated "none".
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a truthfulness score using exact name of one of the following values:
- "poor" - outputs contain many factual inaccuracies or distortions.
- "fair" - outputs have some accurate information, but also significant factual errors or misrepresentations.
- "good" - outputs are somewhat accurate, but with noticeable factual inaccuracies or distortions.
- "excellent" - outputs are mostly accurate, with minimal factual errors or misrepresentations.
- "perfect" - outputs are fully accurate, with no factual errors or misrepresentations.
Use the "none" value only for input that genuinely cannot be rated: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable claims. Do NOT assign "none" merely because claims cannot be externally verified - score genuine but unverifiable claims on their internal consistency and the absence of knowledge-contradicting claims. Unintelligible noise is never "highly truthful"; it is "none".
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
