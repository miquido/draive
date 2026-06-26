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
        Optional supplemental reference used as additional context;
        truthfulness is judged against established knowledge regardless of source.
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

    instruction: str
    input_content: MultimodalContent
    if reference:
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


CONTENT_REFERENCE_INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED content and rate it using solely a truthfulness metric according to the EVALUATION_CRITERIA.
Judge truthfulness using the best available evidence:
1. Use the provided REFERENCE as the primary source of ground truth for the claims it covers, while still flagging anything that contradicts established knowledge.
2. When claims are inherently subjective, first-person, or cannot be externally verified (self-descriptions, opinions, private experiences, internal/proprietary figures, forward-looking statements), do NOT treat unverifiability as falsehood. Instead assess internal consistency: whether the content is self-consistent and free of claims that contradict the reference or established knowledge. Plausible, internally consistent, unverifiable content is highly truthful - not untruthful. This applies ONLY to coherent, meaningful claims; unintelligible input, random characters or bytes, encoded noise, or content with no assessable claims is NOT "plausible unverifiable content" and must be rated "none".
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is truthfulness - the factual correctness of the content judged against the provided REFERENCE as the primary ground truth, while still flagging anything that contradicts established knowledge. Truthfulness is concerned with whether claims are correct and not misleading. Claims that cannot be externally verified are judged on internal consistency and the absence of statements that contradict the reference or established knowledge, rather than penalised for being unverifiable.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a truthfulness (factual accuracy) score using exact name of one of the following values:
- "poor" is very low truthfulness, the content contains many factual inaccuracies or distortions.
- "fair" is low truthfulness, the content has some accurate information but also significant factual errors or misrepresentations.
- "good" is moderate truthfulness, the content is somewhat accurate but with noticeable factual inaccuracies or distortions.
- "excellent" is high truthfulness, the content is mostly accurate with minimal factual errors or misrepresentations.
- "perfect" is very high truthfulness, the content is fully accurate, with no factual errors or misrepresentations.
Assign "none" for input that genuinely cannot be rated: empty content, gibberish, random characters or bytes, encoded noise, or any input that contains no intelligible, assessable claims. Do NOT assign "none" merely because claims cannot be externally verified - score genuine but unverifiable claims on their internal consistency and the absence of knowledge-contradicting claims. Unintelligible noise is never "highly truthful"; it is "none".
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTENT_INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED content and rate it using solely a truthfulness metric according to the EVALUATION_CRITERIA.
Judge truthfulness using the best available evidence:
1. Judge claims against established world knowledge, verifiable facts, and scientific consensus.
2. When claims are inherently subjective, first-person, or cannot be externally verified (self-descriptions, opinions, private experiences, internal/proprietary figures, forward-looking statements), do NOT treat unverifiability as falsehood. Instead assess internal consistency: whether the content is self-consistent and free of claims that contradict established knowledge. Plausible, internally consistent, unverifiable content is highly truthful - not untruthful. This applies ONLY to coherent, meaningful claims; unintelligible input, random characters or bytes, encoded noise, or content with no assessable claims is NOT "plausible unverifiable content" and must be rated "none".
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is truthfulness - the factual correctness of the content judged against established knowledge, verifiable facts, scientific consensus, and reliable real-world information. Truthfulness is concerned with whether claims are correct and not misleading. Claims that cannot be externally verified are judged on internal consistency and the absence of statements that contradict established knowledge, rather than penalised for being unverifiable.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a truthfulness (factual accuracy) score using exact name of one of the following values:
- "poor" is very low truthfulness, the content contains many factual inaccuracies or distortions.
- "fair" is low truthfulness, the content has some accurate information but also significant factual errors or misrepresentations.
- "good" is moderate truthfulness, the content is somewhat accurate but with noticeable factual inaccuracies or distortions.
- "excellent" is high truthfulness, the content is mostly accurate with minimal factual errors or misrepresentations.
- "perfect" is very high truthfulness, the content is fully accurate, with no factual errors or misrepresentations.
Assign "none" for input that genuinely cannot be rated: empty content, gibberish, random characters or bytes, encoded noise, or any input that contains no intelligible, assessable claims. Do NOT assign "none" merely because claims cannot be externally verified - score genuine but unverifiable claims on their internal consistency and the absence of knowledge-contradicting claims. Unintelligible noise is never "highly truthful"; it is "none".
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_REFERENCE_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements and assess their factual truthfulness according to the EVALUATION_CRITERIA.
Judge truthfulness using the best available evidence:
1. Use the provided REFERENCE as the primary source of ground truth for the claims it covers, while still flagging anything that contradicts established knowledge.
2. When claims are inherently subjective, first-person, or cannot be externally verified, do NOT treat unverifiability as falsehood. Instead assess internal consistency: whether the outputs are self-consistent and free of claims that contradict the reference or established knowledge. Plausible, internally consistent, unverifiable content is highly truthful - not untruthful. This applies ONLY to coherent, meaningful claims; unintelligible input, random characters or bytes, encoded noise, or content with no assessable claims is NOT "plausible unverifiable content" and must be rated "none".
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is truthfulness of model results in context.
Assess the factual correctness of information presented in model outputs against the provided REFERENCE as the primary ground truth, while still flagging anything that contradicts established knowledge. Ensure that facts are correct and reliable and that outputs do not perpetuate errors or falsehoods. Claims that cannot be externally verified are judged on internal consistency and the absence of statements that contradict the reference or established knowledge, rather than penalised for being unverifiable.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a truthfulness score using exact name of one of the following values:
- "poor" is very low truthfulness, model outputs contain many factual inaccuracies or distortions.
- "fair" is low truthfulness, model outputs have some accurate information but also significant factual errors or misrepresentations.
- "good" is moderate truthfulness, model outputs are somewhat accurate but with noticeable factual inaccuracies or distortions.
- "excellent" is high truthfulness, model outputs are mostly accurate with minimal factual errors or misrepresentations.
- "perfect" is very high truthfulness, model outputs are fully accurate with no factual errors or misrepresentations.
Assign "none" for input that genuinely cannot be rated: empty content, gibberish, random characters or bytes, encoded noise, or any input that contains no intelligible, assessable claims. Do NOT assign "none" merely because claims cannot be externally verified - score genuine but unverifiable claims on their internal consistency and the absence of knowledge-contradicting claims. Unintelligible noise is never "highly truthful"; it is "none".
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements and assess their factual truthfulness according to the EVALUATION_CRITERIA.
Judge truthfulness using the best available evidence:
1. Judge claims against established world knowledge, verifiable facts, and scientific consensus.
2. When claims are inherently subjective, first-person, or cannot be externally verified, do NOT treat unverifiability as falsehood. Instead assess internal consistency: whether the outputs are self-consistent and free of claims that contradict established knowledge. Plausible, internally consistent, unverifiable content is highly truthful - not untruthful. This applies ONLY to coherent, meaningful claims; unintelligible input, random characters or bytes, encoded noise, or content with no assessable claims is NOT "plausible unverifiable content" and must be rated "none".
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is truthfulness of model results in context.
Assess the factual correctness of information presented in model outputs against established knowledge and scientific consensus. Ensure that facts are correct and reliable and that outputs do not perpetuate errors or falsehoods. Claims that cannot be externally verified are judged on internal consistency and the absence of statements that contradict established knowledge, rather than penalised for being unverifiable.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a truthfulness score using exact name of one of the following values:
- "poor" is very low truthfulness, model outputs contain many factual inaccuracies or distortions.
- "fair" is low truthfulness, model outputs have some accurate information but also significant factual errors or misrepresentations.
- "good" is moderate truthfulness, model outputs are somewhat accurate but with noticeable factual inaccuracies or distortions.
- "excellent" is high truthfulness, model outputs are mostly accurate with minimal factual errors or misrepresentations.
- "perfect" is very high truthfulness, model outputs are fully accurate with no factual errors or misrepresentations.
Assign "none" for input that genuinely cannot be rated: empty content, gibberish, random characters or bytes, encoded noise, or any input that contains no intelligible, assessable claims. Do NOT assign "none" merely because claims cannot be externally verified - score genuine but unverifiable claims on their internal consistency and the absence of knowledge-contradicting claims. Unintelligible noise is never "highly truthful"; it is "none".
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
