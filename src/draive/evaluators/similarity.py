from base64 import b64decode
from collections.abc import Sequence

from draive.embedding import Embedded, ImageEmbedding, TextEmbedding, vector_similarity_score
from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import (
    FORMAT_INSTRUCTION,
    extract_evaluation_result,
    is_empty_content,
    model_context_multimodal,
)
from draive.models import ModelContext, ModelInput
from draive.multimodal import Multimodal, MultimodalContent
from draive.resources import ResourceContent
from draive.steps import Step


@evaluator(name="similarity")
async def similarity_evaluator(
    evaluated: Multimodal,
    /,
    *,
    reference: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate similarity.

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
    if is_empty_content(evaluated):
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Input was empty!"},
        )

    if is_empty_content(reference):
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


@evaluator(name="similarity_context")
async def similarity_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    reference: Multimodal | None = None,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate similarity using model context.

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


@evaluator(name="text_vector_similarity")
async def text_vector_similarity_evaluator(
    evaluated: str,
    /,
    *,
    reference: str,
) -> float:
    """
    Evaluate text vector similarity.

    Parameters
    ----------
    evaluated : str
        Evaluator input parameter.
    reference : str
        Evaluator input parameter.

    Returns
    -------
    float
        Evaluation result clipped to [0.0, 1.0].
    """
    if not evaluated or not reference:
        return 0.0

    embedding: Sequence[Embedded[str]] = await TextEmbedding.embed_many([reference, evaluated])

    similarity: float = vector_similarity_score(
        value_vector=embedding[0].vector,
        reference_vector=embedding[1].vector,
    )

    return max(0.0, min(1.0, similarity))


@evaluator(name="image_vector_similarity")
async def image_vector_similarity_evaluator(
    evaluated: ResourceContent | bytes,
    /,
    *,
    reference: ResourceContent | bytes,
) -> float:
    """
    Evaluate image vector similarity.

    Parameters
    ----------
    evaluated : ResourceContent | bytes
        Evaluator input parameter.
    reference : ResourceContent | bytes
        Evaluator input parameter.

    Returns
    -------
    float
        Evaluation result clipped to [0.0, 1.0].
    """
    evaluated_data: bytes
    match evaluated:
        case ResourceContent() as media:
            evaluated_data = b64decode(media.data.encode("utf-8"))

        case raw_data:
            evaluated_data = raw_data

    reference_data: bytes
    match reference:
        case ResourceContent() as media:
            reference_data = b64decode(media.data.encode("utf-8"))

        case raw_data:
            reference_data = raw_data

    if not evaluated_data or not reference_data:
        return 0.0

    embedding: Sequence[Embedded[bytes]] = await ImageEmbedding.embed_many(
        [reference_data, evaluated_data]
    )

    similarity: float = vector_similarity_score(
        value_vector=embedding[0].vector,
        reference_vector=embedding[1].vector,
    )

    return max(0.0, min(1.0, similarity))


CONTENT_INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a similarity metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is similarity - the degree of semantic similarity between the REFERENCE and the EVALUATED content, judged by meaning rather than surface form.
Different formatting, wording, structure, or serialization (e.g. JSON vs XML) that conveys the same meaning does not reduce similarity. Judge only similarity of meaning: do not reward or penalize factual accuracy, style, or length on their own.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a similarity score using the exact name of one of the following values:
- "poor" - completely unrelated in meaning.
- "fair" - only superficial overlap, with notable divergence in meaning.
- "good" - shares common themes or ideas, with some meaningful differences.
- "excellent" - conveys closely related meaning, with only minor divergences.
- "perfect" - very close in meaning, or conveys essentially the same information.
Use the "none" value only for content that cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or content with no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_REFERENCE_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate them using solely a similarity metric against the REFERENCE according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is similarity of model results in context.
Assess the degree of semantic similarity between model outputs and the provided REFERENCE, judged by meaning rather than surface form. Different formatting, wording, or structure that conveys the same meaning does not reduce similarity. Judge only similarity of meaning: do not reward or penalize factual accuracy, style, or length on their own.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a similarity score using the exact name of one of the following values:
- "poor" - outputs are completely unrelated in meaning to the reference.
- "fair" - only superficial overlap with the reference, with notable divergence in meaning.
- "good" - shares common themes or ideas with the reference, with some meaningful differences.
- "excellent" - conveys meaning closely related to the reference, with only minor divergences.
- "perfect" - very close in meaning to the reference, or conveys essentially the same information.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements, then rate the degree of thematic and semantic consistency among them using solely a similarity metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is similarity of model results in context.
Assess the degree of thematic and semantic consistency among model outputs across the context, judged by meaning rather than surface form. Judge only similarity of meaning: do not reward or penalize factual accuracy, style, or length on their own.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a similarity score using the exact name of one of the following values:
- "poor" - outputs are completely unrelated in meaning to one another.
- "fair" - only superficial overlap, with notable divergence in meaning.
- "good" - shares common themes or ideas, with some meaningful differences.
- "excellent" - conveys closely related meaning, with only minor divergences.
- "perfect" - very close in meaning, or conveys essentially the same information.
Use the "none" value only when the model outputs cannot be rated at all: empty, whitespace-only, gibberish, random characters or bytes, encoded noise, or no intelligible, assessable payload.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
