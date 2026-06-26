from base64 import urlsafe_b64decode
from collections.abc import Sequence

from draive.embedding import Embedded, ImageEmbedding, TextEmbedding, vector_similarity_score
from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import (
    FORMAT_INSTRUCTION,
    extract_evaluation_result,
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
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a similarity metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is similarity - the degree of semantic similarity between the REFERENCE and the EVALUATED content.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a similarity score using the exact name of one of the following values:
- "poor" is very low similarity, the content is completely unrelated in meaning.
- "fair" is low similarity, the content shares only superficial overlap with notable divergence in meaning.
- "good" is moderate similarity, the content shares common themes or ideas with some meaningful differences.
- "excellent" is high similarity, the content conveys closely related meaning with only minor divergences.
- "perfect" is very high similarity, the content is very close in meaning or conveys essentially the same information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501


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
            evaluated_data = urlsafe_b64decode(media.data.encode("utf-8"))

        case raw_data:
            evaluated_data = raw_data

    reference_data: bytes
    match reference:
        case ResourceContent() as media:
            reference_data = urlsafe_b64decode(media.data.encode("utf-8"))

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


CONTEXT_REFERENCE_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements and assess how semantically similar they are to the REFERENCE.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is similarity of model results in context.
Assess the degree of semantic similarity between model outputs and the provided REFERENCE.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a similarity score using the exact name of one of the following values:
- "poor" is very low similarity, model outputs are completely unrelated in meaning to the reference.
- "fair" is low similarity, model outputs share only superficial overlap with the reference with notable divergence in meaning.
- "good" is moderate similarity, model outputs share common themes or ideas with the reference with some meaningful differences.
- "excellent" is high similarity, model outputs convey meaning closely related to the reference with only minor divergences.
- "perfect" is very high similarity, model outputs are very close in meaning to the reference or convey essentially the same information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501

CONTEXT_INSTRUCTION: str = f"""\
You are evaluating model results produced within a conversation context according to the defined criteria.

<INSTRUCTION>
Carefully examine the EVALUATED conversation timeline. Focus on model-produced results in output elements and assess the degree of thematic and semantic consistency among them.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is similarity of model results in context.
Assess the degree of thematic and semantic consistency among model outputs across the context.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a similarity score using the exact name of one of the following values:
- "poor" is very low similarity, model outputs are completely unrelated in meaning to one another.
- "fair" is low similarity, model outputs share only superficial overlap with notable divergence in meaning.
- "good" is moderate similarity, model outputs share common themes or ideas with some meaningful differences.
- "excellent" is high similarity, model outputs convey closely related meaning with only minor divergences.
- "perfect" is very high similarity, model outputs are very close in meaning or convey essentially the same information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501
