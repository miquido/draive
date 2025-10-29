from base64 import urlsafe_b64decode
from collections.abc import Sequence

from draive.embedding import Embedded, ImageEmbedding, TextEmbedding, vector_similarity_score
from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import FORMAT_INSTRUCTION, extract_evaluation_result
from draive.multimodal import Multimodal, MultimodalContent
from draive.resources import ResourceContent
from draive.stages import Stage

__all__ = (
    "image_vector_similarity_evaluator",
    "similarity_evaluator",
    "text_vector_similarity_evaluator",
)


INSTRUCTION: str = f"""\
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
- "poor" is very low similarity; the content is completely unrelated in meaning.
- "good" is moderate similarity; the content shares some common themes or ideas.
- "perfect" is very high similarity; the content is very close in meaning or conveys the same information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501


@evaluator(name="similarity")
async def similarity_evaluator(
    evaluated: Multimodal,
    /,
    *,
    reference: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
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
        await Stage.completion(
            MultimodalContent.of(
                "<REFERENCE>",
                reference,
                "</REFERENCE>\n<EVALUATED>",
                evaluated,
                "</EVALUATED>",
            ),
            instructions=INSTRUCTION.format(
                guidelines=f"\n<GUIDELINES>\n{guidelines}\n</GUIDELINES>\n" if guidelines else "",
            ),
        ).execute()
    )


@evaluator(name="text_vector_similarity")
async def text_vector_similarity_evaluator(
    evaluated: str,
    /,
    *,
    reference: str,
) -> float:
    embedding: Sequence[Embedded[str]] = await TextEmbedding.embed_many([reference, evaluated])

    return vector_similarity_score(
        value_vector=embedding[0].vector,
        reference_vector=embedding[1].vector,
    )


@evaluator(name="image_vector_similarity")
async def image_vector_similarity_evaluator(
    evaluated: ResourceContent | bytes,
    /,
    *,
    reference: ResourceContent | bytes,
) -> float:
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

    embedding: Sequence[Embedded[bytes]] = await ImageEmbedding.embed_many(
        [reference_data, evaluated_data]
    )

    return vector_similarity_score(
        value_vector=embedding[0].vector,
        reference_vector=embedding[1].vector,
    )
