from base64 import b64decode

from draive.embedding import Embedded, embed_images, embed_texts
from draive.evaluation import EvaluationScore, evaluator
from draive.generation import generate_text
from draive.similarity.score import vector_similarity_score
from draive.types import (
    ImageBase64Content,
    Multimodal,
    MultimodalTemplate,
    xml_tag,
)

__all__ = [
    "similarity_evaluator",
    "text_vector_similarity_evaluator",
    "image_vector_similarity_evaluator",
]


INSTRUCTION: str = """\
Assistant is an evaluator scoring the provided content.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate \
the EVALUATED content using solely a similarity metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is similarity - the degree of semantic similarity between the REFERENCE \
and the EVALUATED content.
</EVALUATION_CRITERIA>

<RATING>
Assign a similarity score using value between 0.0 and 2.0 where:
0.0 is no similarity - the content is completely unrelated in meaning.
1.0 is moderate similarity - the content share some common themes or ideas.
2.0 is high similarity - the content is very close in meaning \
or convey the same information.
</RATING>

<FORMAT>
The final result containing only the numerical score value HAVE to be put inside a `RESULT` \
xml tag within the result i.e. `<RESULT>score</RESULT>`.
</FORMAT>
"""


INPUT_TEMPLATE: MultimodalTemplate = MultimodalTemplate.of(
    "<REFERENCE>",
    ("reference",),
    "</REFERENCE>",
    "<EVALUATED>",
    ("evaluated",),
    "</EVALUATED>",
)


@evaluator(name="similarity")
async def similarity_evaluator(
    evaluated: Multimodal,
    /,
    reference: Multimodal,
) -> EvaluationScore:
    if not evaluated:
        return EvaluationScore(
            value=0,
            comment="Input was empty!",
        )

    if not reference:
        return EvaluationScore(
            value=0,
            comment="Reference was empty!",
        )

    if result := xml_tag(
        "RESULT",
        source=await generate_text(
            instruction=INSTRUCTION,
            input=INPUT_TEMPLATE.format(
                reference=reference,
                evaluated=evaluated,
            ),
        ),
        conversion=str,
    ):
        return EvaluationScore(
            value=float(result) / 2,
            comment=None,
        )

    else:
        raise ValueError("Invalid result")


@evaluator(name="text_vector_similarity")
async def text_vector_similarity_evaluator(
    evaluated: str,
    /,
    reference: str,
) -> float:
    embedding: list[Embedded[str]] = await embed_texts([reference, evaluated])

    return vector_similarity_score(embedding[0].vector, embedding[1].vector)


@evaluator(name="image_vector_similarity")
async def image_vector_similarity_evaluator(
    evaluated: ImageBase64Content | bytes,
    /,
    reference: ImageBase64Content | bytes,
) -> float:
    evaluated_data: bytes
    match evaluated:
        case ImageBase64Content() as base64_data:
            evaluated_data = b64decode(base64_data.image_base64)

        case raw_data:
            evaluated_data = raw_data

    reference_data: bytes
    match reference:
        case ImageBase64Content() as base64_data:
            reference_data = b64decode(base64_data.image_base64)

        case raw_data:
            reference_data = raw_data

    embedding: list[Embedded[bytes]] = await embed_images([reference_data, evaluated_data])

    return vector_similarity_score(embedding[0].vector, embedding[1].vector)
