from typing import cast

from draive.embedding import Embedded, embed_images, embed_texts
from draive.evaluation import EvaluationScore, EvaluationScoreValue, evaluator
from draive.multimodal import MediaContent, Multimodal, MultimodalContent, MultimodalTagElement
from draive.similarity.score import vector_similarity_score
from draive.steps import steps_completion

__all__ = [
    "similarity_evaluator",
    "text_vector_similarity_evaluator",
    "image_vector_similarity_evaluator",
]


INSTRUCTION: str = """\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate\
 the EVALUATED content using solely a similarity metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is similarity - the degree of semantic similarity between the REFERENCE\
 and the EVALUATED content.
</EVALUATION_CRITERIA>

<RATING>
Assign a similarity score using exact name of one of the following values:
- "poor" is very low similarity, the content is completely unrelated in meaning.
- "good" is moderate similarity, the content share some common themes or ideas.
- "perfect" is very high similarity, the content is very close in meaning\
 or convey the same information.
Use the "none" value for content that cannot be rated at all.
</RATING>

<FORMAT>
The final result containing only the numerical score value HAVE to be put inside a `RESULT` \
xml tag within the result i.e. `<RESULT>good</RESULT>`.
</FORMAT>
"""


@evaluator(name="similarity")
async def similarity_evaluator(
    evaluated: Multimodal,
    /,
    *,
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

    completion: MultimodalContent = await steps_completion(
        MultimodalContent.of(
            "<REFERENCE>",
            reference,
            "</REFERENCE>\n<EVALUATED>",
            evaluated,
            "</EVALUATED>",
        ),
        instruction=INSTRUCTION,
    )

    if result := MultimodalTagElement.parse_first(
        completion,
        tag="RESULT",
    ):
        return EvaluationScore.of(
            cast(EvaluationScoreValue, result.content.as_string()),
            comment=completion.as_string(),
        )

    else:
        raise ValueError("Invalid evaluator result")


@evaluator(name="text_vector_similarity")
async def text_vector_similarity_evaluator(
    evaluated: str,
    /,
    *,
    reference: str,
) -> float:
    embedding: list[Embedded[str]] = await embed_texts([reference, evaluated])

    return vector_similarity_score(
        value_vector=embedding[0].vector,
        reference_vector=embedding[1].vector,
    )


@evaluator(name="image_vector_similarity")
async def image_vector_similarity_evaluator(
    evaluated: MediaContent | bytes,
    /,
    *,
    reference: MediaContent | bytes,
) -> float:
    evaluated_data: bytes
    match evaluated:
        case MediaContent() as media:
            match media.source:
                case bytes() as data:
                    evaluated_data = data

                case str():
                    raise ValueError("Unsupported media source")

        case raw_data:
            evaluated_data = raw_data

    reference_data: bytes
    match reference:
        case MediaContent() as media:
            match media.source:
                case bytes() as data:
                    reference_data = data

                case str():
                    raise ValueError("Unsupported media source")

        case raw_data:
            reference_data = raw_data

    embedding: list[Embedded[bytes]] = await embed_images([reference_data, evaluated_data])

    return vector_similarity_score(
        value_vector=embedding[0].vector,
        reference_vector=embedding[1].vector,
    )
