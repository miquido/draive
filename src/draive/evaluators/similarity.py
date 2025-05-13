from collections.abc import Sequence
from typing import cast

from draive.embedding import Embedded, ImageEmbedding, TextEmbedding
from draive.evaluation import EvaluationScore, EvaluationScoreValue, evaluator
from draive.multimodal import Multimodal, MultimodalContent, MultimodalTagElement
from draive.multimodal.media import MediaData
from draive.similarity.score import vector_similarity_score
from draive.stages import Stage

__all__ = (
    "image_vector_similarity_evaluator",
    "similarity_evaluator",
    "text_vector_similarity_evaluator",
)


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
{guidelines}
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
    guidelines: str | None = None,
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

    completion: MultimodalContent = await Stage.completion(
        MultimodalContent.of(
            "<REFERENCE>",
            reference,
            "</REFERENCE>\n<EVALUATED>",
            evaluated,
            "</EVALUATED>",
        ),
        instruction=INSTRUCTION.format(
            guidelines=f"\n<GUIDELINES>\n{guidelines}\n</GUIDELINES>\n"
            if guidelines is not None
            else ""
        ),
    ).execute()

    if result := MultimodalTagElement.parse_first(
        "RESULT",
        content=completion,
    ):
        return EvaluationScore.of(
            cast(EvaluationScoreValue, result.content.to_str()),
            comment=completion.to_str(),
        )

    else:
        raise ValueError("Invalid evaluator result:\n%s", completion)


@evaluator(name="text_vector_similarity")
async def text_vector_similarity_evaluator(
    evaluated: str,
    /,
    *,
    reference: str,
) -> float:
    embedding: Sequence[Embedded[str]] = await TextEmbedding.embed([reference, evaluated])

    return vector_similarity_score(
        value_vector=embedding[0].vector,
        reference_vector=embedding[1].vector,
    )


@evaluator(name="image_vector_similarity")
async def image_vector_similarity_evaluator(
    evaluated: MediaData | bytes,
    /,
    *,
    reference: MediaData | bytes,
) -> float:
    evaluated_data: bytes
    match evaluated:
        case MediaData() as media:
            evaluated_data = media.data

        case raw_data:
            evaluated_data = raw_data

    reference_data: bytes
    match reference:
        case MediaData() as media:
            reference_data = media.data

        case raw_data:
            reference_data = raw_data

    embedding: Sequence[Embedded[bytes]] = await ImageEmbedding.embed(
        [reference_data, evaluated_data]
    )

    return vector_similarity_score(
        value_vector=embedding[0].vector,
        reference_vector=embedding[1].vector,
    )
