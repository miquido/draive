from draive.embedding import Embedded, embed_texts
from draive.evaluation import evaluator
from draive.generation import generate_model
from draive.parameters import DataModel
from draive.similarity.score import vector_similarity_score

__all__ = [
    "text_similarity_evaluator",
    "text_vector_similarity_evaluator",
]


class SimilarityScore(DataModel):
    score: float
    comment: str | None = None


INSTRUCTION: str = """\
You will be given two texts: a reference text and a compared text. \
Your task is to rate the compared text using only the Similarity metric, \
which is described in the Evaluation Criteria.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Similarity (0.0-2.0) - the degree of semantic similarity between the reference text \
and the compared text.

Rating Scale:
0.0: No similarity - the reference text and compared text are completely unrelated in meaning.
1.0: Moderate similarity - the reference text and compared text share some common themes or ideas.
2.0: High similarity - the reference text and compared text are very close in meaning \
or convey the same information.

Evaluation Steps:
1. Read both the reference text and the compared text carefully.
2. Compare the semantic meaning of the reference text and the compared text.
3. Assign a similarity score from 0.0 to 2.0 based on the provided criteria.

Important: The score must be a decimal number from 0.0 to 2.0. 2.0 is the maximum, \
do not exceed this value.
"""

INPUT: str = """
Reference text: {reference}

Compered text: {compared}
"""


@evaluator(name="text_similarity")
async def text_similarity_evaluator(
    compared: str,
    /,
    reference: str,
) -> float:
    model: SimilarityScore = await generate_model(
        SimilarityScore,
        instruction=INSTRUCTION,
        input=f"Reference text: {reference}\n\nCompered text: {compared}",
        examples=[
            (
                INPUT.format(
                    reference=(
                        "Cats are popular pets. They are independent and like to groom themselves."
                    ),
                    compared=(
                        "Bananas are a healthy fruit. They are rich in potassium and easy to peel."
                    ),
                ),
                SimilarityScore(score=0.0),
            ),
            (
                INPUT.format(
                    reference=(
                        "The beach is a great place for relaxation. "
                        "People enjoy swimming and sunbathing."
                    ),
                    compared=(
                        "Many people like to spend time outdoors. "
                        "Parks are popular for picnics and walking."
                    ),
                ),
                SimilarityScore(score=1.0),
            ),
            (
                INPUT.format(
                    reference=(
                        "Coffee is a popular morning drink. It contains caffeine which helps "
                        "people feel more alert."
                    ),
                    compared=(
                        "Many people start their day with coffee. "
                        "The caffeine in coffee can increase alertness and energy."
                    ),
                ),
                SimilarityScore(score=2.0),
            ),
        ],
    )

    return model.score / 2


@evaluator(name="text_vector_similarity")
async def text_vector_similarity_evaluator(
    compared: str,
    /,
    reference: str,
) -> float:
    embedding: list[Embedded[str]] = await embed_texts([reference, compared])

    return vector_similarity_score(embedding[0].vector, embedding[1].vector)
