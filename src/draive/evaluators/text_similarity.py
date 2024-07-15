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
Similarity (1-3) - the degree of semantic similarity between the reference text \
and the compared text.

Rating Scale:
1: No similarity - the reference text and compared text are completely unrelated in meaning.
2: Moderate similarity - the reference text and compared text share some common themes or ideas.
3: High similarity - the reference text and compared text are very close in meaning \
or convey the same information.

Evaluation Steps:
1. Read both the reference text and the compared text carefully.
2. Compare the semantic meaning of the reference text and the compared text.
3. Assign a similarity score from 1 to 3 based on the provided criteria.
"""

@evaluator(name="similarity")
async def text_similarity_evaluator(
    reference: str,
    compared: str,
    /,
) -> float:
    model: SimilarityScore = await generate_model(
        SimilarityScore,
        instruction=INSTRUCTION,
        input=f"Reference text: {reference}\n\nCompered text: {compared}",
    )

    return model.score/3

@evaluator(name="text vector similarity")
async def text_vector_similarity_evaluator(
    reference: str,
    compared: str,
    /,
) -> float:
    embedding: list[Embedded[str]] = await embed_texts([reference, compared])

    return vector_similarity_score(embedding[0].vector, embedding[1].vector)
