from draive.embedding import Embedded, embed_text
from draive.evaluation import evaluator
from draive.generation import generate_model
from draive.parameters import DataModel
from draive.similarity.score import vector_similarity_score
from draive.types.multimodal import MultimodalContent, TextContent

__all__ = [
    "SimilarityScore",
    "text_similarity_evaluator",
    "vector_similarity_evaluator",
]

class SimilarityScore(DataModel):
    score: float
    comment: str | None = None

INSTRUCTION: str = """\
You will be given two texts. Your task is to rate the similarity between these texts based on their semantic meaning.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Similarity (0-1) - the degree of semantic similarity between the two texts.
0: No similarity. The texts are completely unrelated in meaning.
0.5: Moderate similarity. The texts share some common themes or ideas.
1: High similarity. The texts are very close in meaning or convey the same information.

Evaluation Steps:
1. Read both texts carefully.
2. Compare the semantic meaning of the texts.
3. Assign a similarity score from 0 to 1, where higher scores indicate greater similarity.
"""  # noqa: E501

@evaluator(name="Similarity Evaluator", threshold=0.75)
async def text_similarity_evaluator(
    text1: str,
    text2: str,
    /,
) -> float:
    input_content = MultimodalContent.of(
        TextContent(text=f"Text 1: {text1}\n\nText 2: {text2}") # type: ignore
    )
    model: SimilarityScore = await generate_model(
        SimilarityScore,
        instruction=INSTRUCTION,
        input=input_content,
    )

    return model.score

@evaluator(name="Vector Similarity Evaluator", threshold=0.75)
async def vector_similarity_evaluator(
    text1: str,
    text2: str,
    /,
) -> float:
    embedding1: Embedded[str] = await embed_text(text1)
    embedding2: Embedded[str] = await embed_text(text2)

    return vector_similarity_score(embedding1.vector, embedding2.vector)
