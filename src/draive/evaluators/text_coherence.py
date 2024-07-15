from draive.evaluation import evaluator
from draive.generation import generate_model
from draive.parameters import DataModel

__all__ = [
    "text_coherence_evaluator",
]


class CoherenceScore(DataModel):
    score: float
    comment: str | None = None


INSTRUCTION: str = """\
You will be given a reference text and a compared text based on the reference text.
Your task is to rate the compared text using only the Coherence metric, \
which is described in the Evaluation Criteria.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Coherence (1-5) - the collective quality of all sentences.
We align this dimension with the DUC (Document Understanding Conference) quality question of \
structure and coherence, whereby the text should be well-structured and well-organized.
The compared text should not just be a heap of related information, but should build from sentence
to sentence into a coherent body of information about a topic.

Rating Scale:
1: Very low coherence - the text is chaotic, lacking logical connections between sentences.
2: Low coherence - some connections are visible, but the overall structure is weak.
3: Moderate coherence - the text has a noticeable structure, but with some shortcomings.
4: Good coherence - the text is well-organized with minor imperfections.
5: Excellent coherence - the text is exemplarily structured, with smooth transitions between ideas.

Evaluation Steps:
1. Read the reference text carefully and identify the main topic and key points.
2. Read the compared text and compare it to the reference text.
Check if the compared text covers the main topic and key points of the reference text, \
and if it presents them in a clear and logical order.
3. Assign a coherence score from 1 to 5 based on the provided criteria.
"""

@evaluator(name="coherence")
async def text_coherence_evaluator(
    reference: str,
    compared: str,
    /,
) -> float:
    model: CoherenceScore = await generate_model(
        CoherenceScore,
        instruction=INSTRUCTION,
        input=f"Reference text: {reference}\n\nCompered text: {compared}",
    )

    return model.score/5
