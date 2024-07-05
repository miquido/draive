from draive.evaluation import evaluator
from draive.generation import generate_model
from draive.parameters import DataModel
from draive.types.multimodal import MultimodalContent, TextContent

__all__ = [
    "CoherenceScore",
    "text_coherence_evaluator",
]


class CoherenceScore(DataModel):
    score: float
    comment: str | None = None


INSTRUCTION: str = """\
You will be given a reference text and a derived text based on the reference text.
Your task is to rate the derived text based on one metric.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Coherence (1-5) - the collective quality of all sentences.
We align this dimension with the DUC quality question of structure and coherence,
whereby the text should be well-structured and well-organized.
The derived text should not just be a heap of related information, but should build from sentence
to sentence into a coherent body of information about a topic.

**Evaluation Steps:**
1. Read the reference text carefully and identify the main topic and key points.
2. Read the derived text and compare it to the reference text.
Check if the derived text covers the main topic and key points of the reference text,
and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest,
based on the Evaluation Criteria."""

@evaluator(name="Coherence Evaluator", threshold=0.75)
async def text_coherence_evaluator(
    reference: str,
    text: str,
    /,
) -> float:
    input_content = MultimodalContent.of(
        TextContent(text=f"Reference text: {reference}\n\nDerived text: {text}") # type: ignore
    )
    model: CoherenceScore = await generate_model(
        CoherenceScore,
        instruction=INSTRUCTION,
        input=input_content,
    )

    return model.score/5
