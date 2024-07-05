from draive.evaluation import evaluator
from draive.generation import generate_model
from draive.parameters import DataModel
from draive.types.multimodal import MultimodalContent, TextContent

__all__ = [
    "ConsistencyScore",
    "text_consistency_evaluator",
]


class ConsistencyScore(DataModel):
    score: float
    comment: str | None = None


INSTRUCTION: str = """\
You will be given a reference text and a derived text based on the reference text.
Your task is to rate the derived text based on one metric.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Consistency(1-5) - the factual alignment between the reference text and the derived text. \
A factually consistent derived text contains only statements that are entailed by the reference text. \
Annotators should penalize derived texts that contain hallucinated facts.

Evaluation Steps:
1. Read the derived text and the reference document carefully.
2. Compare the derived text to the reference document and identify the main points of the article.
3. Assess how well the derived text covers the main points of the article and how much irrelevant or redundant information it contains.
4. Assign a consistency score from 1 to 5.
"""  # noqa: E501

@evaluator(name="Consistency Evaluator", threshold=0.75)
async def text_consistency_evaluator(
    reference: str,
    text: str,
    /,
) -> float:
    input_content = MultimodalContent.of(
        TextContent(text=f"Reference text: {reference}\n\nDerived text: {text}") # type: ignore
    )
    model: ConsistencyScore = await generate_model(
        ConsistencyScore,
        instruction=INSTRUCTION,
        input=input_content,
    )

    return model.score/5
