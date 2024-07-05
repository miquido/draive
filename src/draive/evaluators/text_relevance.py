from draive.evaluation import evaluator
from draive.generation import generate_model
from draive.parameters import DataModel
from draive.types.multimodal import MultimodalContent, TextContent

__all__ = [
    "RelevanceScore",
    "text_relevance_evaluator",
]


class RelevanceScore(DataModel):
    score: float
    comment: str | None = None


INSTRUCTION: str = """\
You will be given a reference text and a derived text based on the reference text.
Your task is to rate the derived text based on one metric.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Relevance (1-5) - selection of important content from the reference text.
The derived text should include only important information from the reference text.
Annotators should penalize summaries that contain redundancies and excess information.

Evaluation Steps:
1. Read the derived text and the reference text carefully.
2. Compare the derived text to the reference text and identify the main points of the reference text.
3. Assess how well the derived text covers the main points of the reference text,
and note any irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.
"""  # noqa: E501

@evaluator(name="Relevance Evaluator", threshold=0.75)
async def text_relevance_evaluator(
    reference: str,
    text: str,
    /,
) -> float:
    input_content = MultimodalContent.of(
        TextContent(text=f"Reference text: {reference}\n\nDerived text: {text}") # type: ignore
     )
    model: RelevanceScore = await generate_model(
        RelevanceScore,
        instruction=INSTRUCTION,
        input=input_content,
    )

    return model.score/5
