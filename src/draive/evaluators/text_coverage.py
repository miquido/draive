from draive.evaluation import evaluator
from draive.generation import generate_model
from draive.parameters import DataModel
from draive.types.multimodal import MultimodalContent, TextContent

__all__ = [
    "CoverageScore",
    "text_coverage_evaluator",
]


class CoverageScore(DataModel):
    score: float
    comment: str | None = None


INSTRUCTION: str = """\
You will be given a reference text and a derived text based on the reference text.
Your task is to rate the derived text based on one metric.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Coverage (1-5) - the extent to which the derived text includes all the key points from the source document.
A derived text with good coverage includes all the important information from the source without omitting critical points.
Annotators should penalize summaries that miss significant content.

Evaluation Steps:
1. Read the reference text carefully and identify all key points and important information.
2. Read the derived text and compare it to the reference text. Check if the derived text includes all the key points and important information from the reference text.
3. Assess how well the summary covers the reference text, and if any critical points are missing.
4. Assign a coverage score from 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""  # noqa: E501

@evaluator(name="Coverage Evaluator", threshold=0.75)
async def text_coverage_evaluator(
    reference: str,
    text: str,
    /,
) -> float:
    input_content = MultimodalContent.of(
        TextContent(text=f"Reference text: {reference}\n\nDerived text: {text}") # type: ignore
     )
    model: CoverageScore = await generate_model(
        CoverageScore,
        instruction=INSTRUCTION,
        input=input_content,
    )

    return model.score/5
