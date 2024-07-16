from draive.evaluation import evaluator
from draive.generation import generate_model
from draive.parameters import DataModel

__all__ = [
    "text_coverage_evaluator",
]


class CoverageScore(DataModel):
    score: float
    comment: str | None = None


INSTRUCTION: str = """\
You will be given a reference text and a compared text based on the reference text.
Your task is to rate the compared text using only the Coverage metric, \
which is described in the Evaluation Criteria.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Coverage (1-5) - the extent to which the compared text includes all \
the key points from the reference text.
A compared text with good coverage includes all the important information from \
the reference text without omitting critical points.
Annotators should penalize compared texts that miss significant content.

Rating Scale:
1: Very low coverage - the text misses most key points from the reference text.
2: Low coverage - the text includes some key points but omits several important ones.
3: Moderate coverage - the text covers most key points but misses a few important details.
4: Good coverage - the text includes nearly all key points with minor omissions.
5: Excellent coverage - the text comprehensively covers all key points from the reference text.

Evaluation Steps:
1. Read the reference text carefully and identify all key points and important information.
2. Read the compared text and compare it to the reference text. \
Check if the compared text includes all the key points and important information \
from the reference text.
3. Assess how well the compared text covers the reference text, \
and if any critical points are missing.
4. Assign a coverage score from 1 to 5 based on the provided criteria.
"""

@evaluator(name="coverage")
async def text_coverage_evaluator(
    reference: str,
    compared: str,
    /,
) -> float:
    model: CoverageScore = await generate_model(
        CoverageScore,
        instruction=INSTRUCTION,
        input=f"Reference text: {reference}\n\nCompered text: {compared}",
    )

    return model.score/5
