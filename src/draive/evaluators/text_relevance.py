from draive.evaluation import evaluator
from draive.generation import generate_model
from draive.parameters import DataModel

__all__ = [
    "text_relevance_evaluator",
]


class RelevanceScore(DataModel):
    score: float
    comment: str | None = None


INSTRUCTION: str = """\
You will be given a reference text and a compared text based on the reference text.
Your task is to rate the compared text using only the Relevance metric, \
which is described in the Evaluation Criteria.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Relevance (1-5) - selection of important content from the reference text.
The compared text should include only important information from the reference text.
Annotators should penalize compared texts that contain redundancies and excess information.

Rating Scale:
1: Very low relevance - the text contains mostly irrelevant or redundant information.
2: Low relevance - the text includes some important points but has \
significant irrelevant content.
3: Moderate relevance - the text covers most important points but includes \
some unnecessary information.
4: Good relevance - the text focuses on important information with minor inclusions \
of less relevant content.
5: Excellent relevance - the text precisely captures only the most important information \
from the reference text.

Evaluation Steps:
1. Read the compared text and the reference text carefully.
2. Compare the compared text to the reference text and identify \
the main points of the reference text.
3. Assess how well the compared text covers the main points of the reference text, \
and note any irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5 based on the provided criteria.
"""

@evaluator(name="relevance")
async def text_relevance_evaluator(
    reference: str,
    compared: str,
    /,
) -> float:
    model: RelevanceScore = await generate_model(
        RelevanceScore,
        instruction=INSTRUCTION,
        input=f"Reference text: {reference}\n\nCompered text: {compared}",
    )

    return model.score/5
