from draive.evaluation import evaluator
from draive.generation import generate_model
from draive.parameters import DataModel

__all__ = [
    "text_conciseness_evaluator",
]


class ConcisenessScore(DataModel):
    score: float
    comment: str | None = None


INSTRUCTION: str = """\
You will be given a reference text and a compared text based on the reference text.
Your task is to rate the compared text using only the Conciseness metric, \
which is described in the Evaluation Criteria.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Conciseness (1-5) - the extent to which the compared text is brief and to the point \
while still covering all key information.
A concise compared text avoids unnecessary details and repetition.
Annotators should penalize compared texts that are overly verbose or include irrelevant information.

Rating Scale:
1: Very low conciseness - the text is excessively verbose with much irrelevant information.
2: Low conciseness - the text contains unnecessary details and some irrelevant information.
3: Moderate conciseness - the text is somewhat concise but could be more focused.
4: Good conciseness - the text is mostly concise with minimal unnecessary information.
5: Excellent conciseness - the text is highly concise, containing only essential information.

Evaluation Steps:
1. Read the derived text and the reference text carefully.
2. Compare the compared text to the reference text and identify the main \
points of the reference text.
3. Assess how well the compared text covers the main points of the reference text, \
and how much irrelevant or redundant information it contains.
4. Assign a conciseness score from 1 to 5 based on the provided criteria.
"""

@evaluator(name="conciseness")
async def text_conciseness_evaluator(
    reference: str,
    compared: str,
    /,
) -> float:
    model: ConcisenessScore = await generate_model(
        ConcisenessScore,
        instruction=INSTRUCTION,
        input=f"Reference text: {reference}\n\nCompered text: {compared}",
    )

    return model.score/5
