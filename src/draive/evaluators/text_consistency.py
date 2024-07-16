from draive.evaluation import evaluator
from draive.generation import generate_model
from draive.parameters import DataModel

__all__ = [
    "text_consistency_evaluator",
]


class ConsistencyScore(DataModel):
    score: float
    comment: str | None = None


INSTRUCTION: str = """\
You will be given a reference text and a compared text based on the reference text.
Your task is to rate the compared text using only the Consistency metric, \
which is described in the Evaluation Criteria.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Consistency(1-5) - the factual alignment between the reference text and the compared text.
A factually consistent compared text contains only statements that are entailed \
by the reference text.
Annotators should penalize compared texts that contain hallucinated facts.

Rating Scale:
1: Very low consistency - the text contains multiple hallucinated facts \
or significant misalignments with the reference text.
2: Low consistency - the text has several instances of information not supported by \
the reference text.
3: Moderate consistency - the text is mostly consistent but contains a few unsupported statements.
4: Good consistency - the text is largely consistent with minor discrepancies.
5: Excellent consistency - the text is fully consistent with the reference text, \
containing only supported information.

Evaluation Steps:
1. Read the compared text and the reference text carefully.
2. Compare the compared text to the reference text and identify the main points \
of the reference text.
3. Assess how well the compared text covers the main points of the reference text \
and how much irrelevant or redundant information it contains.
4. Assign a consistency score from 1 to 5 based on the provided criteria.
"""


@evaluator(name="text_consistency")
async def text_consistency_evaluator(
    compared: str,
    /,
    reference: str,
) -> float:
    model: ConsistencyScore = await generate_model(
        ConsistencyScore,
        instruction=INSTRUCTION,
        input=f"Reference text: {reference}\n\nCompered text: {compared}",
    )

    return model.score / 5
