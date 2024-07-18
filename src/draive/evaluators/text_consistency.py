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
Consistency(0.0-4.0) - the factual alignment between the reference text and the compared text.
A factually consistent compared text contains only statements that are entailed \
by the reference text.
Annotators should penalize compared texts that contain hallucinated facts.

Rating Scale:
0.0: Very low consistency - the text contains multiple hallucinated facts \
or significant misalignments with the reference text.
1.0: Low consistency - the text has several instances of information not supported by \
the reference text.
2.0: Moderate consistency - the text is mostly consistent but contains a few unsupported statements.
3.0: Good consistency - the text is largely consistent with minor discrepancies.
4.0: Excellent consistency - the text is fully consistent with the reference text, \
containing only supported information.

Evaluation Steps:
1. Read the compared text and the reference text carefully.
2. Compare the compared text to the reference text and identify the main points \
of the reference text.
3. Assess how well the compared text covers the main points of the reference text \
and how much irrelevant or redundant information it contains.
4. Assign a consistency score from 0.0 to 4.0 based on the provided criteria.

Important: The score must be a decimal number from 0.0 to 4.0. 4.0 is the maximum, \
do not exceed this value.
"""

INPUT: str = """
Reference text: {reference}

Compered text: {compared}
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
        examples=[
            (
                INPUT.format(
                    reference=(
                        "Dolphins are intelligent marine mammals. They use echolocation "
                        "to navigate and hunt. Dolphins live in social groups called pods."
                    ),
                    compared=(
                        "Dolphins are smart fish that can fly short distances. They use sonar "
                        "to talk to whales. Dolphins live in families and go to school "
                        "to learn hunting techniques."
                    ),
                ),
                ConsistencyScore(score=0.0),
            ),
            (
                INPUT.format(
                    reference=(
                        "Coffee is a popular beverage worldwide. "
                        "It's made from roasted coffee beans. Caffeine in coffee "
                        "can boost energy and alertness. However, excessive consumption may "
                        "lead to sleep issues."
                    ),
                    compared=(
                        "Coffee is a widely consumed drink around the world. It's produced "
                        "by roasting coffee beans. The caffeine in coffee can increase energy "
                        "levels and improve alertness. However, drinking too much coffee might "
                        "cause sleep problems. Coffee is also known to improve memory and reduce "
                        "the risk of certain diseases."
                    ),
                ),
                ConsistencyScore(score=2.0),
            ),
            (
                INPUT.format(
                    reference=(
                        "Photosynthesis is the process by which plants use sunlight to "
                        "produce energy. It requires water, carbon dioxide, and chlorophyll. "
                        "Oxygen is released as a byproduct of photosynthesis."
                    ),
                    compared=(
                        "Plants carry out photosynthesis to create energy from sunlight. "
                        "This process needs water, carbon dioxide, and the green pigment "
                        "chlorophyll. As plants photosynthesize, "
                        "they release oxygen into the environment."
                    ),
                ),
                ConsistencyScore(score=4.0),
            ),
        ],
    )
    return model.score / 4
