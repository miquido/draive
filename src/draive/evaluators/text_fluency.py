from draive.evaluation import evaluator
from draive.generation import generate_model
from draive.parameters import DataModel

__all__ = [
    "text_fluency_evaluator",
]


class FluencyScore(DataModel):
    score: float
    comment: str | None = None


INSTRUCTION: str = """\
You will be given a text. Your task is to rate this text using only the Fluency metric, \
which is described in the Evaluation Criteria.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Fluency (1-3) - the quality of the text in terms of grammar, spelling, punctuation, word choice, \
and sentence structure.

Rating Scale:
1: Poor - the text has many errors that make it hard to understand or sound unnatural.
2: Fair - the text has some errors that affect the clarity or smoothness of the text, \
but the main points are still comprehensible.
3: Good - the text has few or no errors and is easy to read and follow.

Evaluation Steps:
1. Read the text and evaluate its fluency based on the given criteria.
2. Assign a fluency score from 1 to 3 based on the provided criteria.
"""

@evaluator(name="fluency")
async def text_fluency_evaluator(
    text: str,
    /,
) -> float:
    model: FluencyScore = await generate_model(
        FluencyScore,
        instruction=INSTRUCTION,
        input=text,
    )

    return model.score/3
