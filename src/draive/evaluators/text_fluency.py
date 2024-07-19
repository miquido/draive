from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.score import CommonScoreModel
from draive.generation import generate_model

__all__ = [
    "text_fluency_evaluator",
]


INSTRUCTION: str = """\
You will be given a text. Your task is to rate this text using only the Fluency metric, \
which is described in the Evaluation Criteria.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Fluency (0.0-2.0) - the quality of the text in terms of grammar, spelling, punctuation, \
word choice, and sentence structure.

Rating Scale:
0.0: Poor - the text has many errors that make it hard to understand or sound unnatural.
1.0: Fair - the text has some errors that affect the clarity or smoothness of the text, \
but the main points are still comprehensible.
2.0: Good - the text has few or no errors and is easy to read and follow.

Evaluation Steps:
1. Read the text and evaluate its fluency based on the given criteria.
2. Assign a fluency score from 0.0 to 2.0 based on the provided criteria.

Important: The score must be a decimal number from 0.0 to 2.0. 2.0 is the maximum, \
do not exceed this value.
"""


INPUT_TEMPLATE: str = """
<TEXT>
{text}
</TEXT>
"""


@evaluator(name="text_fluency")
async def text_fluency_evaluator(
    text: str,
    /,
) -> EvaluationScore:
    if not text:
        return EvaluationScore(
            value=0,
            comment="Input text was empty!",
        )

    score: CommonScoreModel = await generate_model(
        CommonScoreModel,
        instruction=INSTRUCTION,
        input=INPUT_TEMPLATE.format(text=text),
        examples=[
            (
                INPUT_TEMPLATE.format(
                    text=(
                        "The cat sitted on mat. It were very comfrotable. "
                        "The sun shine bright in sky."
                    ),
                ),
                CommonScoreModel(score=0.0),
            ),
            (
                INPUT_TEMPLATE.format(
                    text=(
                        "The movie was good, but I didn't liked the ending. "
                        "It left me feeling confuse and unsatisfied."
                    ),
                ),
                CommonScoreModel(score=1.0),
            ),
            (
                INPUT_TEMPLATE.format(
                    text=(
                        "The concert last night was amazing. "
                        "The band played all their hit songs, and the crowd was energetic "
                        "throughout the performance."
                    ),
                ),
                CommonScoreModel(score=2.0),
            ),
        ],
    )

    return score.normalized(divider=2)
