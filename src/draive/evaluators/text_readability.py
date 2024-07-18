from draive.evaluation import evaluator
from draive.generation import generate_model
from draive.parameters import DataModel

__all__ = [
    "text_readability_evaluator",
]


class ReadabilityScore(DataModel):
    score: float
    comment: str | None = None


INSTRUCTION: str = """\
You will be given a text. Your task is to rate this text using only the Readability metric, \
which is described in the Evaluation Criteria.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Readability (0.0-4.0) - the ease with which a reader can understand the text.
A readable text uses clear and concise language, is well-structured,
and avoids complex or convoluted sentences. Annotators should penalize texts that \
are difficult to read or understand.

Rating Scale:
0.0: Very low readability - the text is extremely difficult to understand, \
with complex language and convoluted structure.
1.0: Low readability - the text is challenging to read, with frequent use of \
complex sentences or unclear language.
2.0: Moderate readability - the text is somewhat clear but has some areas \
that are difficult to understand.
3.0: Good readability - the text is mostly clear and easy to read, with minor instances \
of complexity.
4.0: Excellent readability - the text is highly clear, concise, and easy to understand throughout.

Evaluation Steps:
1. Read the text carefully and evaluate how easy it is to read and understand.
2. Consider the language used in the text, including clarity, simplicity, and sentence structure.
3. Assess whether the text is well-structured and free from complex or convoluted sentences.
4. Assign a readability score from 0.0 to 4.0 based on the provided criteria.

Important: The score must be a decimal number from 0.0 to 4.0. 4.0 is the maximum, \
do not exceed this value.
"""

INPUT: str = """
Text: {text}
"""


@evaluator(name="text_readability")
async def text_readability_evaluator(
    text: str,
    /,
) -> float:
    model: ReadabilityScore = await generate_model(
        ReadabilityScore,
        instruction=INSTRUCTION,
        input=text,
        examples=[
            (
                INPUT.format(
                    text=(
                        "The canine species, frequently domesticated for companionship purposes, "
                        "exhibit characteristics of fidelity and ludic propensities that engender "
                        "their widespread appeal among human populations as domestic "
                        "animal companions."
                    ),
                ),
                ReadabilityScore(score=0.0),
            ),
            (
                INPUT.format(
                    text=(
                        "Pizza, a widely consumed dish, consists of a circular bread foundation "
                        "adorned with various ingredients. Typically, it includes a layer of "
                        "tomato-based sauce and cheese, though additional toppings may be "
                        "incorporated to suit individual preferences."
                    ),
                ),
                ReadabilityScore(score=2.0),
            ),
            (
                INPUT.format(
                    text=(
                        "Exercise is good for health. It helps maintain fitness and reduces stress."
                    ),
                ),
                ReadabilityScore(score=4.0),
            ),
        ],
    )

    return model.score / 4
