from draive.evaluation import evaluator
from draive.generation import generate_model
from draive.parameters import DataModel

__all__ = [
    "ReadabilityScore",
    "text_readability_evaluator",
]


class ReadabilityScore(DataModel):
    score: float
    comment: str | None = None


INSTRUCTION: str = """\
You will be given a text. Your task is to rate this text based on one metric.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Readability (1-5) - the ease with which a reader can understand the text.
A readable text uses clear and concise language, is well-structured,
and avoids complex or convoluted sentences. Annotators should penalize texts that are difficult to read or understand.

Evaluation Steps:
1. Read the text carefully and evaluate how easy it is to read and understand.
2. Consider the language used in the text, including clarity, simplicity, and sentence structure.
3. Assess whether the text is well-structured and free from complex or convoluted sentences.
4. Assign a readability score from 1 to 5, where 1 is the lowest and 5 is the highest, based on the Evaluation Criteria.
"""  # noqa: E501

@evaluator(name="Readability Evaluator", threshold=0.75)
async def text_readability_evaluator(
    text: str,
    /,
) -> float:
    model: ReadabilityScore = await generate_model(
        ReadabilityScore,
        instruction=INSTRUCTION,
        input=text,
    )

    return model.score/5
