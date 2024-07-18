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
Relevance (0.0-4.0) - selection of important content from the reference text.
The compared text should include only important information from the reference text.
Annotators should penalize compared texts that contain redundancies and excess information.

Rating Scale:
0.0: Very low relevance - the text contains mostly irrelevant or redundant information.
1.0: Low relevance - the text includes some important points but has \
significant irrelevant content.
2.0: Moderate relevance - the text covers most important points but includes \
some unnecessary information.
3.0: Good relevance - the text focuses on important information with minor inclusions \
of less relevant content.
4.0: Excellent relevance - the text precisely captures only the most important information \
from the reference text.

Evaluation Steps:
1. Read the compared text and the reference text carefully.
2. Compare the compared text to the reference text and identify \
the main points of the reference text.
3. Assess how well the compared text covers the main points of the reference text, \
and note any irrelevant or redundant information it contains.
4. Assign a relevance score from 0.0 to 4.0 based on the provided criteria.

Important: The score must be a decimal number from 0.0 to 4.0. 4.0 is the maximum, \
do not exceed this value.
"""

INPUT: str = """
Reference text: {reference}

Compered text: {compared}
"""


@evaluator(name="text_relevance")
async def text_relevance_evaluator(
    compared: str,
    /,
    reference: str,
) -> float:
    model: RelevanceScore = await generate_model(
        RelevanceScore,
        instruction=INSTRUCTION,
        input=f"Reference text: {reference}\n\nCompered text: {compared}",
        examples=[
            (
                INPUT.format(
                    reference=(
                        "The sun is the star at the center of our solar system. "
                        "It provides light and heat to Earth."
                    ),
                    compared=(
                        "Stars twinkle in the night sky. Some people believe in astrology. "
                        "The moon orbits the Earth. Astronauts have been to space. "
                        "Solar panels use energy from the sun."
                    ),
                ),
                RelevanceScore(score=0.0),
            ),
            (
                INPUT.format(
                    reference=(
                        "Elephants are the largest land animals. They have long trunks and tusks. "
                        "Elephants live in herds and are known for their intelligence."
                    ),
                    compared=(
                        "Elephants are very big animals. They use their trunks to grab food "
                        "and water. Elephants live together in groups. They're smart and have "
                        "good memories. Some people ride elephants in zoos, "
                        "but this can be harmful to the animals."
                    ),
                ),
                RelevanceScore(score=2.0),
            ),
            (
                INPUT.format(
                    reference=(
                        "Bicycles are a popular mode of transportation. They are eco-friendly "
                        "and provide exercise. However, cyclists need to follow "
                        "traffic rules for safety."
                    ),
                    compared=(
                        "Bicycles are widely used for travel. "
                        "They don't pollute and help people stay fit. "
                        "Cyclists must obey traffic laws to stay safe."
                    ),
                ),
                RelevanceScore(score=4.0),
            ),
        ],
    )

    return model.score / 4
