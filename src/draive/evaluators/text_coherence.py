from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.score import CommonScoreModel
from draive.generation import generate_model

__all__ = [
    "text_coherence_evaluator",
]


INSTRUCTION: str = """\
You will be given a reference text and a compared text based on the reference text.
Your task is to rate the compared text using only the Coherence metric, \
which is described in the Evaluation Criteria.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Coherence (0.0-4.0) - the collective quality of all sentences.
We align this dimension with the DUC (Document Understanding Conference) quality question of \
structure and coherence, whereby the text should be well-structured and well-organized.
The compared text should not just be a heap of related information, but should build from sentence
to sentence into a coherent body of information about a topic.

Rating Scale:
0.0: Very low coherence - the text is chaotic, lacking logical connections between sentences.
1.0: Low coherence - some connections are visible, but the overall structure is weak.
2.0: Moderate coherence - the text has a noticeable structure, but with some shortcomings.
3.0: Good coherence - the text is well-organized with minor imperfections.
4.0: Excellent coherence - the text is exemplarily structured, with smooth transitions \
between ideas.

Evaluation Steps:
1. Read the reference text carefully and identify the main topic and key points.
2. Read the compared text and compare it to the reference text.
Check if the compared text covers the main topic and key points of the reference text, \
and if it presents them in a clear and logical order.
3. Assign a coherence score from 0.0 to 4.0 based on the provided criteria.

Important: The score must be a decimal number from 0.0 to 4.0. 4.0 is the maximum, \
do not exceed this value.
"""

INPUT_TEMPLATE: str = """
<REFERENCE_TEXT>
{reference}
</REFERENCE_TEXT>

<COMPARED_TEXT>
{compared}
</COMPARED_TEXT>
"""


@evaluator(name="text_coherence")
async def text_coherence_evaluator(
    compared: str,
    /,
    reference: str,
) -> EvaluationScore:
    if not compared:
        return EvaluationScore(
            value=0,
            comment="Input text was empty!",
        )

    if not reference:
        return EvaluationScore(
            value=0,
            comment="Reference text was empty!",
        )

    score: CommonScoreModel = await generate_model(
        CommonScoreModel,
        instruction=INSTRUCTION,
        input=INPUT_TEMPLATE.format(
            reference=reference,
            compared=compared,
        ),
        examples=[
            (
                INPUT_TEMPLATE.format(
                    reference=(
                        "Solar energy is a renewable energy source that is gaining popularity. "
                        "Solar panels convert sunlight into electricity. "
                        "This technology is environmentally friendly and can reduce electricity "
                        "bills. However, installing solar panels requires an initial investment "
                        "and is dependent on weather conditions."
                    ),
                    compared=(
                        "Solar panels are on roofs. Energy is important. "
                        "The sun shines brightly. Electricity bills can be high. "
                        "Technology is developing fast. People like to save money."
                    ),
                ),
                CommonScoreModel(score=0.0),
            ),
            (
                INPUT_TEMPLATE.format(
                    reference=(
                        "Coffee is a popular beverage worldwide. "
                        "It's made from roasted coffee beans. Caffeine in coffee "
                        "can boost energy and alertness. However, excessive consumption may "
                        "lead to sleep issues."
                    ),
                    compared=(
                        "Coffee is drunk by many people. It comes from beans that are roasted. "
                        "Caffeine makes you feel more awake. "
                        "Drinking too much coffee might make it hard to sleep. "
                        "Some people add milk or sugar to their coffee."
                    ),
                ),
                CommonScoreModel(score=2.0),
            ),
            (
                INPUT_TEMPLATE.format(
                    reference=(
                        "Honey is a natural sweetener produced by bees. "
                        "It has antibacterial properties and is rich in antioxidants. "
                        "People use honey in cooking, as a spread, and for medicinal "
                        "purposes. However, it's high in calories and should be consumed "
                        "in moderation."
                    ),
                    compared=(
                        "Bees create honey, a natural sweetener with multiple benefits. "
                        "Its antibacterial and antioxidant-rich composition makes it valuable "
                        "for culinary, nutritional, and medicinal uses. While versatile, "
                        "honey's high caloric content necessitates mindful consumption."
                    ),
                ),
                CommonScoreModel(score=4.0),
            ),
        ],
    )
    return score.normalized(divider=4)
