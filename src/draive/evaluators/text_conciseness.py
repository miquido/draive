from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.score import CommonScoreModel
from draive.generation import generate_model

__all__ = [
    "text_conciseness_evaluator",
]


INSTRUCTION: str = """\
You will be given a reference text and a compared text based on the reference text.
Your task is to rate the compared text using only the Conciseness metric, \
which is described in the Evaluation Criteria.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Conciseness (0.0-4.0) - the extent to which the compared text is brief and to the point \
while still covering all key information.
A concise compared text avoids unnecessary details and repetition.
Annotators should penalize compared texts that are overly verbose or include irrelevant information.

Rating Scale:
0.0: Very low conciseness - the text is excessively verbose with much irrelevant information.
1.0: Low conciseness - the text contains unnecessary details and some irrelevant information.
2.0: Moderate conciseness - the text is somewhat concise but could be more focused.
3.0: Good conciseness - the text is mostly concise with minimal unnecessary information.
4.0: Excellent conciseness - the text is highly concise, containing only essential information.

Evaluation Steps:
1. Read the derived text and the reference text carefully.
2. Compare the compared text to the reference text and identify the main \
points of the reference text.
3. Assess how well the compared text covers the main points of the reference text, \
and how much irrelevant or redundant information it contains.
4. Assign a conciseness score from 0.0 to 4.0 based on the provided criteria.

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


@evaluator(name="text_conciseness")
async def text_conciseness_evaluator(
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
                        "bills. However,installing solar panels requires an initial investment and "
                        "is dependent on weather conditions."
                    ),
                    compared=(
                        "Did you know that solar energy is becoming super popular these days? "
                        "It's this amazing, eco-friendly way to make electricity using "
                        "the sun's rays. People are getting really excited about it! Basically, "
                        "you put these special panels on your roof, and they soak up the sunlight "
                        "like a sponge. Then, through some pretty cool science stuff, "
                        "they turn that sunlight into electricity you can use in your house. "
                        "It's pretty neat, right? And get this - it can actually help you save "
                        "money on your electricity bills in the long run. But here's the thing: "
                        "you've got to shell out some cash upfront to get those panels installed. "
                        "It's kind of like buying a fancy coffee machine - costs a bit at first, "
                        "but then you save on all those coffee shop visits."
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
                        "Coffee is a widely consumed beverage made from roasted coffee beans. "
                        "It contains caffeine, which can enhance energy and alertness. However, "
                        "drinking too much coffee may cause sleep problems. "
                        "People enjoy coffee for its taste and stimulating effects, but it's "
                        "important to consume it in moderation."
                    ),
                ),
                CommonScoreModel(score=2.0),
            ),
            (
                INPUT_TEMPLATE.format(
                    reference=(
                        "The water cycle, also known as the hydrologic cycle, "
                        "describes the continuous movement of water within the Earth and "
                        "atmosphere. It involves processes such as evaporation, condensation, "
                        "precipitation, and runoff."
                    ),
                    compared=(
                        "The water cycle is the continuous movement of water on Earth. "
                        "It includes evaporation, condensation, precipitation, and runoff."
                    ),
                ),
                CommonScoreModel(score=4.0),
            ),
        ],
    )
    return score.normalized(divider=4)
