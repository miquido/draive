from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.score import CommonScoreModel
from draive.generation import generate_model

__all__ = [
    "text_coverage_evaluator",
]


INSTRUCTION: str = """\
You will be given a reference text and a compared text based on the reference text.
Your task is to rate the compared text using only the Coverage metric, \
which is described in the Evaluation Criteria.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Coverage (0.0-4.0) - the extent to which the compared text includes all \
the key points from the reference text.
A compared text with good coverage includes all the important information from \
the reference text without omitting critical points.
Annotators should penalize compared texts that miss significant content.

Rating Scale:
0.0: Very low coverage - the text misses most key points from the reference text.
1.0: Low coverage - the text includes some key points but omits several important ones.
2.0: Moderate coverage - the text covers most key points but misses a few important details.
3.0: Good coverage - the text includes nearly all key points with minor omissions.
4.0: Excellent coverage - the text comprehensively covers all key points from the reference text.

Evaluation Steps:
1. Read the reference text carefully and identify all key points and important information.
2. Read the compared text and compare it to the reference text. \
Check if the compared text includes all the key points and important information \
from the reference text.
3. Assess how well the compared text covers the reference text, \
and if any critical points are missing.
4. Assign a coverage score from 0.0 to 4.0 based on the provided criteria.

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


@evaluator(name="text_coverage")
async def text_coverage_evaluator(
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
                        "Smartphones are versatile devices. They can make calls, send messages, "
                        "access the internet, take photos, and run various apps. "
                        "Many people use smartphones for work and entertainment. "
                        "However, excessive use can lead to addiction and sleep problems."
                    ),
                    compared=(
                        "Smartphones can make calls and send messages. They are popular devices."
                    ),
                ),
                CommonScoreModel(score=0.0),
            ),
            (
                INPUT_TEMPLATE.format(
                    reference=(
                        "Recycling helps protect the environment. It reduces waste in landfills, "
                        "conserves natural resources, and saves energy. Common recyclable items "
                        "include paper, plastic, glass, and metal. Many cities have recycling "
                        "programs, but individual participation is crucial for success."
                    ),
                    compared=(
                        "Recycling is good for the environment. "
                        "It reduces waste and saves resources. "
                        "People can recycle things like paper and plastic. "
                        "Many cities have recycling programs."
                    ),
                ),
                CommonScoreModel(score=2.0),
            ),
            (
                INPUT_TEMPLATE.format(
                    reference=(
                        "Regular exercise is important for health. It strengthens the heart, "
                        "builds muscle, and improves flexibility. Exercise can also reduce stress "
                        "and boost mood. Experts recommend at least 30 minutes of moderate "
                        "activity most days of the week. Walking, swimming, and cycling are "
                        "good options for many people."
                    ),
                    compared=(
                        "Regular exercise is crucial for maintaining good health. "
                        "It has many benefits, including strengthening the heart, "
                        "building muscle, and enhancing flexibility. Exercise also has "
                        "mental health benefits, such as reducing stress and improving mood. "
                        "Health experts advise doing at least 30 minutes of moderate exercise "
                        "on most days. Some popular and accessible forms of exercise "
                        "include walking, swimming, and cycling."
                    ),
                ),
                CommonScoreModel(score=4.0),
            ),
        ],
    )

    return score.normalized(divider=4)
