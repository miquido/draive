from draive.evaluation import EvaluationScore, evaluator
from draive.generation import generate_text
from draive.types import Multimodal, MultimodalTemplate, xml_tag

__all__ = [
    "fluency_evaluator",
]


INSTRUCTION: str = """\
Assistant is an evaluator scoring the provided content.

<INSTRUCTION>
Carefully examine provided CONTENT, then rate it using solely a \
fluency metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is fluency - the quality of the content in terms of grammar, spelling, \
punctuation, content choice, and overall structure.
</EVALUATION_CRITERIA>

<RATING>
Assign a fluency score using value between 0.0 and 2.0 where:
0.0 is poor fluency - the content has many errors that make it hard to understand or look unnatural.
1.0 is fair fluency - the content has some errors that affect the clarity or smoothness, \
but the main points are still comprehensible.
2.0 is good fluency - the content has few or no errors and is easy to read and follow.
</RATING>

<FORMAT>
The final result containing only the numerical score value HAVE to be put inside a `RESULT` \
xml tag within the result i.e. `<RESULT>score</RESULT>`.
</FORMAT>
"""


INPUT_TEMPLATE: MultimodalTemplate = MultimodalTemplate.of(
    "<CONTENT>",
    ("content",),
    "</CONTENT>",
)


@evaluator(name="fluency")
async def fluency_evaluator(
    content: Multimodal,
    /,
) -> EvaluationScore:
    if not content:
        return EvaluationScore(
            value=0,
            comment="Input was empty!",
        )

    if result := xml_tag(
        "RESULT",
        source=await generate_text(
            instruction=INSTRUCTION,
            input=INPUT_TEMPLATE.format(
                content=content,
            ),
        ),
        conversion=str,
    ):
        return EvaluationScore(
            value=float(result) / 2,
            comment=None,
        )

    else:
        raise ValueError("Invalid result")
