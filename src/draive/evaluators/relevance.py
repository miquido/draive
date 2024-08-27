from draive.evaluation import EvaluationScore, evaluator
from draive.generation import generate_text
from draive.types import Multimodal, MultimodalTemplate, xml_tag

__all__ = [
    "relevance_evaluator",
]


INSTRUCTION: str = """\
Assistant is an evaluator scoring the provided content.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate \
the EVALUATED content using solely a relevance metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is relevance - selection of important parts from the REFERENCE content.
The EVALUATED content should include only important information from the REFERENCE avoiding \
redundancies and excess information.
</EVALUATION_CRITERIA>

<RATING>
Assign a relevance score using value between 0.0 and 4.0 where:
0.0 is very low relevance - the content contains mostly irrelevant or redundant information.
1.0 is low coverage - the content includes some important points but has \
significant irrelevant parts.
2.0 is moderate relevance - the content covers most important points but includes \
some unnecessary information.
3.0 is good relevance - the content focuses on important information with minor inclusions \
of less relevant content.
4.0 is excellent relevance - the content precisely captures only the most important information \
from the reference.
</RATING>

<FORMAT>
The final result containing only the numerical score value HAVE to be put inside a `RESULT` \
xml tag within the result i.e. `<RESULT>score</RESULT>`.
</FORMAT>
"""


INPUT_TEMPLATE: MultimodalTemplate = MultimodalTemplate.of(
    "<REFERENCE>",
    ("reference",),
    "</REFERENCE>",
    "<EVALUATED>",
    ("evaluated",),
    "</EVALUATED>",
)


@evaluator(name="relevance")
async def relevance_evaluator(
    evaluated: Multimodal,
    /,
    reference: Multimodal,
) -> EvaluationScore:
    if not evaluated:
        return EvaluationScore(
            value=0,
            comment="Input was empty!",
        )

    if not reference:
        return EvaluationScore(
            value=0,
            comment="Reference was empty!",
        )

    if result := xml_tag(
        "RESULT",
        source=await generate_text(
            instruction=INSTRUCTION,
            input=INPUT_TEMPLATE.format(
                reference=reference,
                evaluated=evaluated,
            ),
        ),
        conversion=str,
    ):
        return EvaluationScore(
            value=float(result) / 4,
            comment=None,
        )

    else:
        raise ValueError("Invalid result")
