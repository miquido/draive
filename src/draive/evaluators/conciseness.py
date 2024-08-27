from draive.evaluation import EvaluationScore, evaluator
from draive.generation import generate_text
from draive.types import Multimodal, MultimodalTemplate, xml_tag

__all__ = [
    "conciseness_evaluator",
]


INSTRUCTION: str = """\
Assistant is an evaluator scoring the provided content.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate \
the EVALUATED content using solely a conciseness metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is conciseness - a extent to which the EVALUATED content is brief and to the \
point while still covering all key information.
A concise content avoids unnecessary details and repetition, also avoiding being overly verbose \
or include irrelevant information.
</EVALUATION_CRITERIA>

<RATING>
Assign a conciseness score using value between 0.0 and 4.0 where:
0.0 is very low conciseness - the content is excessively verbose with much irrelevant information.
1.0 is low conciseness - the content contains unnecessary details and some irrelevant information.
2.0 is moderate conciseness - the content is somewhat concise but could be more focused.
3.0 is good conciseness - the content is mostly concise with minimal unnecessary information.
4.0 is excellent conciseness - the content is highly concise, containing only essential information.
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


@evaluator(name="conciseness")
async def conciseness_evaluator(
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
