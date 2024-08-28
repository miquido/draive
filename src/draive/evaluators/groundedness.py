from draive.evaluation import EvaluationScore, evaluator
from draive.generation import generate_text
from draive.types import Multimodal, MultimodalTemplate, xml_tag

__all__ = [
    "groundedness_evaluator",
]


INSTRUCTION: str = """\
Assistant is an evaluator scoring the provided content.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate \
the EVALUATED content using solely a groundedness metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is groundedness - this metric assesses the extent to which the evaluated content \
directly ties back to and is anchored in the source data. Grounded content should demonstrate a clear \
and traceable connection to the provided source material, ensuring that the information presented is \
not only accurate but also faithfully represents the original context. This metric focuses on how well \
the content reflects the source material without introducing extraneous information, unsupported claims, \
or interpretations that stray from the source. Groundedness is about maintaining fidelity to the original \
data, ensuring that every detail and conclusion is rooted in the provided information.
</EVALUATION_CRITERIA>

<RATING>
Assign a groundedness score using value between 0.0 and 4.0 where:
0.0 is very low groundedness - the content is mostly ungrounded with many unsupported claims.
1.0 is low groundedness - the content contains some accurate information but also significant ungrounded content.
2.0 is moderate groundedness - the content is somewhat grounded but with noticeable ungrounded elements.
3.0 is good groundedness - the content is mostly grounded with minimal unverified or unsupported claims.
4.0 is excellent groundedness - the content is fully grounded, accurately reflecting the source information.
</RATING>

<FORMAT>
The final result containing only the numerical score value HAVE to be put inside a `RESULT` \
xml tag within the result i.e. `<RESULT>score</RESULT>`.
</FORMAT>
"""  # noqa: E501


INPUT_TEMPLATE: MultimodalTemplate = MultimodalTemplate.of(
    "<REFERENCE>",
    ("reference",),
    "</REFERENCE>",
    "<EVALUATED>",
    ("evaluated",),
    "</EVALUATED>",
)


@evaluator(name="groundedness")
async def groundedness_evaluator(
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
