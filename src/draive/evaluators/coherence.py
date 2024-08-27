from draive.evaluation import EvaluationScore, evaluator
from draive.generation import generate_text
from draive.types import Multimodal, MultimodalTemplate, xml_tag

__all__ = [
    "coherence_evaluator",
]


INSTRUCTION: str = """\
Assistant is an evaluator scoring the provided content.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate \
the EVALUATED content using solely a coherence metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is coherence - a collective quality of the content.
We align this dimension with the DUC (Document Understanding Conference) quality question of \
structure and coherence, whereby the content should be well-structured and well-organized.
EVALUATED content should not just be a heap of related information, but should build from part
to part into a coherent body of information about the topic.
</EVALUATION_CRITERIA>

<RATING>
Assign a coherence score using value between 0.0 and 4.0 where:
0.0 is very low coherence - the content is chaotic, lacking logical connections between parts.
1.0 is low coherence - some connections are visible, but the overall structure is weak.
2.0 is moderate coherence - the content has a noticeable structure, but with some shortcomings.
3.0 is good coherence - the content is well-organized with minor imperfections.
4.0 is excellent coherence - the content is exemplarily structured, with smooth transitions \
between ideas.
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


@evaluator(name="coherence")
async def coherence_evaluator(
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
