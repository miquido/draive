from draive.evaluation import EvaluationScore, evaluator
from draive.generation import generate_text
from draive.types import Multimodal, MultimodalTemplate, xml_tag

__all__ = [
    "truthfulness_evaluator",
]


INSTRUCTION: str = """\
Assistant is an evaluator scoring the provided content.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate \
the EVALUATED content using solely a truthfulness (factual accuracy) metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is truthfulness (factual accuracy) - this metric evaluates the factual correctness \
of the content, regardless of its relation to the source material. Truthfulness is concerned with whether \
the information presented is accurate in the broader context, ensuring that the facts are correct and reliable, \
and that the content does not perpetuate errors or falsehoods. This metric is more concerned with the overall \
factual integrity of the content, meaning that even if a claim is not directly traceable to the provided source, \
it must still be correct and represent the truth of the matter. Truthfulness is about the correctness of the \
facts themselves, not just their alignment with a specific source.
</EVALUATION_CRITERIA>

<RATING>
Assign a truthfulness (factual accuracy) score using value between 0.0 and 4.0 where:
0.0 is very low truthfulness - the content contains many factual inaccuracies or distortions.
1.0 is low truthfulness - the content has some accurate information but also significant factual errors or misrepresentations.
2.0 is moderate truthfulness - the content is somewhat accurate but with noticeable factual inaccuracies or distortions.
3.0 is good truthfulness - the content is mostly accurate with minimal factual errors or misrepresentations.
4.0 is excellent truthfulness - the content is fully accurate, with no factual errors or misrepresentations.
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


@evaluator(name="truthfulness")
async def truthfulness_evaluator(
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
