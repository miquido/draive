from draive.evaluation import EvaluationScore, evaluator
from draive.generation import generate_text
from draive.types import Multimodal, MultimodalTemplate, xml_tag

__all__ = [
    "readability_evaluator",
]


INSTRUCTION: str = """\
Assistant is an evaluator scoring the provided content.

<INSTRUCTION>
Carefully examine provided CONTENT, then rate it using solely a \
readability metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is readability - the ease with which a reader can understand the content.
A readable content uses clear and concise language, is well-structured,
and avoids complex or convoluted elements.
</EVALUATION_CRITERIA>

<RATING>
Assign a readability score using value between 0.0 and 4.0 where:
0.0 is very low readability - the content is extremely difficult to understand, \
with complex language and convoluted structure.
1.0 is low readability - the content is challenging to read, with frequent use of \
complex sentences, unclear language or irrelevant parts.
2.0 is moderate readability - the content is somewhat clear but has some areas \
that are difficult to understand.
3.0 is good readability - the content is mostly clear and easy to read, with minor instances \
of complexity.
4.0 is excellent readability - the content is highly clear, concise, and easy to understand throughout.
</RATING>

<FORMAT>
The final result containing only the numerical score value HAVE to be put inside a `RESULT` \
xml tag within the result i.e. `<RESULT>score</RESULT>`.
</FORMAT>
"""  # noqa: E501


INPUT_TEMPLATE: MultimodalTemplate = MultimodalTemplate.of(
    "<CONTENT>",
    ("content",),
    "</CONTENT>",
)


@evaluator(name="readability")
async def readability_evaluator(
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
            value=float(result) / 4,
            comment=None,
        )

    else:
        raise ValueError("Invalid result")
