from draive.evaluation import EvaluationScore, evaluator
from draive.multimodal import Multimodal, MultimodalContent, MultimodalTagElement
from draive.steps import steps_completion

__all__ = [
    "coverage_evaluator",
]


INSTRUCTION: str = """\
Assistant is an evaluator scoring the provided content.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate \
the EVALUATED content using solely a coverage metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is coverage - the extent to which the EVALUATED content includes all \
the key points from the REFERENCE content.
EVALUATED content with good coverage includes all the important information from \
the REFERENCE content without omitting critical points.
</EVALUATION_CRITERIA>

<RATING>
Assign a coverage score using value between 0.0 and 4.0 where:
0.0 is very low coverage - the content misses most key points from the reference content.
1.0 is low coverage - the content includes some key points but omits several important ones.
2.0 is moderate coverage - the content covers most key points but misses a few important details.
3.0 is good coverage - the content includes nearly all key points with minor omissions.
4.0 is excellent coverage - the content comprehensively covers all key points from the reference content.
</RATING>

<FORMAT>
The final result containing only the numerical score value HAVE to be put inside a `RESULT` \
xml tag within the result i.e. `<RESULT>score</RESULT>`.
</FORMAT>
"""  # noqa: E501


@evaluator(name="coverage")
async def coverage_evaluator(
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

    completion: MultimodalContent = await steps_completion(
        MultimodalContent.of(
            "<REFERENCE>",
            reference,
            "</REFERENCE>\n<EVALUATED>",
            evaluated,
            "</EVALUATED>",
        ),
        instruction=INSTRUCTION,
    )

    if result := MultimodalTagElement.parse_first(
        completion,
        tag="RESULT",
    ):
        return EvaluationScore(
            value=float(result.content.as_string()) / 4,
            comment=None,
        )

    else:
        raise ValueError("Invalid result")
