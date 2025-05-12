from typing import cast

from draive.evaluation import EvaluationScore, EvaluationScoreValue, evaluator
from draive.multimodal import Multimodal, MultimodalContent, MultimodalTagElement
from draive.stages import Stage

__all__ = ("groundedness_evaluator",)


INSTRUCTION: str = """\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate\
 the EVALUATED content using solely a groundedness metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is groundedness - this metric assesses the extent to which the evaluated content\
 directly ties back to and is anchored in the source data. Grounded content should demonstrate a clear\
 and traceable connection to the provided source material, ensuring that the information presented is\
 not only accurate but also faithfully represents the original context. This metric focuses on how well\
 the content reflects the source material without introducing extraneous information, unsupported claims,\
 or interpretations that stray from the source. Groundedness is about maintaining fidelity to the original\
 data, ensuring that every detail and conclusion is rooted in the provided information.
</EVALUATION_CRITERIA>
{guidelines}
<RATING>
Assign a groundedness score using exact name of one of the following values:
- "poor" is very low groundedness, the content is mostly ungrounded with many unsupported claims.
- "fair" is low groundedness, the content contains some accurate information but also significant ungrounded content.
- "good" is moderate groundedness, the content is somewhat grounded but with noticeable ungrounded elements.
- "excellent" is high groundedness, the content is mostly grounded with minimal unverified or unsupported claims.
- "perfect" is very high groundedness, the content is fully grounded, accurately reflecting the source information.
Use the "none" value for content that cannot be rated at all.
</RATING>

<FORMAT>
The final result containing only the numerical score value HAVE to be put inside a `RESULT` \
xml tag within the result i.e. `<RESULT>good</RESULT>`.
</FORMAT>
"""  # noqa: E501


@evaluator(name="groundedness")
async def groundedness_evaluator(
    evaluated: Multimodal,
    /,
    *,
    reference: Multimodal,
    guidelines: str | None = None,
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

    completion: MultimodalContent = await Stage.completion(
        MultimodalContent.of(
            "<REFERENCE>",
            reference,
            "</REFERENCE>\n<EVALUATED>",
            evaluated,
            "</EVALUATED>",
        ),
        instruction=INSTRUCTION.format(
            guidelines=f"\n<GUIDELINES>\n{guidelines}\n</GUIDELINES>\n"
            if guidelines is not None
            else ""
        ),
    ).execute()

    if result := MultimodalTagElement.parse_first(
        "RESULT",
        content=completion,
    ):
        return EvaluationScore.of(
            cast(EvaluationScoreValue, result.content.to_str()),
            comment=completion.to_str(),
        )

    else:
        raise ValueError("Invalid evaluator result:\n%s", completion)
