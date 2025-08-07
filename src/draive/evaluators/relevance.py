from typing import cast

from draive.evaluation import EvaluationScore, EvaluationScoreValue, evaluator
from draive.multimodal import Multimodal, MultimodalContent, MultimodalTagElement
from draive.stages import Stage

__all__ = ("relevance_evaluator",)


INSTRUCTION: str = """\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate\
 the EVALUATED content using solely a relevance metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>
{guidelines}
<EVALUATION_CRITERIA>
Evaluated metric is relevance - selection of important parts from the REFERENCE content.
The EVALUATED content should include only important information from the REFERENCE avoiding\
 redundancies and excess information.
</EVALUATION_CRITERIA>

<RATING>
Assign a relevance score using exact name of one of the following values:
- "poor" is very low relevance, the content contains mostly irrelevant or redundant information.
- "fair" is low relevance, the content includes some important points but has\
 significant irrelevant parts.
- "good" is moderate relevance, the content covers most important points but includes\
 some unnecessary information.
- "excellent" is high relevance, the content focuses on important information with minor inclusions\
 of less relevant content.
- "perfect" is very high relevance, the content precisely captures only the most important\
 information from the reference.
Use the "none" value for content that cannot be rated at all.
</RATING>

<FORMAT>
The final result containing only the rating value, HAVE to be put inside a `RESULT`\
 xml tag within the result i.e. `<RESULT>good</RESULT>`.
</FORMAT>
"""


@evaluator(name="relevance")
async def relevance_evaluator(
    evaluated: Multimodal,
    /,
    *,
    reference: Multimodal,
    guidelines: str | None = None,
) -> EvaluationScore:
    if not evaluated:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Input was empty!"},
        )

    if not reference:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Reference was empty!"},
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
            meta={"comment": completion.to_str()},
        )

    else:
        raise ValueError(f"Invalid evaluator result:\n{completion}")
