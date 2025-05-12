from typing import cast

from draive.evaluation import EvaluationScore, EvaluationScoreValue, evaluator
from draive.multimodal import Multimodal, MultimodalContent, MultimodalTagElement
from draive.stages import Stage

__all__ = ("fluency_evaluator",)


INSTRUCTION: str = """\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Carefully examine provided CONTENT, then rate it using solely a\
 fluency metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is fluency - the quality of the content in terms of grammar, spelling,\
 punctuation, content choice, and overall structure.
</EVALUATION_CRITERIA>
{guidelines}
<RATING>
Assign a fluency score using exact name of one of the following values:
- "poor" is very low fluency, the content has many errors that make it hard to understand or\
 look unnatural.
- "good" is moderate fluency, the content has some errors that affect the clarity or smoothness,\
 but the main points are still comprehensible.
- "perfect" is very high fluency - the content has few or no errors and is easy to read and follow.
Use the "none" value for content that cannot be rated at all.
</RATING>

<FORMAT>
The final result containing only the numerical score value HAVE to be put inside a `RESULT`\
 xml tag within the result i.e. `<RESULT>good</RESULT>`.
</FORMAT>
"""


@evaluator(name="fluency")
async def fluency_evaluator(
    content: Multimodal,
    /,
    guidelines: str | None = None,
) -> EvaluationScore:
    if not content:
        return EvaluationScore(
            value=0,
            comment="Input was empty!",
        )

    completion: MultimodalContent = await Stage.completion(
        MultimodalContent.of(
            "<CONTENT>",
            content,
            "</CONTENT>",
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
