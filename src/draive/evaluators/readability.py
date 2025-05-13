from typing import cast

from draive.evaluation import EvaluationScore, EvaluationScoreValue, evaluator
from draive.multimodal import Multimodal, MultimodalContent, MultimodalTagElement
from draive.stages import Stage

__all__ = ("readability_evaluator",)


INSTRUCTION: str = """\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Carefully examine provided CONTENT, then rate it using solely a\
 readability metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is readability - the ease with which a reader can understand the content.
A readable content uses clear and concise language, is well-structured,
and avoids complex or convoluted elements.
</EVALUATION_CRITERIA>
{guidelines}
<RATING>
Assign a readability score using exact name of one of the following values:
- "poor" is very low readability, the content is extremely difficult to understand,\
 with complex language and convoluted structure.
- "fair" is low readability, the content is challenging to read, with frequent use of\
 complex sentences, unclear language or irrelevant parts.
- "good" is moderate readability, the content is somewhat clear but has some areas\
 that are difficult to understand.
- "excellent" is high readability, the content is mostly clear and easy to read, with minor instances\
 of complexity.
- "perfect" is very high readability, the content is highly clear, concise, and easy to understand throughout.
Use the "none" value for content that cannot be rated at all.
</RATING>

<FORMAT>
The final result containing only the numerical score value HAVE to be put inside a `RESULT`\
 xml tag within the result i.e. `<RESULT>good</RESULT>`.
</FORMAT>
"""  # noqa: E501


@evaluator(name="readability")
async def readability_evaluator(
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
