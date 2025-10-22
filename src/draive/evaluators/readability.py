from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import FORMAT_INSTRUCTION, extract_evaluation_result
from draive.multimodal import Multimodal, MultimodalContent
from draive.stages import Stage

__all__ = ("readability_evaluator",)


INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Carefully examine provided CONTENT, then rate it using solely a readability metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is readability - the ease with which a reader can understand the content.
A readable content uses clear and concise language, is well-structured,
and avoids complex or convoluted elements.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a readability score using exact name of one of the following values:
- "poor" is very low readability, the content is extremely difficult to understand, with complex language and convoluted structure.
- "fair" is low readability, the content is challenging to read, with frequent use of complex sentences, unclear language or irrelevant parts.
- "good" is moderate readability, the content is somewhat clear but has some areas that are difficult to understand.
- "excellent" is high readability, the content is mostly clear and easy to read, with minor instances of complexity.
- "perfect" is very high readability, the content is highly clear, concise, and easy to understand throughout.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501


@evaluator(name="readability")
async def readability_evaluator(
    evaluated: Multimodal,
    /,
    guidelines: str | None = None,
) -> EvaluationScore:
    if not evaluated:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Input was empty!"},
        )

    return extract_evaluation_result(
        await Stage.completion(
            MultimodalContent.of(
                "<CONTENT>",
                evaluated,
                "</CONTENT>",
            ),
            instructions=INSTRUCTION.format(
                guidelines=f"\n<GUIDELINES>\n{guidelines}\n</GUIDELINES>\n" if guidelines else "",
            ),
        ).execute()
    )
