from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import FORMAT_INSTRUCTION, extract_evaluation_result
from draive.multimodal import Multimodal, MultimodalContent
from draive.stages import Stage

__all__ = ("fluency_evaluator",)


INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Carefully examine provided CONTENT, then rate it using solely a fluency metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is fluency - the quality of the content in terms of grammar, spelling, punctuation, content choice, and overall structure.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a fluency score using exact name of one of the following values:
- "poor" is very low fluency, the content has many errors that make it hard to understand or look unnatural.
- "fair" is low fluency, the content partially reads well but contains noticeable issues affecting clarity or naturalness.
- "good" is moderate fluency, the content has some errors that affect clarity or smoothness, but the main points remain comprehensible.
- "excellent" is high fluency, the content reads well with only minor issues that rarely impact understanding.
- "perfect" is very high fluency - the content has few or no errors and is easy to read and follow.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501


@evaluator(name="fluency")
async def fluency_evaluator(
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
