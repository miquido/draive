from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import FORMAT_INSTRUCTION, extract_evaluation_result
from draive.multimodal import Multimodal, MultimodalContent
from draive.stages import Stage

__all__ = ("conciseness_evaluator",)


INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a conciseness metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is conciseness — the extent to which the EVALUATED content is brief and to the point while still covering all key information.
Concise content avoids unnecessary details and repetition, while not being overly verbose or including irrelevant information.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a conciseness score using exact name of one of the following values:
- "poor" is very low conciseness, the content is excessively verbose with much irrelevant information.
- "fair" is low conciseness, the content contains unnecessary details and some irrelevant information.
- "good" is moderate conciseness, the content is somewhat concise but could be more focused.
- "excellent" is high conciseness, the content is mostly concise with minimal unnecessary information.
- "perfect" is very high conciseness, the content is highly concise, containing only essential information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501


@evaluator(name="conciseness")
async def conciseness_evaluator(
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

    return extract_evaluation_result(
        await Stage.completion(
            MultimodalContent.of(
                "<REFERENCE>",
                reference,
                "</REFERENCE>\n<EVALUATED>",
                evaluated,
                "</EVALUATED>",
            ),
            instructions=INSTRUCTION.format(
                guidelines=f"\n<GUIDELINES>\n{guidelines}\n</GUIDELINES>\n" if guidelines else "",
            ),
        ).execute()
    )
