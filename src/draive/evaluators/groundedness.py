from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import FORMAT_INSTRUCTION, extract_evaluation_result
from draive.multimodal import Multimodal, MultimodalContent
from draive.stages import Stage

__all__ = ("groundedness_evaluator",)


INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a groundedness metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is groundedness - this metric assesses the extent to which the evaluated content directly ties back to and is anchored in the source data. Grounded content should demonstrate a clear and traceable connection to the provided source material, ensuring that the information presented is not only accurate but also faithfully represents the original context. This metric focuses on how well the content reflects the source material without introducing extraneous information, unsupported claims, or interpretations that stray from the source. Groundedness is about maintaining fidelity to the original data, ensuring that every detail and conclusion is rooted in the provided information.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a groundedness score using exact name of one of the following values:
- "poor" is very low groundedness, the content is mostly ungrounded with many unsupported claims.
- "fair" is low groundedness, the content contains some accurate information but also significant ungrounded content.
- "good" is moderate groundedness, the content is somewhat grounded but with noticeable ungrounded elements.
- "excellent" is high groundedness, the content is mostly grounded with minimal unverified or unsupported claims.
- "perfect" is very high groundedness, the content is fully grounded, accurately reflecting the source information.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
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
