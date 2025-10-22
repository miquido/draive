from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import FORMAT_INSTRUCTION, extract_evaluation_result
from draive.multimodal import Multimodal, MultimodalContent
from draive.stages import Stage

__all__ = ("coherence_evaluator",)


INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Compare the REFERENCE and the EVALUATED content by carefully examining them, then rate the EVALUATED content using solely a coherence metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is coherence - a collective quality of the content.
We align this dimension with the DUC (Document Understanding Conference) quality question of structure and coherence, whereby the content should be well-structured and well-organized.
EVALUATED content should not just be a heap of related information, but should build from part to part into a coherent body of information about the topic.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a coherence score using exact name of one of the following values:
- "poor" is very low coherence, the content is chaotic, lacking logical connections between parts.
- "fair" is low coherence, some connections are visible, but the overall structure is weak.
- "good" is moderate coherence, the content has a noticeable structure, but with some shortcomings.
- "excellent" is high coherence, the content is well-organized with minor imperfections.
- "perfect" is very high coherence, the content is exemplarily structured, with smooth transitions between ideas.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501


@evaluator(name="coherence")
async def coherence_evaluator(
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
