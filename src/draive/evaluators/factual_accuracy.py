from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import FORMAT_INSTRUCTION, extract_evaluation_result
from draive.multimodal import Multimodal, MultimodalContent
from draive.stages import Stage

__all__ = ("factual_accuracy_evaluator",)


INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Carefully examine provided CONTENT, then rate it using solely a factual accuracy metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is factual accuracy - the extent to which the content contains factually correct information based on established knowledge, verifiable facts, and reliable sources. This evaluates the correctness of claims, data, statements, and assertions made in the content, regardless of any reference material. The evaluation should consider well-established facts, scientific consensus, historical accuracy, and generally accepted knowledge.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a factual accuracy score using exact name of one of the following values:
- "poor" is very low factual accuracy, the content contains many significant factual errors or false information.
- "fair" is low factual accuracy, the content has several factual inaccuracies or questionable claims mixed with some correct information.
- "good" is moderate factual accuracy, the content is mostly factually correct but contains some minor inaccuracies or unverified claims.
- "excellent" is high factual accuracy, the content is largely factually correct with minimal or very minor factual issues.
- "perfect" is very high factual accuracy, the content is completely factually accurate with all information being correct and verifiable.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501


@evaluator(name="factual_accuracy")
async def factual_accuracy_evaluator(
    evaluated: Multimodal,
    /,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate factual correctness of content based on established knowledge.

    This evaluator assesses the factual accuracy of content by examining
    claims, data, statements, and assertions against well-established facts,
    scientific consensus, and generally accepted knowledge without requiring
    reference material.

    Parameters
    ----------
    evaluated : Multimodal
        The content to evaluate for factual accuracy
    guidelines : str | None, optional
        Additional guidelines for factual accuracy evaluation, by default None

    Returns
    -------
    EvaluationScore
        Factual accuracy score with categorical rating and explanation

    Raises
    ------
    ValueError
        When the evaluator fails to parse the result
    """
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
