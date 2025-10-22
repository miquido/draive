from draive.evaluation import EvaluationScore, evaluator
from draive.evaluators.utils import FORMAT_INSTRUCTION, extract_evaluation_result
from draive.multimodal import Multimodal, MultimodalContent
from draive.stages import Stage

__all__ = ("creativity_evaluator",)


INSTRUCTION: str = f"""\
You are evaluating the provided content according to the defined criteria.

<INSTRUCTION>
Carefully examine provided CONTENT, then rate it using solely a creativity metric according to the EVALUATION_CRITERIA.
Think step by step and provide explanation of the score before the final score.
Use the explained RATING scale and the requested FORMAT to provide the result.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Evaluated metric is creativity - the degree of originality, novelty, and innovative thinking demonstrated in the content. Creative content should show original ideas, unique perspectives, imaginative approaches, novel combinations of concepts, or innovative solutions. This includes artistic creativity, problem-solving creativity, conceptual originality, and the ability to think outside conventional patterns. Consider uniqueness of ideas, originality of expression, innovative approaches, and imaginative elements.
</EVALUATION_CRITERIA>
{{guidelines}}
<RATING>
Assign a creativity score using exact name of one of the following values:
- "poor" is very low creativity, the content is entirely conventional, generic, or lacks any original or novel elements.
- "fair" is low creativity, the content shows minimal originality with mostly conventional ideas and approaches.
- "good" is moderate creativity, the content demonstrates some original thinking and creative elements mixed with conventional approaches.
- "excellent" is high creativity, the content shows significant originality, innovative thinking, and creative approaches with minor conventional elements.
- "perfect" is very high creativity, the content is highly original, innovative, and demonstrates exceptional creative thinking throughout.
Use the "none" value for content that cannot be rated at all.
</RATING>

{FORMAT_INSTRUCTION}
"""  # noqa: E501


@evaluator(name="creativity")
async def creativity_evaluator(
    evaluated: Multimodal,
    /,
    guidelines: str | None = None,
) -> EvaluationScore:
    """
    Evaluate the originality and creative thinking in content.

    This evaluator assesses the degree of creativity, originality, and
    innovative thinking demonstrated in content, including novel ideas,
    unique perspectives, and imaginative approaches.

    Parameters
    ----------
    evaluated : Multimodal
        The content to evaluate for creativity
    guidelines : str | None, optional
        Additional guidelines for creativity evaluation, by default None

    Returns
    -------
    EvaluationScore
        Creativity score with categorical rating and explanation

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
