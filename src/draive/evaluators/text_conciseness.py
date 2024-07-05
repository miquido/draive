from draive.evaluation import evaluator
from draive.generation import generate_model
from draive.parameters import DataModel
from draive.types.multimodal import MultimodalContent, TextContent

__all__ = [
    "ConcisenessScore",
    "text_conciseness_evaluator",
]


class ConcisenessScore(DataModel):
    score: float
    comment: str | None = None


INSTRUCTION: str = """\
You will be given a reference text and a derived text based on the reference text.
Your task is to rate the derived text based on one metric.
Please make sure you read and understand these instructions very carefully.
Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Conciseness (1-5) - the extent to which the derived text is brief and to the point while still covering all key information.
A concise derived text avoids unnecessary details and repetition.
Annotators should penalize derived texts that are overly verbose or include irrelevant information.

Evaluation Steps:
1. Read the derived text and the reference text carefully.
2. Compare the derived text to the reference text and identify the main points of the reference text.
3. Assess how well the derived text covers the main points of the reference text, and how much irrelevant or redundant information it contains.
4. Assign a conciseness score from 1 to 5.
"""  # noqa: E501

@evaluator(name="Conciseness Evaluator", threshold=0.75)
async def text_conciseness_evaluator(
    reference: str,
    text: str,
    /,
) -> float:
    input_content = MultimodalContent.of(
        TextContent(text=f"Reference text: {reference}\n\nDerived text: {text}") # type: ignore
    )
    model: ConcisenessScore = await generate_model(
        ConcisenessScore,
        instruction=INSTRUCTION,
        input=input_content,
    )

    return model.score/5
