from draive.evaluation import evaluator

__all__ = [
    "text_keywords_evaluator",
]


@evaluator(name="text_keywords")
async def text_keywords_evaluator(
    text: str,
    /,
    keywords: list[str],
) -> float:
    return len([word for word in keywords if word.lower() in text.lower()]) / len(keywords)
