from draive.evaluation import evaluator

__all__ = [
    "text_keyword_evaluator",
]

@evaluator(name="Keyword Evaluator", threshold=0.95)
async def text_keyword_evaluator(text: str, keywords: list[str]) -> float:
        return len([word for word in keywords if word.lower() in text.lower()]) / len(keywords)
