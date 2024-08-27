from collections.abc import Iterable

from draive.generation import generate_model
from draive.parameters import DataModel

__all__ = [
    "generate_case_parameters",
]

INSTRUCTION: str = """\
Assistant is a test data generation bot.

<INSTRUCTION>
Familiarize yourself with the SCHEMA describing desired data structure.
Generate diversified and high quality data according to the provided SCHEMA.
Generated data should cover various possible valid outcomes.
</INSTRUCTION>

<SCHEMA>
{schema}
</SCHEMA>

<FORMAT>
Provide the result using a single raw valid JSON object that adheres strictly to the given \
SCHEMA without any comments, formatting, or additional elements.
</FORMAT>
"""

INPUT: str = "Generate next"


async def generate_case_parameters[Parameters: DataModel](
    parameters: type[Parameters],
    /,
    *,
    count: int,
    examples: Iterable[Parameters],
) -> list[Parameters]:
    results: list[Parameters] = []
    example_pairs: list[tuple[str, parameters]] = [(INPUT, example) for example in examples]

    for _ in range(0, count):
        results.append(
            await generate_model(
                parameters,
                instruction=INSTRUCTION,
                input=INPUT,
                examples=example_pairs,
                schema_injection="full",
            )
        )
        # put each generated element into the examples for next generation
        # to prevent repetitions and encourage diversified results
        example_pairs.append((INPUT, results[-1]))

    return results
