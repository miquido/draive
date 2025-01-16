from collections.abc import Iterable
from typing import Any

from draive.generation import generate_model
from draive.parameters import DataModel

__all__ = [
    "generate_case_parameters",
]

INSTRUCTION: str = """\
You are preparing automated evaluation test scenarios.

<INSTRUCTION>
Familiarize yourself with the SCHEMA describing desired data structure.
Prepare diversified and high quality test scenarios according to the provided SCHEMA.
Result should cover various possible valid outcomes to cover multiple possible scenarios.
</INSTRUCTION>
{guidelines}
<SCHEMA>
{schema}
</SCHEMA>

<FORMAT>
Provide the result using a single raw valid JSON object that adheres strictly to the given \
SCHEMA without any comments, formatting, or additional elements.
</FORMAT>
"""

INPUT: str = "Prepare additional scenario"


async def generate_case_parameters[Parameters: DataModel](
    parameters: type[Parameters],
    /,
    *,
    count: int,
    examples: Iterable[Parameters],
    guidelines: str | None = None,
) -> list[Parameters]:
    results: list[Parameters] = []
    example_pairs: list[tuple[str, Any]] = [(INPUT, example) for example in examples]

    for _ in range(0, count):
        results.append(
            await generate_model(
                parameters,
                instruction=INSTRUCTION.format(
                    guidelines=f"\n<GUIDELINES>\n{guidelines}\n</GUIDELINES>\n"
                    if guidelines is not None
                    else ""
                ),
                input=INPUT,
                examples=example_pairs,
                schema_injection="full",
            )
        )
        # put each generated element into the examples for next generation
        # to prevent repetitions and encourage diversified results
        example_pairs.append((INPUT, results[-1]))

    return results
