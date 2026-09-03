from collections.abc import Iterable, Sequence
from typing import Any

from haiway import State

from draive.generation import ModelGeneration

__all__ = ("generate_case_parameters",)

INSTRUCTION: str = """\
You are preparing automated evaluation test scenarios.

<INSTRUCTION>
Familiarize yourself with the SCHEMA describing desired data structure.
Prepare diversified and high quality test scenarios according to the provided SCHEMA.
Results should cover various possible valid outcomes to cover multiple possible scenarios.
</INSTRUCTION>
{guidelines}
<SCHEMA>
{{model_schema}}
</SCHEMA>

<FORMAT>
Provide the result using a single raw valid JSON object that adheres strictly to the given \
SCHEMA without any comments, formatting, or additional elements.
</FORMAT>
"""

INPUT: str = "Prepare additional scenario"


async def generate_case_parameters[Parameters: State](
    parameters: type[Parameters],
    /,
    *,
    count: int,
    examples: Iterable[Parameters],
    guidelines: str | None = None,
) -> Sequence[Parameters]:
    results: list[Parameters] = []
    example_pairs: list[tuple[str, Any]] = [(INPUT, example) for example in examples]
    # keep curly braces of the guidelines escaped, schema injection formats the result again
    instructions: str = INSTRUCTION.format(
        guidelines=f"\n<GUIDELINES>\n{_escape_curly_braces(guidelines)}\n</GUIDELINES>\n"
        if guidelines
        else ""
    )

    for _ in range(0, count):
        results.append(
            await ModelGeneration.generate(
                parameters,
                instructions=instructions,
                input=INPUT,
                examples=example_pairs,
                schema_injection="full",
            )
        )
        # put each generated element into the examples for next generation
        # to prevent repetitions and encourage diversified results
        example_pairs.append((INPUT, results[-1]))

    return results


def _escape_curly_braces(value: str) -> str:
    return value.replace("{", "{{").replace("}", "}}")
