# Basic Data Extraction

Use `ModelGeneration` to extract typed structured output from unstructured text.

This pattern is useful when you need stable field-level outputs instead of free-form text.

## Setup

```python
from draive import load_env

load_env()
```

## Define Target Schema

Define the structure you want the model to return.

```python
from draive import State


class PersonalData(State, serializable=True):
    first_name: str
    last_name: str
    age: int | None = None
    country: str | None = None
```

## Run Extraction

```python
from draive import ModelGeneration, ctx
from draive.openai import OpenAI, OpenAIResponsesConfig


document = "John Doe is 21 and lives in Vancouver, Canada."

async with ctx.scope(
    "data_extraction",
    OpenAIResponsesConfig(model="gpt-5-mini"),
    disposables=(OpenAI(),),
):
    result: PersonalData = await ModelGeneration.generate(
        PersonalData,
        instructions="Extract personal information from the input.",
        input=document,
    )

    print(result)
```

## Customize Schema Injection

Schema injection controls how much schema guidance is injected into instructions.

```python
result: PersonalData = await ModelGeneration.generate(
    PersonalData,
    instructions="Extract fields from the input. Return JSON matching the schema:\n{%schema%}",
    input=document,
    schema_injection="simplified",
)
```

`schema_injection` values:

- `"full"` inject full JSON schema
- `"simplified"` inject simplified schema
- `"skip"` keep instructions unchanged
