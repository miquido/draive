# Basic data extraction

One of the most useful abilities of LLMs is to extract the information from an unstructured source and put it into a desired structure. The typical flow would be to load some kind of a document and instruct the model to provide a json with a list of required fields. Draive comes with a dedicated solution for generating structured output which we can use to extract required information from any source. We will use OpenAI for this task, make sure to provide .env file with `OPENAI_API_KEY` key before running. 


```python
from draive import load_env

load_env() # load .env variables
```

## Data source

For our data source we will use a local file with plain text. For more advanced use and more data formats you might need to perform additional steps like the text extraction.


```python
# load document contents for the extraction
document: str
with open("data/personal_document.txt") as file:
    document = file.read()
```

## Data structure

`DataModel` base class is a perfect tool to define required data structure. We will use it to define a simple structure for this example. Please see [AdvancedState](../guides/AdvancedState.md) for more details about the advanced data model customization.


```python
from draive import DataModel

class PersonalData(DataModel):
    first_name: str
    last_name: str
    age: int | None = None
    country: str | None = None
```

## Extraction

Since we are going to generate a structured output we will use the `generate_model` function which automatically decodes the result of LLM call into a python object. We have to prepare the execution context, provide an instruction, source document and specify the data model and we can expect extracted data to be available. Keep in mind that depending on the task it might be beneficial to prepare a specific instruction and/or specify more details about the model itself.


```python
from draive import ctx, generate_model
from draive.openai import OpenAIChatConfig, OpenAI

async with ctx.scope(
    "data_extraction",
    # define used LMM to be OpenAI within the context
    OpenAI().lmm_invoking(),
    OpenAIChatConfig(model="gpt-4o-mini")
):
    result: PersonalData = await generate_model(
        # define model to generate
        PersonalData,
        # provide additional instructions
        # note that the data structure will be automatically explained to LLM
        instruction="Please extract information from the given input",
        # we will provide the document as an input
        input=document,
    )

    print(result)
```
    first_name: John
    last_name: Doe
    age: 21
    country: Canada


## Customization

We can customize data extraction by specifying more details about the model itself i.e. description of some fields. We can also choose to take more control over the prompt used to generate the model. Instead of relying on the automatically generated model description we can specify it manually or provide a placeholder inside the instruction to put it in specific place. Additionally we can choose between full json schema description and the simplified description which works better with smaller, less capable models.


```python
from draive import ctx, generate_model

async with ctx.scope(
    "customized_extraction",
    # define used LMM to be OpenAI within the context
    OpenAI().lmm_invoking(),
    OpenAIChatConfig(model="gpt-4o-mini")
):
    result: PersonalData = await generate_model(
        PersonalData,
        instruction=(
            # provide extended instructions and take full control over the prompt
            "Please extract information from the given input."
            " Put it into the JSON according to the following description:\n"
            "{schema}"  # 'schema' is the name of the format argument used to fill in the schema
        ),
        input=document,
        # we will use the simplified schema description
        # you can also skip adding schema at all
        schema_injection="simplified",
    )

    print(f"JSON schema:\n{PersonalData.json_schema(indent=2)}")
    print(f"Simplified schema:\n{PersonalData.simplified_schema(indent=2)}")
    print(f"Result:\n{result}")
```

    JSON schema:
    {
      "type": "object",
      "properties": {
        "first_name": {
          "type": "string"
        },
        "last_name": {
          "type": "string"
        },
        "age": {
          "oneOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ]
        },
        "country": {
          "oneOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ]
        }
      },
      "required": [
        "first_name",
        "last_name"
      ]
    }
    Simplified schema:
    {
      "first_name": "string",
      "last_name": "string",
      "age": "integer|null",
      "country": "string|null"
    }
    Result:
    first_name: John
    last_name: Doe
    age: 21
    country: Canada

