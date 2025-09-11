# Basic data extraction

One of the most useful abilities of LLMs is to extract information from an unstructured source and put it into a desired structure. The typical flow would be to load some kind of document and instruct the model to provide a JSON with a list of required fields. Draive comes with a dedicated solution for generating structured output which we can use to extract required information from any source. We will use OpenAI for this task, make sure to provide .env file with `OPENAI_API_KEY` key before running.


```python
from draive import load_env




```python
# load document contents for the extraction
document: str = """
John Doe, originally from a small farm in Texas, has been living in Vancouver for more than three years. His fascination with Canada began when he first visited the country at the age of seven. The experience left a lasting impression on him, and he knew that one day he would make Canada his home.

At the age of 18, John made the bold decision to leave his rural life in Texas behind and move to Vancouver. The transition was not without its challenges, as he had to adapt to the fast-paced city life, which was a stark contrast to the slow, quiet days on the farm. However, John's determination and love for his new home helped him overcome any obstacles he faced.

Now, at 21, John has fully embraced his life in Vancouver. He has made new friends, discovered his favorite local spots, and even started attending college to pursue his passion for environmental science. The city's stunning natural beauty, with its lush forests and pristine coastline, reminds him of why he fell in love with Canada in the first place.

John's days are filled with exploring the city's diverse neighborhoods, trying new cuisines, and participating in various outdoor activities. He has become an avid hiker, taking advantage of the numerous trails in and around Vancouver. On weekends, he often finds himself hiking in the nearby mountains, breathing in the crisp air and marveling at the breathtaking views.

Despite the occasional homesickness for his family and the familiarity of his Texas farm, John knows that Vancouver is where he belongs. The city has captured his heart, and he can't imagine living anywhere else. He dreams of one day working in the field of environmental conservation, helping to protect the natural wonders that made him fall in love with Canada.

As John reflects on his journey from a small farm in Texas to the vibrant city of Vancouver, he feels a sense of pride and accomplishment. He knows that his seven-year-old self would be proud of the life he has built in the country that captured his imagination all those years ago. With a smile on his face, John looks forward to the future and all the adventures that Vancouver has in store for him.



```python
from draive import DataModel

class PersonalData(DataModel):
    first_name: str
    last_name: str
    age: int | None = None



```python
from draive import ctx, ModelGeneration
from draive.openai import OpenAIResponsesConfig, OpenAI

async with ctx.scope(
    "data_extraction",
    # initialize OpenAI client within the context
    OpenAIResponsesConfig(model="gpt-4o-mini"),
    disposables=(OpenAI(),),
):
    result: PersonalData = await ModelGeneration.generate(
        # define model to generate
        PersonalData,
        # provide additional instructions
        # note that the data structure will be automatically explained to the model
        instructions="Extract information from the given input",
        # we will provide the document as an input
        input=document,
    )

    print(result)
```
    first_name: John
    last_name: Doe
    age: 21



```python
from draive import ctx, ModelGeneration

async with ctx.scope(
    "customized_extraction",
    # initialize OpenAI client within the context
    OpenAIResponsesConfig(model="gpt-4o-mini"),
    disposables=(OpenAI(),),
):
    result: PersonalData = await ModelGeneration.generate(
        PersonalData,
        instructions=(
            # provide extended instructions and take full control over the prompt
            "Extract information from the given input."
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
