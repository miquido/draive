## Generate structured data with OpenAI

Add OPENAI_API_KEY key to the .env file to allow access to OpenAI services.


```python
from draive import load_env

load_env()  # loads OPENAI_API_KEY from .env file
```


```python
from draive import DataModel


# define a Model class to describe the data
class InterestingPlace(DataModel):
    name: str
    description: str | None = None
```


```python
from draive import ctx, generate_model
from draive.openai import OpenAIChatConfig, OpenAI

# initialize dependencies and configuration
async with ctx.scope(
    "basics",
    OpenAI().lmm_invoking(),  # define used LMM use OpenAI
    OpenAIChatConfig(model="gpt-4o-mini"),  # configure OpenAI model
):
    # request model generation
    generated: InterestingPlace = await generate_model(
        # define model to generate
        InterestingPlace,
        # provide a prompt instruction
        instruction="You are a helpful assistant.",
        # add user input
        input="What is the most interesting place to visit in London?",
    )
    print(generated)
```

    name: The British Museum
    description: A world-famous museum dedicated to human history, art, and culture, home to millions of works from ancient civilizations.

