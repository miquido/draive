# Basic RAG

RAG is a technique of adding additional information to the LLM context, to achieve better results.
This applies to adding a knowledge source to be used by LLM to generate correct response as well as
to introducing details about the task or personalization for more tailored result. The typical flow
would be to prepare a vector index and search through the data to extend the model input with
suitable context. We will use OpenAI services for this task, make sure to provide .env file with
`OPENAI_API_KEY` key before running.

````python
from draive import load_env

```python
from draive import DataModel, ctx, split_text
from draive.openai import OpenAI

# document is a short text about John Doe
document: str = """
...
"""

# define chunk data structure
class DocumentChunk(DataModel):
    full_document: str
    content: str

# prepare data chunks
document_chunks: list[DocumentChunk]
async with ctx.scope("basic_rag"):
    document_chunks = [
        DocumentChunk(
            full_document=document,
            content=chunk,
        )
        # split document text into smaller parts
        for chunk in split_text(
            text=document,
            separators=("\n\n", " "),
            part_size=64,
            part_overlap_size=16,
            count_size=len,
        )
    ]

print(f"Prepared {len(document_chunks)} chunks:\n---")
print("\n---\n".join(chunk.content for chunk in document_chunks))
````

```
Prepared 11 chunks:
---
John Doe, originally from a small farm in Texas, has been living in Vancouver for more than three years. His fascination with Canada began when he first visited the country at the age of seven. The experience left a lasting impression on him, and he knew that one day he would make Canada his home.
---
At the age of 18, John made the bold decision to leave his rural life in Texas behind and move to Vancouver. The transition was not without its challenges, as he had to adapt to the fast-paced city life, which was a stark contrast to the slow, quiet days on the farm. However,
---
contrast to the slow, quiet days on the farm. However, John's determination and love for his new home helped him overcome any obstacles he faced.
---
Now, at 21, John has fully embraced his life in Vancouver. He has made new friends, discovered his favorite local spots, and even started attending college to pursue his passion for environmental science. The city's stunning natural beauty, with its lush forests and pristine coastline, reminds him of why he fell in love
---
lush forests and pristine coastline, reminds him of why he fell in love with Canada in the first place.
---
John's days are filled with exploring the city's diverse neighborhoods, trying new cuisines, and participating in various outdoor activities. He has become an avid hiker, taking advantage of the numerous trails in and around Vancouver. On weekends, he often finds himself hiking in the nearby mountains, breathing in the crisp air and
---
himself hiking in the nearby mountains, breathing in the crisp air and marveling at the breathtaking views.
---
Despite the occasional homesickness for his family and the familiarity of his Texas farm, John knows that Vancouver is where he belongs. The city has captured his heart, and he can't imagine living anywhere else. He dreams of one day working in the field of environmental conservation, helping to protect the natural wonders that made him
---
field of environmental conservation, helping to protect the natural wonders that made him fall in love with Canada.
---
As John reflects on his journey from a small farm in Texas to the vibrant city of Vancouver, he feels a sense of pride and accomplishment. He knows that his seven-year-old self would be proud of the life he has built in the country that captured his imagination all those years ago. With a smile on his
---
```

````python
from collections.abc import Sequence

from draive import VectorIndex
from draive.helpers import VolatileVectorIndex
from draive.openai import OpenAIEmbeddingConfig, OpenAI

# prepare vector index
vector_index = VolatileVectorIndex()
async with ctx.scope(
    "indexing",
    # use vector index
    vector_index,
    # define embedding provider for this context
    OpenAIEmbeddingConfig(model="text-embedding-3-small"),
    disposables=(OpenAI(),),
):
    # add document chunks to the index
    await vector_index.index(
        DocumentChunk,
        values=document_chunks,
        # define what value will be embedded for each chunk
        attribute=DocumentChunk._.content,

```python
from collections.abc import Sequence

from draive import Toolbox, TextGeneration, tool, VectorIndex
from draive.openai import OpenAIResponsesConfig, OpenAI, OpenAIEmbeddingConfig

@tool(name="search") # prepare a simple tool for searching the index
async def index_search_tool(query: str) -> str:
    results: Sequence[DocumentChunk] = await ctx.state(VectorIndex).search(
        DocumentChunk,
        query=query,
        limit=3,
    )

    return "\n---\n".join(result.content for result in results)

async with ctx.scope(
    "searching",
    # use our vector index
    vector_index,
    # define used dependencies and services
    OpenAIResponsesConfig(model="gpt-4o-mini"),
    OpenAIEmbeddingConfig(model="text-embedding-3-small"),
    disposables=(OpenAI(),),
):
    # use the tool to augment generation by suitable document parts
    result: str = await TextGeneration.generate(
        instructions="Answer the questions based on provided data",
        input="Where is John Doe living?",
        # suggest the tool to ensure its usage
        tools=Toolbox.of(index_search_tool, suggest=index_search_tool),
    )
    print(result)
````

```
According to available sources, John Doe lives in Vancouver.
```
