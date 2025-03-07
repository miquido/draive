# Basic RAG

RAG is a technique of adding additional information to the LLM context to achieve better results. This applies to adding a knowledge source to be used by LLM to generate correct response as well as to introducing details about the task or personalization for more tailored result. The typical flow would be to prepare a vector index and search through the data to extend the model input with suitable context. We will use OpenAI services for this task, make sure to provide .env file with `OPENAI_API_KEY` key before running. 

```python
from draive import load_env

load_env() # load .env variables
```

## Data preparation

We are going to use a local file with plain text as out input. After loading the file content we will prepare it for embedding by splitting the original text to smaller chunks. This allows to fit the data into limited context windows and to additionally filter out unnecessary information. We are going to use a basic splitter looking for paragraphs structure defined by multiple subsequent newlines in the text. Chunks will be put into a structured envelope which allows to store additional information such as a full text of the document or additional metadata.

```python
from draive import DataModel, count_text_tokens, ctx, split_text
from draive.openai import OpenAI

# define chunk data structure
class DocumentChunk(DataModel):
    full_document: str
    content: str

# prepare data chunks
document_chunks: list[DocumentChunk]
async with ctx.scope(
    "basic_rag",
    await OpenAI().tokenizer("gpt-4o-mini")
):
    # load the document contents
    with open("data/personal_document.txt") as file:
        document = file.read()

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
            count_size=count_text_tokens,
        )
    ]

print(f"Prepared {len(document_chunks)} chunks:\n---")
print("\n---\n".join(chunk.content for chunk in document_chunks))
```

## Data indexing

When we have prepared the data, we can now use a vector index to make it searchable. We are going to use in-memory, volatile vector index which can be useful for a quick search. Preparing the index requires defining the text embedding method. In this example we are going to use OpenAI embedding solution. After defining all required parameters and providers we can prepare the index using our data. In order to ensure proper data embedding it is required to specify what value will be used to prepare the vector. In this case we specify the chunk content to be used.

```python
from collections.abc import Sequence

from draive import VectorIndex
from draive.openai import OpenAIEmbeddingConfig, OpenAI

# prepare vector index
vector_index: VectorIndex = VectorIndex.volatile()
async with ctx.scope(
    "indexing",
    # define embedding provider for this context
    OpenAI().text_embedding(),
    OpenAIEmbeddingConfig(model="text-embedding-3-small"),
    # use vector index
    vector_index,
):
    # add document chunks to the index
    await vector_index.index(
        DocumentChunk,
        values=document_chunks,
        # define what value will be embedded for each chunk
        indexed_value=DocumentChunk._.content,
    )
```

## Searching the index

Now we have everything set up to do our search. Instead of searching directly, we are going to leverage the ability of LLMs to call tools. We have to define a search tool for that task, which will use our index to deliver the results. LLM will make use of this tool to look for the answer based on the search result.

```python
from draive import Toolbox, generate_text, tool
from draive.openai import OpenAIChatConfig, OpenAI


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
    # define used dependencies and services
    OpenAI().lmm_invoking(),
    OpenAIChatConfig(model="gpt-4o-mini"),
    OpenAI().text_embedding(),
    OpenAIEmbeddingConfig(model="text-embedding-3-small"),
    # use our vector index
    vector_index,
):
    # use the tool to augment LLM generation by suitable document parts
    result: str = await generate_text(
        instruction="Answer the questions based on provided data",
        input="Where is John Doe living?",
        # suggest the tool to ensure its usage
        tools=Toolbox.of(suggest=index_search_tool),
    )
    print(result)
```