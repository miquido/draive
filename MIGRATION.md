**Draive Migration (0.83 → 0.84)**

**TL;DR Quick Path**
- Replace `LMM` with `GenerativeModel` (and `LMM*` types with `Model*`).
- Replace `Media*` with `ResourceContent` / `ResourceReference` and build content via `MultimodalContent.of(...)`.
- Replace `MetaContent` with `ArtifactContent` for custom/hidden payloads.
- Use `instructions` (plural) instead of `instruction`/`prompt` (prompts removed).
- Use typed feature selection (e.g., `GenerativeModel`) instead of string-based (e.g., `"lmm"`).
- Migrate conversation streaming types to `ConversationInputChunk`/`ConversationOutputChunk`.
- Replace `Tools` with `Toolbox`; `ToolHandling` replaced by typed model/tool declarations.
- Catch `ModelRateLimit` instead of `RateLimitError`.

**LMM → GenerativeModel**
Switch from `LMM` to `GenerativeModel` and new `Model*` types.

Before:
```python
from draive.lmm import LMM, LMMInput

await LMM.completion(context=[LMMInput.of("Hello")])
```

After:
```python
from draive import GenerativeModel, ModelInput

await GenerativeModel.completion(context=[ModelInput.of("Hello")])
```

Mappings (most used):
- `LMM` → `GenerativeModel`
- `RealtimeLMM` → `RealtimeGenerativeModel`
- `LMMInput` → `ModelInput`
- `LMMContext` / `LMMContextElement` → `ModelContext` / `ModelContextElement`
- `LMMSession*` → `ModelSession*` (e.g., `LMMSessionScope` → `ModelSessionScope`)
- `LMMStream*` → `ModelInputChunk`/`ModelStreamOutput`/`ModelOutputChunk` (depending on direction)
- `LMMTool*` → `ModelTool*` and `Tool*` (see Tools section)
- `LMMException` → `ModelException`
- `LMMOutputInvalid`/`LMMOutputLimit` → `ModelOutputInvalid`/`ModelOutputLimit`

**Multimodal & Media**
Replace legacy `Media*` types with resources and updated multimodal parts.

Before:
```python
from draive import MultimodalContent, MediaData, MediaReference

content = MultimodalContent.of(
    "hello",
    MediaReference.of("https://example.com/image.png", media="image/png"),
    MediaData.of(b"...bytes...", media="application/octet-stream"),
)
```

After:
```python
from draive import MultimodalContent, ResourceContent, ResourceReference, MimeType

content = MultimodalContent.of(
    "hello",
    ResourceReference.of("https://example.com/image.png", mime_type=MimeType("image/png")),
    ResourceContent.of(b"...bytes...", mime_type="application/octet-stream"),
)
```

Additional renames:
- `MediaKind`/`MediaType` → `MimeType`
- `MultimodalContentElement` → `MultimodalContentPart`
- `MultimodalTagElement` → `MultimodalTag`

**MultimodalTag and tag parsing/extraction**
`MultimodalTagElement` is replaced by `MultimodalTag` with a stricter parser and new helpers for extraction and replacement.

Before:
```python
# Often tags were embedded as raw strings in text
from draive import MultimodalContent

content = MultimodalContent.of(
    "Intro ",
    "<code lang=\"python\">print('hi')</code>",
    " End",
)
```

After:
```python
from draive import MultimodalContent, MultimodalTag

tag = MultimodalTag.of("print('hi')", name="code", meta={"lang": "python"})
content = MultimodalContent.of("Intro ", tag, " End")

# String roundtrip
assert tag.to_str() == '<code lang="python">print(\'hi\')</code>'
assert tag.to_multimodal().to_str() == tag.to_str()
```

New helpers on `MultimodalContent`:
- `content.tag(name)` returns the first `MultimodalTag` with `name` or `None`.
- `content.tags(name)` returns all `MultimodalTag` with `name` as a tuple.
- `content.replacing_tag(name, replacement, strip_tags=False, exhaustive=False)` replaces occurrences while preserving multimodal parts inside the tag body.

Example:
```python
from draive import MultimodalContent, MultimodalTag

content = MultimodalContent.of(
    "A ",
    MultimodalTag.of("x=1", name="var", meta={"name": "x"}),
    " and ",
    MultimodalTag.of("y=2", name="var", meta={"name": "y"}),
)

first = content.tag("var")              # MultimodalTag | None
all_vars = content.tags("var")          # tuple[MultimodalTag, ...]

replaced = content.replacing_tag(
    "var",
    MultimodalTag.of("z=0", name="var", meta={"name": "z"}),
    exhaustive=True,
)
```

Parser rules:
- Tags are detected only within `TextContent`, but captured bodies may span text, resources, and artifacts; extraction preserves original parts.
- Self-closing tags like `<tag .../>` are supported and yield empty `content` for that tag.
- Attributes must be quoted (`key="value"`); unquoted values are rejected. Whitespace around `=` is tolerated.
- Supported escapes in attribute values: `\"`, `\\`, `\n`, `\t`, `\r`. Literal newlines in attribute values are not allowed.
- Nested tags are supported; matching is performed at the correct nesting depth. Replacement can keep or strip tag markers via `strip_tags`.
- Attributes surface as `MultimodalTag.meta` (a `Meta` mapping). Use strings or simple lists as values; nested mappings are not allowed.

**ArtifactContent**
Wrap artifacts and replace `MetaContent` with `ArtifactContent`.

Before:
```python
from draive import MultimodalContent, MetaContent

MultimodalContent.of(CustomArtifact(), MetaContent.of(...))
```

After:
```python
from draive import ArtifactContent, MultimodalContent

MultimodalContent.of(
    ArtifactContent.of(CustomArtifact(...)),
    ArtifactContent.of(CustomMeta(...)),
)
```

**Typed features**
Use typed feature selection within providers.

Before:
```python
from draive import ctx
from draive.openai import OpenAI

async with ctx("example", disposables=(OpenAI(features=("lmm",),))):
    ...
```

After:
```python
from draive import ctx, GenerativeModel
from draive.openai import OpenAI

async with ctx("example", disposables=(OpenAI(features=(GenerativeModel, ),))):
    ...
```

You can request multiple features using types, e.g. `features=(GenerativeModel, ImageGeneration)`. This avoids stringly-typed flags and improves static checks.

**Instructions (prompts removed)**
Use `instructions` (plural) and the new typed instructions repository. The old `instruction`/`Prompt*` APIs are removed.

Before:
```python
await GenerativeModel.completion(instruction="Summarize this", context=[...])
```

After:

```python
await GenerativeModel.completion(instructions=["Summarize this"], context=[...])
```

Common migrations:
- `Prompt*` → `Instructions*`
- `prompt` decorator → `instructions` decorator
- `Prompts` → `InstructionsRepository`

**Conversations**
Streaming types are simplified and renamed.

Before:
```python
from draive import Conversation, ConversationMessageChunk, ConversationStreamElement
```

After:
```python
from draive import Conversation, ConversationInputChunk, ConversationOutputChunk
```

`ConversationElement` → `ConversationInputChunk` (input side)
`ConversationMessageChunk` / `ConversationStreamElement` → `ConversationOutputChunk` (output side)
`ConversationMemory` is removed; use `ModelMemory`/`Memory` utilities where needed.

**Tools**
The `Tools` facade is replaced with `Toolbox`, and handling is typed with model/tool declarations.

Before:
```python
from draive.tools import Tools, tool

@tool
async def sum_tool(a: int, b: int) -> int: ...

tools = Tools.of(sum_tool)
```

After:
```python
from draive import Toolbox, tool

@tool
def sum_tool(a: int, b: int) -> int: ...

tools = Toolbox.of(sum_tool)
```

**Resources**
Resources are split into typed state/contracts and templates.

Key changes and aliases:
- `ResourceException` → `ResourceCorrupted` 
- `ResourceDeclaration`/`ResourceTemplateDeclaration` → `ResourceReferenceTemplate` (alias available as `draive.resources.ResourceTemplateDeclaration`)
- Top-level exports concentrate on `Resource`, `ResourceContent`, `ResourceReference`, `ResourceTemplate`, `ResourcesRepository`, `resource`

Usage example stays similar for templates:
```python
from draive import ResourceTemplate, resource

@resource("images/avatar")
def avatar() -> ResourceTemplate: ...
```

**Exceptions & rate limits**
Use `ModelRateLimit` instead of provider-specific or old `RateLimitError`.

Before:
```python
from draive import RateLimitError
```

After:
```python
from draive import ModelRateLimit
```

**Removed modules**
- `draive.agents` — removed. Compose workflows with your own orchestration or `draive.stages` as building blocks.
- `draive.choice` — removed. Use `ModelOutputSelection` or handle selection in application code.
- `draive.prompts` — removed. Use `instructions` and `InstructionsRepository`.
- `draive.tokenization` — removed. Use provider-native token accounting or model-specific helpers.
- `split_sequence` — removed. Prefer `split_text` or local utilities.

Memory helpers changed:
- `MEMORY_NONE`, `volatile`, `accumulative_volatile` removed from `draive.utils.memory`.
- Use `Memory.constant(value)` for a fixed recall/no-op remember, or implement a small `State` for custom memory behavior. For per-model memory, prefer `ModelMemory` from `draive.models`.

**Minor additions**
- `AudioGeneration` added for audio outputs.
