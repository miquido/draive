# Skills

Draive supports [Agent Skills](https://agentskills.io/specification) as a file-based way to package reusable instructions and bundled resources for agents.

In Draive, skill support is centered around:

- `Skill.from_directory(...)` to load a skill directory from disk,
- `Skill` for validated metadata, instructions, and resource access,
- `Agent.from_skill(...)` to create a model-backed agent from a loaded skill.

## Skill Directory Format

A skill is loaded from a directory containing `SKILL.md` plus optional files.

```text
my-skill/
├── SKILL.md
├── scripts/
├── references/
└── assets/
```

`Skill.from_directory(...)` scans all regular files under the root directory (including `SKILL.md`) and registers them as `SkillResource` entries keyed by relative POSIX paths.

## SKILL.md Frontmatter

`SKILL.md` must begin with YAML frontmatter and then Markdown instructions:

```md
---
name: retrieval-assistant
description: Handles retrieval workflows and explains when to fetch references.
metadata:
  author: example-team
  version: "1.0"
---

# Retrieval Assistant

Use this skill when ...
```

Current Draive parsing behavior:

- `name` is required and must match `^[a-z0-9]+(?:-[a-z0-9]+)*$` with length `1..64`.
- `description` is required.
- `metadata` is optional and merged into `skill.meta`.
- Unknown top-level frontmatter fields raise an error.

This means optional Agent Skills spec fields like `license`, `compatibility`, and `allowed-tools` are currently not accepted by Draive's parser.

## Spec Compatibility Notes

Compared to the Agent Skills spec, Draive currently enforces a strict subset:

- Strictly required: `name`, `description`.
- Supported optional field: `metadata`.
- Not supported as top-level fields: `license`, `compatibility`, `allowed-tools`.
- `name` character constraints are enforced.
- `name` matching parent directory is not enforced.

If you need maximum Draive compatibility today, keep frontmatter limited to:

- `name`
- `description`
- `metadata`

## Loading Skills

```python
from pathlib import Path

from draive import Skill


skill: Skill = await Skill.from_directory(Path("skills/retrieval-assistant"))
print(skill.name)
print(skill.description)
```

`Skill.from_directory(...)` validates:

- the provided path is a directory,
- `SKILL.md` exists,
- frontmatter structure and required fields are valid,
- resource paths stay within the skill root.

## Accessing Bundled Resources

```python
from draive import SkillResourceMissing

try:
    reference = skill.resource("references/REFERENCE.md")
    text = reference.content.to_bytes().decode("utf-8", errors="replace")
except SkillResourceMissing:
    text = "Missing reference"
```

Resource lookup uses normalized relative POSIX paths and rejects invalid paths (absolute paths, `~`, or `..` traversal).

## Creating Agents From Skills

```python
from draive import Agent, MultimodalContentPart, ProcessingEvent, ctx
from collections.abc import AsyncIterable
from draive.openai import OpenAI, OpenAIResponsesConfig


assistant: Agent = Agent.from_skill(skill)

async with ctx.scope(
    "skills.agent",
    OpenAIResponsesConfig(model="gpt-5-mini"),
    disposables=(OpenAI(),),
):
    stream: AsyncIterable[MultimodalContentPart | ProcessingEvent] = assistant.call(
        input="Use local references to answer this question."
    )
    async for chunk in stream:
        print(chunk)
```

`Agent.from_skill(...)` automatically adds a `read_resource(path)` tool that lets the model read bundled files by relative path during execution.
