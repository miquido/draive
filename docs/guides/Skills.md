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

`Skill.from_directory(...)` scans all regular files under the root directory (including `SKILL.md`) and registers them as `SkillResource` entries keyed by relative POSIX paths. Symbolic links are skipped, so a link pointing outside the skill root can't be loaded as a resource.

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

Loading reads files through the scoped filesystem access, so it has to run within a context scope.

```python
from pathlib import Path

from draive import Skill, ctx


async with ctx.scope("skills.load"):
    skill: Skill = await Skill.from_directory(Path("skills/retrieval-assistant"))

print(skill.name)
print(skill.description)
print(skill.meta["skill_source"])  # path the skill was loaded from
```

`Skill.from_directory(...)` validates:

- the provided path is a directory,
- `SKILL.md` exists,
- frontmatter structure and required fields are valid,
- resource paths stay within the skill root.

Besides the frontmatter `metadata` entries, `skill.meta` carries `skill_source` - the directory the
skill was loaded from.

## Accessing Bundled Resources

```python
from draive import SkillResourceMissing

try:
    reference = skill.resource("references/REFERENCE.md")
    text = reference.content.to_bytes().decode("utf-8", errors="replace")
except SkillResourceMissing:
    text = "Missing reference"
```

Resource lookup uses normalized relative POSIX paths and rejects invalid paths (absolute paths, `~`, or `..` traversal). Use `skill.has_resource(path)` to check availability without handling the exception.

## Creating Agents From Skills

```python
from draive import Agent, MultimodalContentPart, ProcessingEvent, ctx
from collections.abc import AsyncGenerator
from draive.openai import OpenAI, OpenAIResponsesConfig


assistant: Agent = Agent.from_skill(skill)

async with ctx.scope(
    "skills.agent",
    OpenAIResponsesConfig(model="gpt-5.5"),
    disposables=(OpenAI(),),
):
    stream: AsyncGenerator[MultimodalContentPart | ProcessingEvent] = assistant.call(
        input="Use local references to answer this question."
    )
    async for chunk in stream:
        print(chunk)
```

`Agent.from_skill(...)` uses the skill instructions as agent instructions, derives the agent
identity from the skill `name`, `description` and `meta`, and automatically adds a
`read_resource(path)` tool that lets the model read bundled files by relative path during execution.

It also accepts the remaining `Agent.generative(...)` configuration:

- `tools=` with additional tools, merged with the generated resources tool,
- `memory=` to persist context across turns (see
    [Agents](./Agents.md#persist-context-across-turns-with-agentmemory)),
- `output=` to select the model output mode,
- `identity=` to override the derived `AgentIdentity`, or `meta=` to extend the skill metadata.

The resources tool can also be obtained on its own with `skill.resources_tool()` - useful when
building the agent by hand or exposing skill files to an existing toolbox.
