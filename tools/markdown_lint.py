from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CODE_FENCE_PATTERN = re.compile(r"^(```+)(.*)$")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+.*$")


@dataclass
class LintError:
    path: Path
    line: int
    message: str

    def __str__(self) -> str:  # pragma: no cover - human readable output
        relative = self.path.as_posix()
        return f"{relative}:{self.line}: {self.message}"


def iter_markdown_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*.md") if p.is_file())
        elif path.suffix == ".md" and path.is_file():
            yield path


def lint_markdown(path: Path) -> list[LintError]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    errors: list[LintError] = []

    if not text.endswith("\n"):
        errors.append(LintError(path, len(lines) or 1, "file must end with a newline"))

    # detect consecutive blank lines greater than one outside code fences
    blank_run = 0
    in_fence = False
    fence_stack: list[str] = []
    current_heading_level = 0
    first_content_line: str | None = None

    for index, line in enumerate(lines, start=1):
        stripped = line.strip()

        fence_match = CODE_FENCE_PATTERN.match(line)
        if fence_match:
            fence_delimiter = fence_match.group(1)
            if in_fence and fence_stack and fence_stack[-1] == fence_delimiter:
                fence_stack.pop()
                in_fence = bool(fence_stack)
            else:
                fence_stack.append(fence_delimiter)
                in_fence = True
            blank_run = 0
        elif not in_fence:
            if stripped == "":
                blank_run += 1
                if blank_run > 1:
                    errors.append(LintError(path, index, "more than one consecutive blank line"))
            else:
                if first_content_line is None:
                    first_content_line = stripped
                blank_run = 0
                heading_match = HEADING_PATTERN.match(line)
                if heading_match:
                    hash_marks = heading_match.group(1)
                    level = len(hash_marks)
                    if current_heading_level and level > current_heading_level + 1:
                        errors.append(
                            LintError(
                                path,
                                index,
                                f"heading level jumps from {current_heading_level} to {level}",
                            )
                        )
                    current_heading_level = level
                elif line.startswith("#"):
                    errors.append(LintError(path, index, "heading markers must be followed by a space"))
        else:
            blank_run = 0

        if line.rstrip() != line:
            errors.append(LintError(path, index, "trailing whitespace"))
        if "\t" in line:
            errors.append(LintError(path, index, "tab character found"))

    if fence_stack:
        errors.append(LintError(path, len(lines), "code fence not closed"))

    if first_content_line is None or not first_content_line.startswith("# "):
        errors.append(LintError(path, 1, "first non-blank line must be an H1 heading"))

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint Markdown files for Draive docs.")
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to lint",
    )
    args = parser.parse_args()

    files = list(iter_markdown_files(args.paths))
    if not files:
        print("No markdown files found.", file=sys.stderr)
        return 1

    all_errors: list[LintError] = []
    for path in files:
        all_errors.extend(lint_markdown(path))

    if all_errors:
        for error in all_errors:
            print(error, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
