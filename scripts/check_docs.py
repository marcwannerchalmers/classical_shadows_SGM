"""Validate API-reference coverage and Python examples without runtime imports."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re

import griffe


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
API_REFERENCE = DOCS / "reference" / "api.md"
DIRECTIVE = re.compile(r"^:::\s+([\w.]+)\s*$")
LIST_ITEM = re.compile(r"^\s+-\s+([\w]+)\s*$")


@dataclass(frozen=True)
class ApiEntry:
    """One mkdocstrings directive and its explicitly selected members."""

    identifier: str
    members: tuple[str, ...]


def api_entries(path: Path) -> list[ApiEntry]:
    """Read mkdocstrings targets and explicit member lists from Markdown."""

    lines = path.read_text(encoding="utf-8").splitlines()
    entries = []
    for index, line in enumerate(lines):
        match = DIRECTIVE.match(line)
        if match is None:
            continue

        members = []
        for option_line in lines[index + 1 :]:
            if DIRECTIVE.match(option_line) or option_line.startswith("##"):
                break
            item = LIST_ITEM.match(option_line)
            if item is not None:
                members.append(item.group(1))
        entries.append(ApiEntry(match.group(1), tuple(members)))
    return entries


def source_module(identifier: str) -> tuple[str, tuple[str, ...]]:
    """Resolve the longest source-module prefix of a dotted API identifier."""

    parts = identifier.split(".")
    for length in range(len(parts), 0, -1):
        module_path = ROOT.joinpath(*parts[:length]).with_suffix(".py")
        if module_path.is_file():
            return ".".join(parts[:length]), tuple(parts[length:])
    raise ValueError(f"No source module found for {identifier}")


def documented_api_errors(entries: list[ApiEntry]) -> list[str]:
    """Return missing-object and missing-docstring errors from Griffe models."""

    loaded: dict[str, griffe.Module] = {}
    errors = []
    for entry in entries:
        module_name, object_path = source_module(entry.identifier)
        if module_name not in loaded:
            loaded[module_name] = griffe.load(
                module_name,
                search_paths=[ROOT],
                allow_inspection=False,
                docstring_parser="google",
            )

        obj = loaded[module_name]
        try:
            for part in object_path:
                obj = obj.members[part]
        except KeyError as exc:
            errors.append(f"{entry.identifier}: missing object {exc.args[0]!r}")
            continue

        objects = [(entry.identifier, obj)]
        objects.extend(
            (f"{entry.identifier}.{name}", obj.members.get(name))
            for name in entry.members
        )
        for identifier, member in objects:
            if member is None:
                errors.append(f"{identifier}: selected member does not exist")
            elif member.docstring is None or not member.docstring.value.strip():
                errors.append(f"{identifier}: public API has no docstring")
            elif isinstance(member, griffe.Function):
                missing = [
                    parameter.name
                    for parameter in member.parameters
                    if parameter.name not in {"self", "cls"}
                    and parameter.annotation is None
                ]
                if missing:
                    errors.append(
                        f"{identifier}: parameters lack annotations: "
                        + ", ".join(missing)
                    )
                if member.returns is None:
                    errors.append(f"{identifier}: return type is not annotated")
    return errors


def python_example_errors() -> list[str]:
    """Return syntax errors from fenced Python examples in documentation pages."""

    errors = []
    for path in sorted(DOCS.rglob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        in_python = False
        start = 0
        block = []
        for line_number, line in enumerate(lines, start=1):
            if not in_python and line.strip() == "```python":
                in_python = True
                start = line_number + 1
                block = []
            elif in_python and line.strip() == "```":
                try:
                    ast.parse("\n".join(block), filename=str(path))
                except SyntaxError as exc:
                    errors.append(
                        f"{path.relative_to(ROOT)}:{start + (exc.lineno or 1) - 1}: "
                        f"invalid Python example: {exc.msg}"
                    )
                in_python = False
            elif in_python:
                block.append(line)
        if in_python:
            errors.append(
                f"{path.relative_to(ROOT)}:{start - 1}: unclosed Python fence"
            )
    return errors


def main() -> None:
    """Run documentation source checks and exit nonzero on any error."""

    entries = api_entries(API_REFERENCE)
    errors = documented_api_errors(entries) + python_example_errors()
    if errors:
        raise SystemExit("Documentation checks failed:\n- " + "\n- ".join(errors))
    print(f"Documentation checks passed for {len(entries)} API entries.")


if __name__ == "__main__":
    main()
