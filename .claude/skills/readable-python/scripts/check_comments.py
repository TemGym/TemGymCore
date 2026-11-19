"""Check the comments, docstrings and markdown cells of Python files and notebooks.

The script collects the prose of each file and gives the prose to the Simplified Technical
English linter (`ste_lint.py`, from github.com/danyuchn/asd-ste100-skill, MIT licence).
The script adds the rules of the readable-python skill that the linter does not have.

Usage:
    python check_comments.py FILE [FILE ...]

The exit status is 1 when a file has a finding of the level "fix".
"""
import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

import ste_lint

# Comments that are instructions to tools. They are not prose.
TOOL_COMMENT = re.compile(r"#\s*(!|%%|noqa|type:|pragma|fmt:|ruff:|pylint:|mypy:)")

# The rules of the readable-python skill that the linter does not have: (name, level, pattern).
OWN_RULES = [
    ("x-not-y", "fix", re.compile(r",\s+not\s+(?:a|an|the|just|only)?\s*\w+", re.I)),
    ("rhetorical-word", "fix", re.compile(r"\b(hence|thus|note that|simply|importantly)\b", re.I)),
    ("session-history", "fix", re.compile(
        r"\b(changed from|now uses?|new version|as discussed|no longer|previously|was:)\b", re.I)),
    ("colon-joins-statements", "look", re.compile(r"\w:\s+[a-z]\w*\s+\w+")),
    ("dash-joins-statements", "look", re.compile(r"\s(--|—|–)\s")),
]


def comments(source):
    """Return (line number, text) for each comment in Python source."""
    found = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and not TOOL_COMMENT.match(token.string):
            found.append((token.start[0], token.string.lstrip("#").strip()))
    return found


def docstrings(source):
    """Return (line number, text) for each docstring line before the first numpydoc section."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if ast.get_docstring(node) is None:
            continue
        first = node.body[0].value
        lines = first.value.splitlines()
        for offset, text in enumerate(lines):
            # A row of dashes under a heading starts the numpydoc sections, which are tables.
            if offset + 1 < len(lines) and re.fullmatch(r"\s*-{3,}\s*", lines[offset + 1]):
                break
            found.append((first.lineno + offset, text.strip()))
        found.append((first.lineno + len(lines), ""))
    return found


def prose_of_python(source, place):
    """Return (place, text) lines of prose, with an empty line between separate blocks."""
    numbered = sorted(comments(source) + docstrings(source))
    lines, last = [], None
    for number, text in numbered:
        if last is not None and number > last + 1:
            lines.append((None, ""))
        lines.append((place.format(line=number), text))
        last = number
    return lines


def prose_of_file(path):
    """Return (place, text) lines of prose for a .py file or a notebook."""
    if path.suffix != ".ipynb":
        return prose_of_python(path.read_text(), f"{path}:{{line}}")
    lines = []
    for index, cell in enumerate(json.loads(path.read_text())["cells"], start=1):
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            for number, text in enumerate(source.splitlines(), start=1):
                lines.append((f"{path} cell {index} line {number}", text))
        elif cell["cell_type"] == "code":
            # IPython magics are not Python. The check reads them as comments of a tool.
            source = re.sub(r"^\s*[%!].*$", "pass", source, flags=re.M)
            lines += prose_of_python(source, f"{path} cell {index} line {{line}}")
        lines.append((None, ""))
    return lines


def check(path):
    """Print the findings for one file and return the number of findings to fix."""
    lines = prose_of_file(path)
    text = "\n".join(text for _, text in lines)
    findings, _ = ste_lint.lint(text)
    to_fix = 0
    for finding in findings:
        place = lines[finding["line"] - 1][0]
        level = "fix" if finding["level"] == "advisory-free" else "look"
        to_fix += level == "fix"
        print(f"{place}: {level}: {finding['rule']}: {finding['match']!r}")
    for place, line in lines:
        for name, level, pattern in OWN_RULES:
            match = pattern.search(re.sub(r"`[^`]*`", "", line))
            if match:
                to_fix += level == "fix"
                print(f"{place}: {level}: {name}: {match.group(0)!r}")
    return to_fix


if __name__ == "__main__":
    total = sum(check(Path(name)) for name in sys.argv[1:])
    print(f"{total} to fix")
    sys.exit(1 if total else 0)
