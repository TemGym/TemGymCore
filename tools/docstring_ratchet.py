"""Count public functions, classes and methods without a docstring, per module.

``tests/test_docstring_ratchet.py`` fails when any module gets worse than the
numbers in ``tests/golden/docstring_baseline.json`` and requires every new module
to be fully documented. Improve a module, then lower its number::

    python tools/docstring_ratchet.py --update
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "temgym_core"
BASELINE = ROOT / "tests" / "golden" / "docstring_baseline.json"


def missing_docstrings(path):
    """Names of public defs in ``path`` that lack a docstring (methods as Class.method)."""
    tree = ast.parse(path.read_text())
    missing = []

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if child.name.startswith("_"):
                    continue
                name = f"{prefix}{child.name}"
                if ast.get_docstring(child) is None:
                    missing.append(name)
                if isinstance(child, ast.ClassDef):
                    visit(child, f"{name}.")

    visit(tree, "")
    return missing


def survey():
    counts = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        module = str(path.relative_to(PACKAGE))
        counts[module] = len(missing_docstrings(path))
    return counts


if __name__ == "__main__":
    counts = survey()
    if "--update" in sys.argv:
        BASELINE.write_text(json.dumps(counts, indent=1, sort_keys=True) + "\n")
        print(f"wrote {BASELINE} ({sum(counts.values())} undocumented public names)")
    else:
        print(json.dumps({k: v for k, v in counts.items() if v}, indent=1))
        print(f"{sum(counts.values())} undocumented public names")
