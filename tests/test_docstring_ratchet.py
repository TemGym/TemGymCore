"""Documentation may only improve: no module gets more undocumented public names."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import docstring_ratchet  # noqa: E402


def test_public_names_are_not_less_documented_than_the_baseline():
    baseline = json.loads(docstring_ratchet.BASELINE.read_text())
    now = docstring_ratchet.survey()
    worse = {m: (baseline.get(m, 0), n) for m, n in now.items() if n > baseline.get(m, 0)}
    assert not worse, (
        "New undocumented public names (baseline, now): "
        f"{worse}. Add numpydoc docstrings; new modules must be fully documented."
    )
