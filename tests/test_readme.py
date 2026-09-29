"""Every Python block in the README runs, and prints what the README says it prints."""
import re
from pathlib import Path

import numpy as np

README = Path(__file__).parents[1] / "README.md"


def readme_namespace():
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.DOTALL)
    assert len(blocks) >= 4
    namespace = {"__name__": "readme"}
    for block in blocks:
        exec(compile(block, str(README), "exec"), namespace)  # noqa: S102
    return namespace


def test_readme_examples_run_and_match_the_text():
    ns = readme_namespace()
    ray_out = ns["ray_out"]
    np.testing.assert_allclose([ray_out.x, ray_out.y, ray_out.dx, ray_out.dy, ray_out.z],
                               [0.275, 0.4, 0.05, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(ray_out.pathlength, 0.88875, atol=1e-12)
    np.testing.assert_allclose(
        ns["matrix"],
        [[0.5, 0, 0.75, 0, 0], [0, 0.5, 0, 0.75, 0], [-1, 0, 0.5, 0, 0],
         [0, -1, 0, 0.5, 0], [0, 0, 0, 0, 1]],
        atol=1e-12,
    )
    np.testing.assert_allclose(ns["gradient"], 0.125, atol=1e-12)
    assert abs(float(ns["image"][32, 32]) - 0.68) < 0.01
