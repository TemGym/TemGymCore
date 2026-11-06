# Contributing to temgym-core

## Set up

```bash
git clone https://github.com/TemGym/TemGymCore.git
cd TemGymCore
uv venv --python 3.12 && source .venv/bin/activate      # or python -m venv
uv pip install -e ".[dev]"                              # add ,gpu for an NVIDIA GPU
pre-commit install
```

## Everyday commands

| Task | Command |
|---|---|
| Fast CPU tests (what CI runs) | `pytest -m "not gpu and not slow"` |
| GPU kernels | `JAX_PLATFORMS=cuda pytest -m gpu` |
| Long tests | `pytest -m slow` |
| Lint | `ruff check .` |
| Import every module in a clean install | `python tools/smoke_imports.py` |

The test suite runs on CPU in float64 unless `JAX_PLATFORMS` is set (`conftest.py`).

## Numbers are frozen on purpose

`tests/test_reference_values.py` compares the physical constants, the focused example column, Gaussian
propagation through the biprism column and the coherent renderer with values stored in
`tests/reference/`. If a change moves one of them, say why in the pull request and regenerate
with `tests/reference/make_baseline.py`. A refactor should move none of them.

Complex fields are compared after removing the global phase, to 1e-6 of the peak. That is
not sloppiness: `Ray.pathlength` is the excess optical path, so $k\cdot\text{path}$ is about
$2\times10^{9}$ rad on the reference column and float64 resolves it to about $2\times10^{-7}$ rad
per beamlet; the GPU renderers agree with the CPU baseline to about $10^{-8}$ of the peak.

## Pull requests

- Add or update tests, and a `CHANGELOG.md` entry under *Unreleased*.
- Public functions and classes need a numpydoc docstring with units.
- Keep components physical and small: one component, one physical element.
- Do not commit large binaries (the pre-commit hook rejects files over 1 MB). Publish figures
  and datasets as release assets.
- GPU changes: start the *GPU* workflow on your branch (Actions → GPU → Run workflow);
  it runs on a maintainer-controlled runner and is deliberately not triggered by pull requests.

Maintainers: releases, PyPI publishing and the GPU runner are described in
[MAINTAINING.md](MAINTAINING.md).
