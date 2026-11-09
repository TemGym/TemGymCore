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
| Long tests | `pytest -m slow` |
| Lint | `ruff check .` |
| Import every module in a clean install | `python tools/smoke_imports.py` |

The test suite runs on CPU in float64 unless `JAX_PLATFORMS` is set (`conftest.py`).

## Comparing complex fields

Complex fields are compared after removing the global phase. That is not sloppiness:
`Ray.pathlength` is the excess optical path, so $k\cdot\text{path}$ is about $10^{9}$ rad and
float64 resolves it to about $10^{-7}$ rad per beamlet.

## Pull requests

- Add or update tests, and a `CHANGELOG.md` entry under *Unreleased*.
- Public functions and classes need a numpydoc docstring with units.
- Keep components physical and small: one component, one physical element.
- Do not commit large binaries (the pre-commit hook rejects files over 1 MB). Publish figures
  and datasets as release assets.

Maintainers: releases, PyPI publishing and the GPU runner are described in
[MAINTAINING.md](MAINTAINING.md).
