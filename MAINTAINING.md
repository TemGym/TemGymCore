# Maintaining temgym-core

Steps that need a maintainer's accounts or machine. Everything else is automated in
`.github/workflows/`.

## One-time setup

### PyPI and TestPyPI (Trusted Publishing, no API token)

1. On <https://pypi.org/manage/account/publishing/> add a **pending publisher** for a new
   project named `temgym-core`: owner `TemGym`, repository `TemGymCore`, workflow
   `release.yml`, environment `pypi`. Repeat on <https://test.pypi.org> with environment
   `testpypi`.
2. In the GitHub repository: *Settings → Environments* → create `pypi` and `testpypi`. For `pypi`,
   add yourself as a required reviewer so a release needs one explicit approval.
3. The names `temgym-core` and `temgym` were both unclaimed on PyPI on 2026-09-29. Claiming
   `temgym` as a placeholder that depends on `temgym-core` is optional.

### Read the Docs

Import the repository at <https://readthedocs.org/dashboard/import/>. `.readthedocs.yaml`
builds with warnings as errors and installs the `docs` extra. Enable *Build pull requests*.

### Self-hosted GPU runner

The GPU kernels and their gradients can only be tested on a GPU, and hosted runners have none.
`gpu.yml` therefore targets a runner with the labels `self-hosted`, `linux`, `gpu`.

1. *Settings → Actions → Runners → New self-hosted runner* (Linux, ARM64 for the GH200 or
   x64). Follow the download and `./config.sh` commands GitHub prints, adding
   `--labels gpu`.
2. Run it as a dedicated unprivileged user, ideally with `./config.sh --ephemeral` under a
   service manager so each job starts clean. Install `uv` for that user. The driver must
   support CUDA 13 (or use the `gpu-cuda12` extra).
3. Put the runner in a runner group restricted to this repository.
4. **Security.** Never trigger this workflow from `pull_request`: a fork could run arbitrary
   code on your machine. It runs on pushes to `main`, nightly, and by hand (*Actions → GPU →
   Run workflow*, with the branch to test). Keep secrets off the runner.
5. Jobs set `XLA_PYTHON_CLIENT_PREALLOCATE=false` so a shared GPU is not filled.

### Branch protection

Require the `lint`, `tests`, `minimum-dependencies` and `package` checks on `main`.

## Releasing

1. Update `CHANGELOG.md` (move *Unreleased* under the new version) and check that any change to
   `tests/reference/` is explained.
2. Confirm `CI`, `GPU` and the docs build are green on `main`.
3. Tag a release candidate: `git tag v0.1.0rc1 && git push origin v0.1.0rc1`. `release.yml`
   publishes it to TestPyPI. In a clean environment run
   `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --pre temgym-core`
   and the README quick start.
4. Tag the release: `git tag v0.1.0 && git push origin v0.1.0`, approve the `pypi` environment
   when asked. The version comes from the tag (hatch-vcs); there is no version to edit.
5. Create a GitHub release from the tag with the changelog entry, and attach large example data
   (never commit it: the pre-commit hook rejects files over 1 MB).

## Numerical changes

`tests/test_reference_values.py` is the guard. If a change must move a frozen number, regenerate with
`JAX_PLATFORMS=cpu PYTHONPATH=src:tests python tests/reference/make_baseline.py`, and say why in the
pull request and in the changelog.
