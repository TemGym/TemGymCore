# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- `temgym_core.enable_x64()`, `require_x64()` and `x64_enabled()`. Propagation
  (`run_to_end`), rendering (`evaluate_gaussians`), `find_imaging_condition` and the
  calibration workflows now raise a clear error when 64-bit arithmetic is off, instead of
  silently computing in single precision. Importing the package changes no JAX setting.
- `temgym_core.__version__`, derived from git tags (hatch-vcs).
- Frozen numerical baseline (`tests/test_golden.py`) covering physical constants, the
  focused example column, Gaussian propagation through the biprism column and the
  coherent renderer, on CPU and GPU.
- Public constants in `temgym_core.constants` (`ELEMENTARY_CHARGE`, `ELECTRON_MASS`,
  `SPEED_OF_LIGHT`, `PLANCK`, `HBAR`, `VACUUM_PERMEABILITY`).
- Continuous integration for Linux, Windows and macOS, a GPU workflow for a
  self-hosted runner, and a tag-driven release workflow.
- Documentation skeleton (Sphinx, MyST-NB), `tools/smoke_imports.py`, a docstring
  ratchet (`tools/docstring_ratchet.py`) and a test that runs the README examples.
- Optional extras: `plot`, `widgets`, `notebooks`, `plotly`, `symbolic`, `gpu`,
  `gpu-cuda12`, `dev`, `docs`.

### Changed
- **Breaking:** `pathlength` is now the *excess* optical path, for rays and Gaussian beamlets alike:
  what is accumulated beyond straight-line axial travel from the launch plane. Free space adds
  `d * (dx**2 + dy**2) / 2` (`FreeSpaceParaxial` used to add `d`), a thin element adds its
  optical path and kicks the slope by that path's gradient, and a ray parallel to the axis
  accumulates none. The common carrier `k * (z - z0)` is the same for every beamlet of one
  voltage and drops out of intensities. `FreeSpacePropagator`, `HardEdgeSolenoid` and
  `DoubleDeflector` lose their `include_axial_path` argument (the behaviour of
  `include_axial_path=False` is now the only one), and `FreeSpaceDirCosine` adds `(N - 1) * d`.
  The README ray example prints `pathlength=-0.048125` instead of `0.88875`.
- Rays and Gaussian beamlets now follow the same chief ray through every component, and a test
  (`tests/test_ray_gaussian_consistency.py`) holds them together: `GaussianActionComponent` gains
  a ray branch (the element's optical path plus a slope kick by its gradient), so `PhaseBiprism`,
  `ZernikeAberration`, the constant/linear/quadratic phase shifts and the specimen phase objects
  accept rays; `HardEdgeSolenoid` rays accumulate the quadratic action its Gaussian branch
  already had (they used to add the body length only); `KrivanekLens` rays follow the lens's
  `phase_shift` (they used to replace the path with the aberration function of the ideal slope);
  `DeflectionBiprism` adds `def_x * |x|` (it added `dx * x + dy * y`).
- **Breaking:** `transfer_matrices` constructors return NumPy arrays by default. Pass
  `xp=jax.numpy` for JAX or `xp=sympy` for symbolic matrices (the `symbolic` extra).
  Previously the default was SymPy, which was not a declared dependency, so the module
  could not be imported from a clean install.
- Python 3.11 or later; JAX `>=0.10.2,<0.11`; `interpax>=0.3.15`; NumPy `>=2.0`.
- `ase` and `numba` are no longer dependencies. The physical constants are explicit and
  numerically identical to the values previously taken from `ase.units`.
- `temgym_core.plotting` raises an `ImportError` naming the `plot` extra when Matplotlib is
  missing.
- The source distribution is 340 kB instead of 240 MB.

### Removed
- Commented-out test modules (`test_gaussians_old.py`, `test_gaussian_taylor_old.py`),
  two skipped tests for an unimplemented Gaussian ABCD propagator, and the legacy
  `.flake8`, `MANIFEST.in` and `test_requirements.txt` (superseded by `ruff` and the
  `dev` extra).

### Fixed
- `SigmoidAperture`'s Gaussian action was singular at the aperture's exact centre: the plain radius
  has a conical tip whose Hessian is about `1 / eps` (-1e15 1/m² for `radius / edge_width = 16`), so
  a beamlet centred exactly on the axis, such as the middle one of an odd lattice, left several times
  narrower. The distance is now softened over a twentieth of the edge width. Beamlets more than a
  tenth of an edge width from the centre, and the profile at the edge, are unchanged.
- `rectangular_input_wave` and `square_input_wave` normalised with the requested aperture area
  instead of the area the beamlet lattice actually covers, so the coherent sum had an interior
  intensity of ((L/d)^2/N)^2 rather than `amp`^2: 0.68 for a 40 um aperture with 22 x 22
  beamlets, 0.81 for 39.9 um with 21 x 21, and about 0.99 at hologram scale. The interior
  intensity is now `amp`^2 whatever the lattice size. Fringe period and contrast are unchanged;
  absolute intensities (and anything scaled from them) change by a factor N/(L/d)^2 (10% for the
  golden biprism column), and `tests/golden/` was regenerated.
- The README and the code map no longer describe `KrivanekLensCartesian` and Seidel ray
  support, which do not exist yet; every README example now runs in the test suite.
