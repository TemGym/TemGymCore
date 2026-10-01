# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- `temgym_core.render`: the differentiable GPU renderer behind `evaluate_gaussians`. Three JAX
  primitives (forward field, tangent map, adjoint map) with hand-written Pallas/Triton float64
  kernels make `method="sparse"` and `method="dense"` first-order differentiable with `jit`,
  `grad`, `jvp`, `jacfwd`, `jacrev`, `vmap` and `checkpoint`, with respect to every beamlet field
  and, in the default pixel-index mode, the detector pose (rotation, flip, pixel size, centre).
  Second derivatives raise `NotImplementedError` (use `scan`). Gradients are those of the
  cutoff-truncated sum: one pair predicate gates the forward, tangent and adjoint kernels.
- `RenderConfig` (cutoff, coordinate mode, radius classes, launch shapes; every field
  documented), `Tiles`, `evaluate_packed` and `describe_plan` (class occupancy of a bundle).
  `evaluate_gaussians(..., config=RenderConfig(...))`.
- Radius classes: beamlets are classified on the device by the footprint of their cutoff
  ellipse, each class with its own bin size, and beamlets that miss the detector are never
  visited, so a few very broad beamlets no longer slow a bundle of narrow ones and no host-side
  culling is needed. The adjoint cuts broad windows into work items of equal size and reduces
  them in a fixed order, so results are bitwise reproducible (no atomics).
- `coordinates="metre"` (`RenderConfig`): renders with the metre-space parameters and pixel
  coordinates exactly as the historical dense kernel; differentiating with respect to the
  detector pose then raises a clear error.
- Benchmarks `benchmarks/render.py` (synthetic cases and mixed radii),
  `benchmarks/hologram_gpu.py` (the 160,801-beamlet hologram bundle of
  `benchmarks/hologram_bundle.py`) and `benchmarks/compare.py` (fails on a 3x regression).
- `temgym_core.wave` (**experimental**): sampled wavefronts propagated through a column by linear
  canonical (Collins) transforms. `propagate(Wavefront, components, out_lattice)` splits a column at
  its thin non-linear elements (apertures, biprisms, phase plates, specimens become exact
  multiplicative masks) and applies one transform per linear span, planned from `run.column_matrix`,
  with the physical Gouy/Maslov phase taken from a chief Gaussian propagated through the same
  components (`column_transform`). Lattices are affine (rotation, scale, shear), so a Larmor rotation
  or astigmatism costs no interpolation; `input_lattice` adapts the source lattice backwards from a
  detector grid, and `resample` interpolates imported arrays between lattices with the exact
  band-limited interpolant. Three branches (matrix Fourier transform, drift + lattice map + chirp near an
  image plane, and the exact `B = 0` map) run in 0.1 to 0.5 ms at 1024^2 and 0.3 to 1.5 ms at 4096^2 on a GH200 (17 ms for the O(N^3) matrix-product mode), are
  differentiable in the field, the component parameters, the wavelength and the lattice, and `vmap`
  over the voltage for energy spread. A sampling planner raises `SamplingError` (naming the failing
  inequality) instead of returning aliased data; `check="off"` is the expert override.
  `benchmarks/wave.py` times the branches and `benchmarks/compare_wave.py` gates a regression at 3x.
  Tutorial: *Skipping planes*.
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
- **Behaviour:** `method="spatial"` is now `"sparse"` (alias kept) and `"gpu_kernel"` is
  `"dense"` (alias kept); both are differentiable. `"sparse"` no longer sends the whole bundle to
  the dense kernel when one beamlet is broad: on the hologram notebook's 160,801-beamlet bundle
  on a 1024 x 1024 camera the forward render takes about 20 ms instead of 4.4 s.
- **Behaviour:** the GPU renderers evaluate in the detector's pixel-index frame by default
  (`RenderConfig(coordinates="index")`), which re-rounds the packed parameters. Fields differ
  from the historical dense kernel by rounding, and by about 1e-7 of the peak for phases of
  1e10 rad; use `coordinates="metre"` for the historical arithmetic (it agrees to 1e-12 of the
  peak). `scan` is unchanged.
- **Behaviour:** a beamlet with a non-finite field contributes zero and receives a zero gradient
  on the GPU renderers; before, NaN spread to every pixel. A beamlet whose `Im Q_inv` is not
  positive definite is summed by every pixel, without forcing the rest of the bundle onto the
  dense kernel.
- `evaluate_gaussians` keywords `tile_pixels`, `tile_beams`, `spatial_cutoff` and `spatial_bins`
  default to `None` (tuned values); explicit values still work and `tile_beams` may be any power
  of two. `gaussian_image(tile_beams=...)` defaults to `None` for the same reason.
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
- Private helpers `evaluate._build_spatial_bins`, `_SpatialBins`, `_spatial_kernel`,
  `_evaluate_spatial` and `_evaluate_gaussians_spatial`, replaced by the plan of
  `temgym_core.render.plan` (`describe_plan` is the supported diagnostic).
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
- `gpu_kernel_available()` cached "no GPU" for the whole process when its first call happened
  while JAX was tracing (inside `jit`, `grad` or `vmap`), because the probe could not convert a
  tracer; `evaluate_gaussians(method="auto")` or `"sparse"` then silently used the far slower
  portable renderer (88 s instead of 50 ms on the hologram bundle). The probe now runs eagerly
  in a worker thread.
- `rectangular_input_wave` and `square_input_wave` normalised with the requested aperture area
  instead of the area the beamlet lattice actually covers, so the coherent sum had an interior
  intensity of ((L/d)^2/N)^2 rather than `amp`^2: 0.68 for a 40 um aperture with 22 x 22
  beamlets, 0.81 for 39.9 um with 21 x 21, and about 0.99 at hologram scale. The interior
  intensity is now `amp`^2 whatever the lattice size. Fringe period and contrast are unchanged;
  absolute intensities (and anything scaled from them) change by a factor N/(L/d)^2 (10% for the
  golden biprism column), and `tests/golden/` was regenerated.
- The README and the code map no longer describe `KrivanekLensCartesian` and Seidel ray
  support, which do not exist yet; every README example now runs in the test suite.
