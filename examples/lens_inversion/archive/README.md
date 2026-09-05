# Research checkpoint and cleanup

The complete pre-cleanup source tree is preserved in git commit **3d51e0e**
(`Checkpoint NEOARM inversion research before section-wise cleanup`). This
checkpoint records work in progress, including unresolved fits and historical
assumptions; it is not a release or an acceptance certificate.

The cleanup removes 91 top-level experimental Python drivers, seven
payload-bound condenser modules, 90 tests specific to those experiments,
three synthetic condenser manifests, three accumulated status documents,
and two historical notebook/document copies from the working tree. Their
original paths and contents remain in the checkpoint. The duplicate paper
copy under `references/` is removed; the supplied PDF under `paper/` is kept.
The main notebook is replaced by a short section-wise recipe and a read-only
audit of the real free-lens table.

Reusable library modules and their tests are retained. The original user
notebooks and the pre-existing `stashed/` directory are untouched. The
synthetic configuration is moved to `tests/fixtures/neoarm_synthetic_column.toml`
and labelled as a regression fixture. It is not a set of NEOARM commands.

## Recovering or inspecting an experiment

Inspect one historical source without changing the working tree:

```bash
git show 3d51e0e:examples/lens_inversion/neoarm_shared_material_images.py
```

For a complete runnable source layout, create a separate worktree at the
checkpoint using a new, unused destination. Configure its Python environment
separately. Historical experiment imports and source checksums assume the
original layout; copying one script into the cleaned notebook directory will
not generally recreate that environment.

Historical generated images, reports, and vector figures under
`examples/generated/neoarm_inversion/` were **not deleted or modified by this
cleanup**. They remain local ignored artifacts, not data committed to git.
Their embedded source hashes describe the historical source snapshot, not
the cleaned tree. A fresh clone obtains source code and the supplied paper,
but not those large generated arrays.

## Evidence worth retaining

- `shared_material_images.json` reports GPU rendering with 10,504,081
  Gaussians at the finest level. The latest conditional 40k image fit has
  magnification error −0.04783% and rotation error −0.02552°. It fits image
  parameters with prescribed sample illumination, not physical lens geometry.
- `condenser_cold_joint_endpoint.json` records an accepted stationary
  29-parameter first-order fit, including six downstream response coordinates.
  It uses separate beam/aperture data and fixed objective assumptions; broad
  initialization success and global uniqueness remain unproved.
- `shared_projector_images.json` contains 135 newly generated images from
  the same objective model. `shared_projector_recovery.json` records no
  accepted stationary physical-projector fit. The all-column recovery
  scaffold was not an accepted full-column result.
- `condenser_factorized_validation.json` verifies selected linear Gaussian
  transports after numerical factorization. The separate
  `condenser_accepted_nonparaxial_audit.json` finds physical approximation
  errors larger than assumed noise in some settings. Numerical convergence
  and accurate instrument physics are separate results.

The old image-derived four-projector fit and the condenser fit used different
objective assumptions. Their posteriors cannot be combined as though they
calibrated the same microscope. The new section-wise recipe keeps a shared
objective/sample interface and propagates downstream calibration uncertainty.

## Test-suite boundary

Before cleanup, collection of the full `tests/` suite already failed in
`test_constants.py` (missing `compute_I0_Cf_from_focal_rotation` export) and
`test_gaussians.py` (uninstalled `skimage`). These tracked, pre-existing test
files are retained and their errors are not hidden by a new pytest setting.
The cleanup verification reports its explicit exclusions separately from the
retained inversion tests. No historical test is described as passing merely
because its associated retired experiment is no longer in the working tree.

Cleanup verification on 5 September 2026:

- Retained suite, explicitly excluding those two collection-error files:
  **743 passed, one failed** in 357 s. The failure is
  `test_component.py::test_double_deflector_jacobian_matches_analytic_matrix`,
  which calls `double_deflector_matrix_5x5` with an unsupported `shift_x`
  keyword. That test and the relevant component/matrix source are unchanged
  from pre-inversion commit `49e090a`.
- New read-only free-lens table tests: **six passed**.
- After moving the synthetic fixture to its committed `tests/fixtures/`
  location, the reader, diffraction, and aperture subset passed **19 tests**
  (including those same six reader tests).
- Clean notebook: **all 14 cells executed**, with no fitting jobs launched.
- GPU check: the Pallas kernel executed on CUDA and passed its independent
  numerical comparison in float64.
- The control figure exports SVG, PDF, and 300-dpi PNG with metadata.
  The raster was inspected for layout; log-axis minor labels were suppressed
  to prevent overlap with the selected magnification ticks.

The executed notebook and figure are local outputs under
`examples/generated/neoarm_inversion/section_recipe/`. Reproduce the retained
suite with:

```bash
OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -m pytest -q tests \
  --ignore=tests/test_constants.py --ignore=tests/test_gaussians.py
```

This command includes the six new reader tests, so its expected pass total
is 749 if the same known failure remains. The suite is not claimed fully green.
