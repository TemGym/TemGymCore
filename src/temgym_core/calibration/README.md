# Reading the calibration code

Start with `workflow.py`, which implements the notebook's public calls.
Read each stage as **inputs → saved measurements → fitted result**.
The package is a coaxial, two-voltage calibration procedure, not a general
replacement for TemGym's component API.

| File | Responsibility |
|---|---|
| `workflow.py` | Public steps, defaults and stage reuse |
| `acquisition.py` | Recorded condenser controls and synthetic acquisition |
| `gaussian.py` | Known specimen, coherent Gaussian images and numerical convergence |
| `projector_images.py` | Projector image records, calibrated pairs and intensity initialization |
| `condenser_images.py` | Independent beam estimates for initialization |
| `projector_model.py`, `condenser_model.py` | Physical parameters mapped into native TemGym components |
| `projector.py` | Projector image moments and fitting |
| `apertures.py` | Transmission, current uncertainties and independent aperture centres |
| `inference.py` | Shared beam uncertainty, multiple starts and final joint fit |
| `checks.py` | Frozen predictions, geometry scoring and display |
| `storage.py` | Hash recorded array contents for cache validation |
| `presets/` | Demonstration controls, guesses and bounds |

The underlying physics stays in the existing library: `solenoid.py`,
`coherent_image.py`, `coherent_moments.py`, `coherent_microscope.py` and
`intensity.py`. Optimization uses `jax_least_squares.py` and
`shared_least_squares.py`. The calibration code does not define another lens
matrix or Gaussian summation engine.

## Audit decisions

Checkpoint `61457c3` preserves the complete state before cleanup. The maintained
functions were extracted from the older experiment scripts, keeping their
numerical operations and consolidating their imports. Superseded acquisition
studies, alternate fitting workflows, the direct-pixel fitting prototype,
archived fitting notebooks and intermediate reports were removed. The final
Gaussian recovery report remains as a reference result.

The cleanup also fixes incomplete copying of known-specimen files, rejects
nested acquisition-copy folders, includes the presets in the installed wheel,
and removes unused beam Jacobians. Stage signatures now cover all TemGym
Python source, including the low-level Gaussian renderer. A source change
invalidates saved stages rather than silently reusing an older calculation.

Some complexity is necessary: coherent sampling and detector integration need
convergence checks; correlated measurements need shared covariance; the final
fit must return to the original image measurements; alternative aperture
centres must remain visible. Removing these would change the experiment or
make its uncertainty misleading. The intensity approximation remains useful
for initialization and the explicit faster acquisition option.

The saved solution still has two acceptable second-aperture centres. Local
rank and successful prediction checks do not prove global uniqueness.

## Validation of this cleanup

- 70 targeted tests pass, covering matrix/Gaussian agreement, image moments,
  correlated currents, shared covariance, nonlinear refinement and stage reuse.
- The full 200/195 kV acquisition completes through the new API using the
  intensity option. The moved Gaussian camera separately matches analytic
  pixel integrals; the complete Gaussian batch was not regenerated for this audit.
- On the recorded Gaussian dataset, the full 52-parameter residual reproduces
  the reference chi-squared of 1650.0543239. Directional finite differences agree
  with JAX to 2.3e-6 relative or better. Recomputed covariance differs by 1.6e-12
  relative. This check forbids generating-parameter reads and Gaussian rendering
  during fitting. It re-evaluates the saved solution, rather than repeating the
  multiple-start search.
- The built wheel imports outside the repository and includes all seven presets.
  Notebook schema and code cells validate; old execution outputs were cleared.

Calibration Python shrank from 39 files / 17,159 lines to 13 files / 6,984 lines
(59% fewer lines). The underlying reusable TemGym physics remains separately tested.
