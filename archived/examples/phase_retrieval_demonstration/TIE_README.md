# Transport-of-intensity reconstruction

Open **`transport_of_intensity.ipynb`** or
**`figures/06_TIE_phase_retrieval.png`**. This is a pixel-based TIE reconstruction;
it does not optimize Gaussian coefficients or run L-BFGS.

Run from the repository root:

```bash
.venv/bin/python examples/phase_retrieval_demonstration/tie_demonstration.py
```

## Acquisition and registration

The ideal microscope and the existing **modified-smiley Gaussian exit wave** are
held fixed. This does not switch back to the original `smiley()` transmission and
does not establish convergence of its 2,304-Gaussian representation.

Three new images are used for reconstruction. The previous widely separated,
asymmetric four-view acquisition is not used as a local TIE derivative.

| Specimen-equivalent defocus | IL1 current |
| --- | --- |
| −200 nm | 2.166455887 A |
| Focus | 2.205677875 A |
| +200 nm | 2.248973896 A |

Only IL1 changes. The camera stays at the same position, with 256 × 256 samples
at 55 µm pitch. The full native component matrix supplies A and B for each current;
`trace(inv(A) @ B)/2` gives the signed specimen-equivalent propagation distance.
A scalar-isotropy check verifies that `inv(A) @ B` has this form.

Before differentiating, detector coordinates are mapped using `x_D = A x_S`.
Intensity is multiplied by `abs(det(A))` to restore specimen-area units. This
accounts for the magnification and rotation changes induced by IL1. Quintic
spline interpolation maps the three camera images to a common **128 × 128 central
crop**, 176 nm wide, at 1.375 nm spacing. The saved full camera images remain
256 × 256. They are noiseless point samples, without pixel integration or blur.

Two additional images at ±100 nm validate the derivative step. They do not enter
the ±200 nm reconstruction. All five simulated images and calibrated settings
are saved, with the three-image acquisition distinguished from validation.

## TIE inverse

For the convention used by native Gaussian propagation,

\[
\nabla_\perp\cdot(I_0\nabla_\perp\phi)
=-\frac{2\pi}{\lambda}\frac{\partial I}{\partial z},
\qquad
\frac{\partial I}{\partial z}\approx
\frac{I(+\Delta z)-I(-\Delta z)}{2\Delta z}.
\]

The measured in-focus intensity is spatially varying, so the implementation
solves the variable-coefficient equation directly. It uses conservative
finite-volume differences with arithmetic face intensities and a sparse direct
linear solve. This retains the illumination gradients; it does not replace I
with a constant or use the auxiliary-current approximation of a two-Poisson
implementation.

The domain has **zero normal phase-current flux** at its boundary. The solver
projects the derivative's incompatible constant component out, pins one phase
value to remove the nullspace, and then sets the mean phase to zero. The removed
component is reported; it is about 0.0015% of the RHS norm in this run. Intensity
is positive everywhere in the chosen crop, and no intensity floor is needed.
These boundary and illumination assumptions would need checking for real images,
particularly near dark regions and vortices.

`solve_tie` receives only the three images through their focus intensity and
axial derivative, the wavelength and the pixel pitch. No true phase, smiley
shape, Gaussian coefficients or fitted complex-wave initialization enter the
inverse. True phase is used afterward to align the unobservable piston and
score errors inside a 70 nm radius. The plotted full field is larger than the
scoring circle.

## Numerical phase precision

The renderer uses the existing
`FreeSpacePropagator(include_axial_path=False)` option throughout the native
objective–IL1–IL2–IL3–PL1 column. This drops the common, unobservable axial carrier
while retaining relative optical actions and diffraction. Subtracting the common
path only after propagation does not recover precision already lost when small
relative actions were added to a metre-scale path.

This matters because a close-defocus intensity difference is small. The initial
full-axial-carrier trial gave approximately 0.12 rad RMS error, increasing to
0.23 rad when the step was halved. Direct-coordinate rendering showed that this
was not mainly registration error. Omitting the carrier during propagation
restored stable TIE results without changing the optical system or fitting phase.
The previous L-BFGS example is preserved as a separate comparison.

## Results and checks

- ±200 nm reconstruction: **0.00609 rad RMS**, phase correlation **0.999916**.
- ±100 nm validation: **0.00624 rad RMS**, phase correlation **0.999912**.
- Relative residual of the discretized TIE solve: approximately **4 × 10⁻¹⁵**.

The main run checks finite/positive registered intensities, isotropic calibrated
defocus, linear-system residual, phase RMS, correlation and half-step stability.
The error includes finite pixel sampling, registration, boundary assumptions and
discretization. This is a noiseless synthetic check of the represented wave, not
an experimental noise tolerance or Gaussian-basis convergence claim.

`tie_dataset.npz` contains camera and registered images, both axial derivatives,
both TIE phases, truth used for scoring, and the scoring mask. `tie_results.json`
records the calibration, method and metrics. Figures are exported as PNG, SVG
and PDF.

For the transport-of-intensity formulation, see Michael Reed Teague,
[Deterministic phase retrieval: a Green’s function solution (1983)](https://doi.org/10.1364/JOSA.73.001434).
