# Cu(001) reflection diffraction

Open [sem_reflection.ipynb](sem_reflection.ipynb) and run from the top in the TemGymCore environment.

The default is a **30 keV, 3 nm ideal FWHM** probe incident at **4°** on ideal Cu(001), with a **256 × 256 Medipix3 chip, 55 µm pixels**, at **120 mm** camera length. The column stays mechanically inclined at **2°** and the objective-to-sample distance stays **40 mm**. Deflectors change incidence while preserving the surface hit. The objective starts with **1 mm spherical aberration** and **10 µm coma**, using the existing `KrivanekLens` Gaussian action.

## Inverting the column

Open [sem_reflection_calibration_pipeline.ipynb](sem_reflection_calibration_pipeline.ipynb)
for a separate, executed calibration experiment. It follows the microscope calibration
pipeline: **saved noisy pixels → measured order moments → independent multistart fit
→ held-out prediction and optional geometry score**. The browser and static comparison
show the initial guess and fitted column beside the same saved detector images, using
the existing native reflection plotting tools and calibration result table.

The nine unknowns are source position/size, three lens positions, three reference focal
lengths and initial camera distance. The surface is z=0. Neither the 40 mm working
distance nor the 120 mm initial detector distance is supplied to the fitter. Known
laboratory-Z detector translations and independent ±20% relative lens-power changes
provide acquisition diversity. Lattice spacing, energy, pixel pitch, detector PSF,
incidence, detector orientation and the surface hit are calibrated assumptions.

This first notebook uses an on-axis **2°** ideal column and **three separated local
Gaussian diffraction orders**, with the existing surface-envelope Jacobian. It is a
reduced model, not the full angular-packet simulation above. It explicitly checks
pixel integration, lateral clipping and order overlap; validation against full angular
reflection and real instrument data remains future work. The control law is relative
lens power, not a measured current-to-focal-length calibration.

The saved example has 111 fitting images and 18 unused prediction images. It recovers
a working distance of **40.154 ± 0.275 mm** and initial camera distance of
**119.99988 ± 0.00011 mm**, with prediction RMS **1.011 noise units** and local rank 9.
Source position remains weak, with **9.82 mm** local standard deviation. These errors
condition on the known model assumptions; they are not hardware accuracy claims.
Wrong initialization branches are retained in `fit.json`.

Implementation lives in `src/temgym_core/reflection_calibration.py`; outputs are under
`output/calibration_gaussian/`. The notebook saves its figures and an optional truth
file separately from recorded inputs. Fit and measurement fingerprints detect changed
numerical code or input files; plotting-only edits do not invalidate them. Use a new
output directory to regenerate a changed acquisition. Tests exercise recovery with the
truth file removed, native-adapter agreement, pixel integration and parameter derivatives.

## Controls

- **Objective aberrations:** toggle the aberrated and ideal objective.
- **Spherical / Coma / Astigmatism:** Krivanek C30 in mm, C21 in µm, and C12 in nm. Angle controls rotate coma and astigmatism.
- **Incidence:** 0.5–6°, measured from the surface. Rebuilds the incident and reflected beams.
- **Follow specular when angle changes:** translates the detector onto the new reflected chief ray. Uncheck to keep the camera stationary.
- **Camera z / Height:** independent laboratory translations; detector orientation stays fixed.
- **Centre on specular:** centres the chip at the current incidence.
- **Reset angle + camera:** restores 4° and 120 mm.
- **Full render + save:** renders at least one million outgoing Gaussian packets and exports the current view.

The microscope side view, probe at the specimen, and detector image appear side by side in one figure, with compact controls above. Sliders update on release using a coherent preview. The colour scale remains fixed to the initial peak. Angle and aberration changes invalidate the cached full render. Moving the detector propagates the cached beam to its new plane and evaluates the actual displaced sensor grid.

## Objective model

The objective applies `KrivanekLens` to every coherent Gaussian packet, including the phase, ray deflection and local curvature. Its coefficients describe the pupil about the fixed objective axis. The two deflectors are solved again after a coefficient change to preserve the central surface hit and requested incidence.

Angular decomposition is constructed through the ideal column, then propagated forward through the actual objective. A nonlinear lens is never inverted with the linear ABCD basis constructor. Displaced packets are intersected with the surface individually; their grating phase and the relative axial carrier between the incident and outgoing frames are retained.

The probe image is the coherent intensity in the column-transverse plane at the specimen, with coordinates in nanometres. Its field of view follows the probe width. The microscope envelope uses the local Gaussian action, while both images evaluate the coherent packet sum. The side-view plotting function accepts this propagated envelope through `incident_envelope`.

## Surface and reflection model

Cu(001) has a primitive square surface lattice along the two <110> directions, with period `a/√2 = 0.2556 nm` for `a = 0.3615 nm`, and one copper atom per cell. The lattice constant and clean 1×1 RHEED surface are documented in [this Cu(001) study](https://www.ritsumei.ac.jp/acd/re/src/memoirs/no15/no15_08.pdf).

The model retains the zeroth-Laue-zone row `(0,m)`, `m = -6…6`. Relative amplitudes use projected neutral-Cu Lobato–Van Dyck electron scattering factors, evaluated at the tangential reciprocal frequency. The [numerical parameters and frequency convention](https://github.com/abTEM/abTEM/blob/main/abtem/parametrizations/data/lobato.json) are included locally; no runtime download or abTEM dependency is needed. A fixed reference row `m = -256…256` normalizes the flux partition, independently of the retained subset. Evanescent orders, exit angles below 0.25°, and detector clipping are never renormalized.

`PotentialStepResponse` estimates a smooth angular reflection envelope at an abrupt attractive potential step. Its real inner potential defaults to **21.2 V**, from [Wang et al. (2003)](https://cpl.iphy.ac.cn/article/id/33824). The absorptive optical potential defaults to **0 eV**; it is editable, not a fitted Cu parameter. The scalar dispersion includes the relativistic energy correction and uses the decaying transmitted branch.

`BraggSample.coefficients_at(angle, voltage)` multiplies each order by the complex reflection amplitude `r(angle)`. The same coefficient has complex action `ΔS = -i log(c)/k`, so `exp(i k ΔS) = c`. An intensity fraction R therefore requires amplitude √R. `BraggSample.order_action` exposes this action. Direct multiplication preserves exact zero coefficients without evaluating log(0).

The response is evaluated for every incident angular packet. It is held constant across each narrow packet; this is an angular-spectrum approximation, not a local spatial absorption map. The notebook's existing 3D surface adapter handles direction changes and the Gaussian envelope Jacobian. The ordinary `BraggSample` thin-grating call does not itself reverse the beam's propagation direction.

## Reading the percentages

The status shows fractions of **incident flux**:

- **Surface response:** `|r|²` for a plane wave at the stated angle.
- **Propagating orders:** that response multiplied by the fixed weights of retained open orders.
- **Detector:** the integral of the actual coherent Gaussian image over the finite chip, divided by incident flux.

The first two percentages use a plane wave; the detector value includes the probe's finite angular spread and can smooth channel-opening thresholds. Basis packets are not independent electrons: summing their separate intensities would give the wrong flux.

The angular envelope and its distribution over projected top-layer orders form a rough model. They do not predict total SEM backscatter yield or dynamical Bragg rocking curves. Normal momentum dependence of atomic factors, deeper-layer interference, other Laue zones, surface relaxation, thermal diffuse and inelastic scattering are omitted. Quantitative reflection requires a dynamical surface-scattering calculation or measured complex order coefficients. The column omits finite apertures and partial coherence; the selected aberration coefficients are illustrative, not calibrated hardware values.

## Results and checks

Outputs are saved under `output/copper/`:

- [Microscope and detector view](output/copper/microscope_detector.png)
- [Angle table](output/copper/reflectivity.csv)
- [Initial numerical checks](output/copper/run.json)

Figures are exported as PNG, SVG and PDF. `detector.npz` retains the initial calculation; `camera_view.npz/.json` retain the latest saved camera, angle and aberration settings; the NPZ also stores the probe image and its extent. Metadata records the material, potentials, angle, detector pose and intensity budget. Previous silicon and paired-surface exports remain in their original folders.

The notebook checks probe reconstruction, fixed-pivot steering, independent detector backends, a unit-reflecting specular reference, angular quadrature and pixel integration. The full run requires the GPU renderer; set `RUN_MILLION=False` for the portable preview.
