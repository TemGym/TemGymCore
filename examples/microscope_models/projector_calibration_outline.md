# Projector calibration: the experiment and its mathematics

Status: design outline, 2026-11-05. Simulation results are in
`projector_calibration.ipynb`; the code is in `projector_calibration.py` (figures in `projector_figures.py`).

**Decision:** the **no-sample track**. Everything is measured from beam centroids in vacuum, with
raw deflector kicks. A sample (a cross grating with gold) is used only for **two anchor frames**,
which turn kick units into metres and radians, and optionally for a few **validation frames**.
The beam width and shape are never used.

---

## 1. What is measured: A and B

The post-specimen column (objective post-field, IL1–IL3, PL1, then a drift to the camera) is
paraxial and rotationally symmetric. Write transverse positions as complex numbers, x + iy. The
specimen-to-detector map is then

$$
x_d = A(u, V)\,x_s + B(u, V)\,\theta_s ,
$$

- $x_s$ and $\theta_s$ are the position and angle of a ray at the specimen plane;
- $u$ is the vector of lens DAC values (the lens setting), and $V$ is the high tension;
- $A$ and $B$ are complex scalars. Their magnitudes are the magnification and the "camera length".
  Their arguments are the Larmor rotations of the lenses.

The camera adds a fixed rotation $e^{i\varphi}$, and $x_d$ is in pixels (pitch $p_{px}$).

Each lens setting $u$ therefore carries two complex numbers, $A(u)$ and $B(u)$. The projector
model (16 parameters: 6 gaps, 5 lens lengths, the objective strength, 4 lens gains) predicts both
for every $u$ and $V$. The calibration fits that model to measured A and B.

## 2. Why the beam centroid is enough, whatever the beam looks like

Let the beam at the specimen be any distribution of rays $\rho(x_s, \theta_s)$: a parallel beam, a
crossover, or a distorted spot. Because the map is linear, the centroid on the detector is

$$
\langle x_d \rangle = A\,\langle x_s \rangle + B\,\langle \theta_s \rangle .
$$

This is linearity of expectation. Nothing about the shape, width or coherence of the beam enters.

A pure shift $\delta$ adds $\delta$ to every $x_s$; a pure tilt $\tau$ adds $\tau$ to every
$\theta_s$. The difference of a ± pair is then

$$
\Delta_{\text{shift}} = 2A\,\delta, \qquad \Delta_{\text{tilt}} = 2B\,\tau ,
$$

exactly, and independent of $\rho$.

**Conditions:**
- The whole beam is on the camera.
- No aperture after the specimen clips it.
- The dark/background frame is subtracted.
- The optics are paraxial over the beam. This is set aside for now, as is deflector linearity.

The second moments (the width) do depend on $\rho$. Using them would need a beam model, which is why
they are not used.

## 3. Kicks in raw deflector units

A deflector command $c$ (CLA DAC values, complex: x + iy) gives

$$
\delta = \kappa_s\,r(V)\,\gamma(V)\,c, \qquad \tau = \kappa_t\,r(V)\,\gamma(V)\,c .
$$

- $\kappa_s$ and $\kappa_t$ are the unknown kick scales, in nm and µrad per deflector unit.
- $r(V) = p_0/p$ is the magnetic-deflection scaling with electron momentum (1.0148 at 195 kV).
- $\gamma(V)$ is an unknown factor for anything the console does to the deflectors when the HT
  changes. $\gamma(V_0) = 1$ by definition, and $\gamma(195\,\text{kV})$ is fitted.

The measured data are

$$
m_{u,V,\text{shift}} = e^{i\varphi} A(u,V)\,\kappa_s\,r\gamma\,\Delta c, \qquad
m_{u,V,\text{tilt}} = e^{i\varphi} B(u,V)\,\kappa_t\,r\gamma\,\Delta c .
$$

**The 195 kV kick factor (checked 2026-11-04).** Fitting $\gamma$ costs almost nothing:

| | detector error, typical anchors | detector error, good anchors |
|---|---|---|
| $\gamma$ known | 0.91 mm | 0.37 mm |
| $\gamma$ fitted | 0.98 mm | 0.43 mm |

$\gamma$ is recovered to about $3\times10^{-5}$. A wrong fixed $\gamma$ is caught by $\chi^2$:
0.3% off gives $\chi^2/\text{dof} = 31$, and a console that holds the kick constant gives 800.
So $\gamma$ is always fitted. This removes the question of what the JEOL console does at an HT
change.

**Kick purity (to test).** Two effects the current model leaves out:

1. **Cross-talk.** Unless the pivot points are set exactly, a "shift" also tilts:
   $\tau = \epsilon_{st}\,c$, and vice versa. The shift data become
   $e^{i\varphi}(A\kappa_s + B\epsilon_{st})\Delta c$. Since $A(u)/B(u)$ varies across the
   settings, $\epsilon_{st}$ and $\epsilon_{ts}$ should be identifiable as extra complex
   unknowns. Not yet tested.
2. **Shift and tilt directions.** The shift and tilt deflections may reach the specimen rotated
   relative to each other, for example by the objective pre-field. That makes $\kappa_s$ and
   $\kappa_t$ complex. One common phase is degenerate with $\varphi$; the relative phase
   $\arg(\kappa_t/\kappa_s)$ is a further unknown. The current code uses real $\kappa$ and one
   $\varphi$, so it assumes the two are aligned. Not yet tested.

Both get tested in the fit-robustness runs, together with random columns and error-bar
calibration.

## 4. Gauges: what the beam data cannot determine

At one HT, the data are unchanged under

$$
(A, B, \kappa_s, \kappa_t) \;\to\; (A/\alpha,\; B/\beta,\; \alpha\kappa_s,\; \beta\kappa_t)
$$

for any real $\alpha$ and $\beta$. They also have the projective gauge of the lens model: the axial
scale is tied to the tilt scale ($dz = s_t z$), and the detector gap is fixed by convention.

The beam data therefore give $A$ and $B$ up to **two real numbers**, one for shifts and one for
tilts, plus the camera rotation. This is the stage-1 result: a model that predicts where the beam
lands for any lens setting and any kick, in kick units. That is all you need to drive the microscope.

The two numbers are what a sample supplies.

## 5. The anchors: two frames of a cross grating with gold

**Image anchor (|A|).** A cross grating of period $p_g$ images to fringes of period

$$
P = |A(u_1)|\,p_g / p_{px} \quad \text{pixels}.
$$

For coherent illumination this holds at any focus. Each diffraction order is a beam, and the
fringe period comes from their angular separation alone. The anchor setting $u_1$ need not be in
focus; it needs enough periods in the frame.

The model predicts $A$ at every set, so the anchor can be taken at **any** set. Choose the set by
the grating, not by the usual viewing magnification:

- **Period visible at 40–60k.** The squares are visible there, but only a few periods fit in the
  frame. The field of view is 28.2 mm / 40 000 = 0.70 µm.
- **Replica gratings are certified as an average period.** The local period of a replica varies
  from cell to cell, so one or two cells can be off by much more than the certified tolerance.
  This was not quantified here; check the certificate.
- **Better: a low-magnification set with many periods.** Use 50–100 or more periods across the
  frame, measured from the FFT peak or a lattice fit. This averages the local errors and comes
  closest to the certified value. The stage-1 lens settings reach that range.

For example, a 463 nm period (2160 lines/mm) on the binned 256 px × 110 µm camera, with the
simulated column:

| set | magnification | period (px) | periods across the frame |
|---|---|---|---|
| 0 (40k row) | 40 000 | 168 | 1.5 |
| 4 | 8 600 | 36 | 7 |
| 6 | 1 300 | 5.4 | 47 |

Rule: number of periods ≈ (frame width / magnification) / grating period. Pick the lowest-
magnification set where the period is still at least about 4 px.

The code currently assumes the image anchor is at set 0, and needs changing.

**Working choice (2026-11-05).** A grating with about 3 × 3 squares in the frame at the 40k row
(set 0; period about 230 nm, about 85 px), measured by a lattice fit. Repeat the anchor at 2–3
stage positions to average, and measure, the cell-to-cell variation.

Cost of the grating uncertainty (8 runs each, 0.01 px wobble, rings 0.1%):

| grating anchor | absolute magnification | IL1 → PL1 position error | detector | χ²/dof |
|---|---|---|---|---|
| 0.2% | 0.09% | 0.5–0.8 mm | 0.8 mm | 0.97 |
| 0.5% | 0.28% | 1.0–1.3 mm | 0.8 mm | 0.95 |
| 1% | 0.58% | 1.9–2.3 mm | 0.8 mm | 0.94 |
| 2% | 1.2% | 3.6–4.3 mm | 0.7 mm | 0.95 |

The beam-position error in stage 1 is 0.018 px throughout.

Above about 0.5%, the IL and PL position errors grow in proportion to the grating error, at about
2 mm per 1%. The detector position is set by the gold rings and does not change.

**$\chi^2$ cannot detect a wrong anchor.** It stays near 1 in every row, because the anchor is a
gauge choice that the beam data cannot contradict. The grating error therefore has to be measured
(repeat positions, certificate), not inferred from the fit.

**Diffraction anchor (|B|).** At the diffraction setting ($A \approx 0$), a polycrystalline
film gives rings of radius

$$
R_{hkl} = |B(u_D)|\,\lambda / d_{hkl} \,/\, p_{px} .
$$

With $\lambda$ = 2.508 pm at 200 kV and $|B(u_D)|$ = 0.754 m:

| ring | d (nm) | angle (mrad) | radius (px) |
|---|---|---|---|
| Au(111) | 0.2355 | 10.65 | 73 |
| Au(200) | 0.2039 | 12.30 | 84 |
| Au(220) | 0.1442 | 17.39 | 119 (at the edge) |

Rings carry no orientation, but only $|B|$ is needed. The ring radius does not depend on
magnification: the diffraction setting is a lens setting, not a magnification. The parallel beam illuminates many
particles, so rings rather than single spots are expected; spotty rings still give the radius.

Pure Au has a lattice constant known far better than the measurement. **Many replicas are
shadowed with Au–Pd or Pt–Pd instead.** An alloy's lattice constant depends on its composition and
can differ from Au by percent. That would make the ring anchor as uncertain as the composition.
Check what the decoration is. Ellipticity of the rings would show projector distortion, as a free check.

With a gold-shadowed cross grating, both anchors come from **one sample at one stage position**.
Only the lens DAC values change between the two frames.

**The two scales.** The fit adds two residuals,

$$
\frac{|A(u_1)|\,p_g}{P\,p_{px}} - 1, \qquad \frac{|B(u_D)|\,\lambda}{R\,d\,p_{px}} - 1 ,
$$

weighted by the anchor tolerances. These fix $\alpha$ and $\beta$, and hence $\kappa_s$ and
$\kappa_t$. The anchors are taken at the specimen height, so they also define which plane
"specimen" means.

**What limits the anchors:**
- The accuracy of the grating period. Replica gratings are nominal; check the certificate. The
  simulations assumed 0.2% (typical) or 0.05% (good).
- Image distortion over the field.
- Specimen tilt (foreshortening).
- Specimen height, which changes $A$ through the objective field.

These errors set the limit for absolute lens positions (section 7). Imaging the sample at more
settings does not reduce them, because the period error is common to all frames.

## 6. Why not a sample at every setting

| | beam + kicks | sample at every setting |
|---|---|---|
| per setting | $A$ and $B$, magnitude and phase | $A$ from the period. $B$ only near diffraction-like settings. Patterns at settings in between are mixed and hard to analyse. |
| units | relative, two scales for the session | absolute in every frame |
| measurement | centroid difference, independent of shape | pattern analysis; spacing must be resolvable |
| coverage | anywhere the beam stays on the camera | only where the pattern is clean |
| systematics | kick purity (section 3) | period accuracy, distortion, specimen tilt and height, drift, contamination, finding the same area again |

At one HT, the beam data determine everything except the two scales. Further sample frames repeat
information the beam gives more precisely, and add the period error again.

The sample-heavy alternative in its strongest form: a crystal in a small beam. Each order is a copy
of the beam displaced by $B\lambda/d$, with fringes of period $|A| d$ where the copies overlap. One
frame gives $A$ and $B$ without kicks, but only at settings where both are measurable. This has not
been simulated.

## 7. Fitting

Weighted least squares on all $m$, plus the anchor residuals. With Gaussian centroid noise
$\sigma = \text{wobble}/\sqrt2$ per component, this is maximum likelihood.

**Free parameters:**
- the projector parameters, minus the gauges at one HT;
- $\varphi$;
- $\log\kappa_s$ and $\log\kappa_t$;
- $\gamma$ when two HTs are used.

**Start.** Lens gains from a closed-form regression on the rotations. $\varphi$ from the strongest
shift, tried at $\varphi$ and $\varphi + \pi$. Then a continuation: kicks alone, then the scales
from the anchors, then everything free.

**Acceptance.** $\chi^2$ inside its 99% range, the discrepancy principle. No tuned tolerances.

**Stage 2 (absolute positions).** The HT step from 200 to 195 kV changes every lens strength by
the known ratio $r(V)$. The DAC values stay fixed, so lens positions become identifiable. Further lens
settings are chosen greedily (A-optimal) from the stage-2a fit, under the DAC limits and with the beam
kept on the camera.

Simulated results at a 0.01 px wobble:

| stage | result |
|---|---|
| stage 1, at a held-out set, for a 50 px kick | 0.024 px |
| stage 2b, detector position, typical anchors | 0.5–0.9 mm |
| stage 2b, detector position, good anchors | 0.2–0.4 mm |
| objective, either case | about 0.01 mm |

**Not yet checked:**
- convergence from the start on columns other than the one simulated;
- whether the fit's own error bars match the Monte Carlo scatter;
- the kick-purity terms from section 3.

## 8. The session

**Before stage 1 (operator):**
- Parallel TEM beam, 200 kV, at the 40k row.
- Free lens control on IL1–IL3 and PL1.
- Condenser aperture in. No objective aperture and no SA aperture.
- Specimen out, or the beam through a hole.
- Pivot points (shift/tilt purity) set. Beam centred.
- Find the diffraction setting by eye: the beam focused to a spot on the camera with the DAC values that give $A \approx 0$.

**Pilot (30–60 min):**
1. Beam-position wobble: repeat a ± pair 10 times at one set.
2. Lens return: leave a set and come back, and look for a centroid jump.
3. Auto-scale loop: a test kick, scaled to 50 px; check the beam stays on the camera.
4. Rotational symmetry: at one set, an x kick and a y kick. Expect $\Delta_y = i\,\Delta_x$.
   A difference means unequal or non-orthogonal deflector axes.
5. Kick-scale stability: a fixed reference kick at the start and end of the session (and of
   later sessions). This decides whether the anchors can be taken in a separate session.

**Stage 1 (no specimen, 200 kV), 36 frames:**
- 9 lens settings (8 plus D).
- At each: ± shift and ± tilt, auto-scaled to about 50 px.
- A dark frame, with the beam blanked, at the start.
- Log every DAC value, HT, command and frame.

**Anchors (cross grating with gold, same height):**
1. Image frame at the lowest-magnification set where the grating period is still at least about 4 px.
2. Ring pattern at the diffraction setting.
3. Optionally 3–5 validation frames: grating images at sets the fit has not seen. Compare the
   measured $|A|$ and rotation with the fitted model. This is the one test of model error that
   simulation cannot provide.

**Stage 2 (optional, specimen out):**
- **2a:** replay stage 1 at 195 kV (36 frames).
- **2b:** about 30 designed sets at both HTs (about 240 frames).
- Ask the staff about stepping the HT by 5 kV.

**Restore the microscope at the end:** free lens control off, CLA values, HT, blanking.

## 9. Open items

- Fit robustness: random columns, error-bar calibration, cross-talk and the shift/tilt relative
  phase.
- Allow the image anchor at any set, not only set 0.
- A gold-ring measurement (circle or ellipse fit) and a cross-grating period measurement, tested
  on TemGym-simulated frames.
- The session script in PyJEM calls, frames to centroids, the loader into the fit, and a dry run
  on the virtual NEOARM.
- Check the grating certificate: the period and its tolerance, and whether the decoration is pure Au or an Au–Pd or Pt–Pd alloy.

## 10. HT pair 60 → 80 kV (2026-11-05)

Working choice: stage 1 and the anchors at **60 kV**; stage 2 adds **80 kV** with the same DAC values.
The kick factor at 80 kV is taken as known, because it can be recalibrated at the new HT.
`projector_calibration.SimulatedMicroscope(v0=60e3)` and `Setup(v0=60e3, v1=80e3, ...)` set this up.

**How it is modelled:**
- At 60 kV, the console re-excites the objective and scales the free-lens DAC values by
  p(60)/p(200) = 0.515. The 40k row becomes IL1–PL1 ≈ [21534, 12814, 18319, 32985].
- This is used only to build the simulated microscope. On the microscope, read the DAC values with
  `GetFLCInfo`.
- PL1 at about 33 000 removes the 200 kV headroom problem.
- At 80 kV every lens weakens by p(60)/p(80) = 0.858, against 0.985 for 200 → 195 kV.
  The beam stays on the camera at all 9 sets.

**The diffraction setting at 60 kV.** λ = 4.87 pm, so the gold rings are about twice as far out as at 200 kV.
The diffraction setting is moved to a shorter camera length, IL2 = 21400 (DAC values
[21534, 21400, 14473, 32985]):

| quantity | value |
|---|---|
| \|B\| | 0.424 m |
| Au(111) ring radius | 80 px |
| Au(200) ring radius | 92 px |

The ring radius changes steeply with IL2 here. On the microscope, the diffraction setting is found by eye.

**Results** (8 runs per row; anchors: typical = grating 0.5% and rings 0.1%, good = 0.05% and
0.03%):

| wobble | anchors | stage 1 beam error (50 px kick) | absolute magnification | 2a: IL1 → detector | 2b: IL1 → PL1 | 2b: detector |
|---|---|---|---|---|---|---|
| 0.01 px | typical | 0.019 px | 0.29% | 2.4–15 mm | 1.0–1.4 mm | 0.8 mm |
| 0.01 px | good | 0.019 px | 0.04% | 2.7–16 mm | 0.15–0.23 mm | 0.27 mm |
| 0.03 px | typical | 0.056 px | 0.24% | 8–47 mm | 1.0–1.4 mm | 0.9 mm |
| 0.03 px | good | 0.056 px | 0.09% | 8–47 mm | 0.18–0.29 mm | 0.35 mm |

χ²/dof is 0.95–0.96 in every row. The objective is within 0.01 mm.

**Comparison with 200 → 195 kV.**
- Stage 1 is unchanged.
- At a 0.01 px wobble the final positions are the same, because the anchors limit them.
- The larger HT step makes stage 2 **robust to a worse beam-position wobble**. With good anchors at
  0.03 px, the error is 0.18–0.35 mm here against 0.36–0.71 mm at 200 → 195 kV.
