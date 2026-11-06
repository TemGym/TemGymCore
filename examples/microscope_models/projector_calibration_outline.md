# Projector calibration: the experiment and its mathematics

Status: 2026-11-06. Two stages:

1. **Stage 1, the predictor** (one HT, no specimen): where the beam lands on the camera for any lens
   setting and any beam shift or tilt, in deflector units. Session: `stage1_acquire.py`; analysis:
   `stage1_analysis.py`; rehearsal: `stage1_virtual_session.ipynb`.
2. **Stage 2, the physical calibration**: two specimen frames (a cross grating and gold) and the
   beam moves at a second HT, fitted together with stage 1. It gives real units and the lens and
   camera positions. Session: `stage2_acquire.py`; analysis: `stage2_analysis.py`; rehearsal:
   `stage2_virtual_session.ipynb`.

The model and fit are in `projector_calibration.py`. The rehearsals run on the virtual NEOARM
(TemGym rays for the beam and the grating, TemGym Gaussian beamlets for the gold rings) through its
PyJEM interface.

**Decision:** the **no-sample track**. Everything is measured from beam centroids in vacuum, with
raw deflector kicks. A specimen is used only for the two frames of stage 2, which turn deflector
units into metres and radians. The beam width and shape are never used.

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
\delta = \kappa_s\,r(V)\,c, \qquad \tau = \kappa_t\,r(V)\,c .
$$

- $\kappa_s$ and $\kappa_t$ are the kick scales, in m and rad per deflector unit, at the stage-1 HT.
- $r(V) = p_0/p$ is the magnetic-deflection scaling with electron momentum (1.0148 at 195 kV). It is
  taken as known: the deflector calibration can be re-measured at the new HT. (A fitted factor
  $\gamma(V)$ on top was tested earlier: it costs almost nothing, and a wrong fixed value shows in
  $\chi^2$, 0.3% off giving $\chi^2/\text{dof} = 31$.)

The measured data are

$$
m_{u,V,\text{shift}} = e^{i\varphi} A(u,V)\,\kappa_s\,r\,\Delta c, \qquad
m_{u,V,\text{tilt}} = e^{i\varphi} B(u,V)\,\kappa_t\,r\,\Delta c .
$$

**Kick purity.** With the pivot points aligned, CLA1 and CLA2 are a pure shift and a pure tilt.
If not, a shift also tilts and a tilt also shifts, and the two may act along rotated axes. The fit
has an optional model for this: each deflector a complex mix of shift and tilt, and an angle
between them. It is used only when it lowers $\chi^2$ by more than chance would; otherwise the
deflectors are taken as pure. A quick check at the console shows the same thing: in focus a pure
tilt does not move the spot, at the diffraction setting a pure shift does not.

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

The two numbers, and with a second HT the lens positions, are what stage 2 supplies.

## 5. Stage 1: the predictor

**Measured** (no specimen, one HT, about 100 frames): 12 visits.
- The 8 lens settings, as multiples of the 40k row.
- The diffraction setting (read in DIFF mode; $A = 0$).
- Setting 0 twice more (the beam-position wobble) and once with y kicks (round deflectors).

At each visit, a ± shift and a ± tilt, each auto-scaled to about 50 px.

**Fit.** Weighted least squares of the half-differences, with $\sigma = \text{wobble}/\sqrt2$ per
component (maximum likelihood for Gaussian centroid noise), and exact derivatives through the
TemGym model. Free: the projector parameters except the gauges, at fixed nominal kick scales (any
error in them is absorbed by the gauge). Start: lens gains from a closed-form regression on the
rotations, the camera rotation from the strongest shift, at $\varphi$ and $\varphi + \pi$. The
camera handedness (mirror) and the deflector y axis are each tried both ways; the one at the noise
level is kept.

**Acceptance.** $\chi^2$ inside its 99% range (the discrepancy principle), and a prediction check
at lens settings the session never visited.

**Rehearsal on the virtual NEOARM:** $\chi^2/\text{dof} = 0.6$; the mirrored camera and the pure
deflectors found by the fit; predictions at 8 unvisited settings within 0.006 px.

## 6. Stage 2: the physical calibration

Stage 1 leaves two numbers open: the kick scales $\kappa_s$ and $\kappa_t$ (equivalently the
true magnification and camera length), and with them the lens positions. Stage 2 supplies both,
and its two parts are useful only together:
- **specimen frames alone** give real units, but at one HT the lens positions are still not unique;
- **a second HT alone** separates the lens positions, but only up to the kick scales (about 2 mm of
  position per 1% of scale).

### 6a. The specimen frames

**The grating frame (|A|).** A cross grating of period $p_g$ images to fringes of period

$$
P = |A(u_1)|\,p_g / p_{px} \quad \text{pixels}.
$$

For coherent illumination this holds at any focus. Each diffraction order is a beam, and the
fringe period comes from their angular separation alone. The setting $u_1$ of this frame need not be in
focus; it needs enough periods in the frame.

The model predicts $A$ at every set, so this frame can be taken at **any** set. Choose the set by
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

**Working choice.** A grating with about 3 × 3 squares in the frame at the 40k row (set 0; period
about 230 nm, about 85 px), with the beam spread to fill the frame. The period comes from a fit of
the whole image: a square lattice times a smooth beam profile. Fitting the profile matters: square
centroids were biased by 0.5–9% by uneven illumination. On the virtual NEOARM the lattice fit is
within 0.1%. Repeat the frame at 2–3
stage positions to average, and measure, the cell-to-cell variation.

Cost of the grating uncertainty (8 runs each, 0.01 px wobble, rings 0.1%):

| grating period error | absolute magnification | IL1 → PL1 position error | detector | χ²/dof |
|---|---|---|---|---|
| 0.2% | 0.09% | 0.5–0.8 mm | 0.8 mm | 0.97 |
| 0.5% | 0.28% | 1.0–1.3 mm | 0.8 mm | 0.95 |
| 1% | 0.58% | 1.9–2.3 mm | 0.8 mm | 0.94 |
| 2% | 1.2% | 3.6–4.3 mm | 0.7 mm | 0.95 |

The beam-position error in stage 1 is 0.018 px throughout.

Above about 0.5%, the IL and PL position errors grow in proportion to the grating error, at about
2 mm per 1%. The detector position is set by the gold rings and does not change.

**$\chi^2$ cannot detect a wrong grating period.** It stays near 1 in every row, because the period is a
gauge choice that the beam data cannot contradict. The grating error therefore has to be measured
(repeat positions, certificate), not inferred from the fit.

**The gold frame (|B|).** At the diffraction setting ($A \approx 0$), a polycrystalline
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
can differ from Au by percent. That would make the ring measurement as uncertain as the composition.
Check what the decoration is. Ellipticity of the rings would show projector distortion, as a free check.

With a gold-shadowed cross grating, both frames come from **one sample at one stage position**.
Only the lens DAC values change between the two frames.

**The two scales.** The fit adds two residuals,

$$
\frac{|A(u_1)|\,p_g}{P\,p_{px}} - 1, \qquad \frac{|B(u_D)|\,\lambda}{R\,d\,p_{px}} - 1 ,
$$

weighted by their uncertainties. These fix $\alpha$ and $\beta$, and hence $\kappa_s$ and
$\kappa_t$. The frames are taken at the specimen height, so they also define which plane
"specimen" means.

**What limits the specimen frames:**
- The accuracy of the grating period. Replica gratings are nominal; check the certificate. The
  simulations assumed 0.2% (typical) or 0.05% (good).
- Image distortion over the field.
- Specimen tilt (foreshortening).
- Specimen height, which changes $A$ through the objective field.

These errors set the limit for absolute lens positions (section 7). Imaging the sample at more
settings does not reduce them, because the period error is common to all frames.


### 6b. The second HT

Specimen out. The lens DAC values stay fixed while the HT changes, so every lens strength changes by
the known ratio $r(V)$, and the lens positions become identifiable. One block:
1. at HT $V_1$: the 9 stage-1 settings and a fixed list of 30 more (a Halton sequence over
   0.1–2× the 40k row, within the DAC range);
2. back at $V_0$: the 30 again.

Each setting is measured as in stage 1; settings whose beam is off the camera are skipped live.

### 6c. The fit

Everything together: stage 1, stage 2 and the two specimen residuals

$$
\frac{|A(u_1)|\,p_g}{P\,p_{px}} - 1, \qquad \frac{|B(u_D)|\,\lambda}{R\,d\,p_{px}} - 1 ,
$$

weighted by their uncertainties (grating 0.2%, rings 0.1% by default). Free: all projector
parameters (the detector gap now too), the camera rotation and both kick scales. Start: the
stage-1 continuation, then the kick scales read off the specimen frames, then everything free.

**Outputs.** The lens and camera positions with 1σ uncertainties, and the deflector calibration:
the beam shift in m and the tilt in rad per deflector unit.

**Status: draft.** The target is 60 → 80 kV, which the virtual NEOARM does not cover yet (its hidden
optics have sources and lens tables for 200 and 195 kV only). A first trial of the scripts at
200 → 195 kV recovered the lens positions to 0.04 mm (1σ 0.6–2.0 mm, set by the 0.2% grating) and
the beam shift and tilt per deflector unit to 0.1%, at χ²/dof 0.2. After the HT change the beam
had to be re-condensed (CL3), which does not affect the beam centres.

## 7. Why not a sample at every setting

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

## 8. Open items

- The virtual NEOARM covers 200 and 195 kV only; 60 → 80 kV needs per-HT lens tables and sources.
- Fit robustness: random columns, and whether the fit's error bars match the scatter over repeats.
- Check the grating certificate: the period and its tolerance, and whether the decoration is pure Au
  or an Au–Pd or Pt–Pd alloy.
- Verify on the real NEOARM: the PyJEM names, the camera data type, lens settling, what DIFF mode
  changes besides IL1–IL3 and PL1, and how the HT is stepped.

## 9. HT pair 60 → 80 kV (2026-11-05, earlier simulator)

Working choice: stage 1 and the specimen frames at **60 kV**; stage 2 adds **80 kV** with the same DAC values.
The kick factor at 80 kV is taken as known, because it can be recalibrated at the new HT.
The results below came from the analytic simulator that was removed on 2026-11-06 (Git revision
`df4c7dc` retains it); they will be re-run on the virtual NEOARM once it covers 60 and 80 kV.

**How it is modelled:**
- At 60 kV, the console re-excites the objective and scales the free-lens DAC values by
  p(60)/p(200) = 0.515. The 40k row becomes IL1–PL1 ≈ [21534, 12814, 18319, 32985].
- On the microscope, read the DAC values with `GetFLCInfo`.
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

**Results** (8 runs per row; specimen frames: typical = grating 0.5% and rings 0.1%, good = 0.05% and
0.03%):

| wobble | specimen frames | stage 1 beam error (50 px kick) | absolute magnification | second HT, stage-1 settings only: IL1 → detector | with the 30 extra settings: IL1 → PL1 | with the 30: detector |
|---|---|---|---|---|---|---|
| 0.01 px | typical | 0.019 px | 0.29% | 2.4–15 mm | 1.0–1.4 mm | 0.8 mm |
| 0.01 px | good | 0.019 px | 0.04% | 2.7–16 mm | 0.15–0.23 mm | 0.27 mm |
| 0.03 px | typical | 0.056 px | 0.24% | 8–47 mm | 1.0–1.4 mm | 0.9 mm |
| 0.03 px | good | 0.056 px | 0.09% | 8–47 mm | 0.18–0.29 mm | 0.35 mm |

χ²/dof is 0.95–0.96 in every row. The objective is within 0.01 mm.

**Comparison with 200 → 195 kV.**
- Stage 1 is unchanged.
- At a 0.01 px wobble the final positions are the same, because the specimen frames limit them.
- The larger HT step makes stage 2 **robust to a worse beam-position wobble**. With good specimen frames at
  0.03 px, the error is 0.18–0.35 mm here against 0.36–0.71 mm at 200 → 195 kV.
