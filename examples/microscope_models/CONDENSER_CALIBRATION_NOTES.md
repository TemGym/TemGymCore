# Condenser calibration from images

**Current full-microscope workflow:** [one two-voltage notebook](microscope_calibration_pipeline.ipynb). The results and standalone procedures below are historical.

These notes describe the original image-only baseline. That baseline was extended with aperture currents; see [the joint experiment notes](CONDENSER_APERTURE_NOTES.md) for the historical aperture-only results.

The original image-only procedure was:

1. Generate noisy MAGICAL specimen images with a hidden condenser and source.
2. Measure the specimen beam independently at each condenser setting.
3. Fit a new condenser to those measurements, starting from engineering guesses.
4. Predict separately generated images at new settings and then open the truth file for scoring.

The projector is treated as exactly calibrated. CL1, CL3 and the mini can change; the objective excitation stays fixed. There is one independent effective objective pre-field in this model. It is not constrained to share a field profile with the calibrated post-field.

## What is measured

There are 24 condenser settings and five calibrated projector views per setting: one specimen-focused view and four defocused views. A further six condenser settings provide 30 prediction images. The controls are recorded in `condenser_measurement_protocol.json`. They are normalized commands, not measured amperes. Simulation setup tunes the mini to keep the illumination visible; inference receives the recorded commands, not the tuning calculation or true beam parameters.

The five matrices in `condenser_projector_views.json` come from the previously fitted projector model. Only the four downstream projector controls change between views. The objective control is one in every view. These are attainable states of the numerical model; hardware limits and real projector calibration errors have not been tested.

Use the ray vector and matrix

$$
\mathbf u=(x,y,\theta_x,\theta_y,1)^T,\qquad
T=\begin{pmatrix}
A_{xx}&A_{xy}&B_{xx}&B_{xy}&o_x\\
A_{yx}&A_{yy}&B_{yx}&B_{yy}&o_y\\
C_{xx}&C_{xy}&D_{xx}&D_{xy}&o_{\theta x}\\
C_{yx}&C_{yy}&D_{yx}&D_{yy}&o_{\theta y}\\
0&0&0&0&1
\end{pmatrix}.
$$

For a calibrated projector view, the detected position is
$\mathbf r_d=A\mathbf r+B\boldsymbol\theta+\mathbf o_r$.
The specimen beam is described by its mean position, mean angle and round covariance:

$$
\Sigma=\begin{pmatrix}s^2 I_2 & h s^2 I_2\\h s^2 I_2 &(h^2s^2+t^2)I_2\end{pmatrix}.
$$

Here $s$ is the spatial RMS in each coordinate, $h$ is the position–angle slope, and $t$ is the angular RMS at a fixed specimen position. These are seven independent quantities including the four means. Conditional on position,

$$
\boldsymbol\theta=\overline{\boldsymbol\theta}
+h(\mathbf r-\overline{\mathbf r})+\boldsymbol\epsilon,
\qquad \operatorname{Cov}(\boldsymbol\epsilon)=t^2I_2.
$$

Thus defocused images contain an effective position map $A+hB$, a displacement $B(\overline{\boldsymbol\theta}-h\overline{\mathbf r})$, and angular blur with covariance $t^2BB^T$. The known MAGICAL pattern helps distinguish the signed map from its blur. A circular illumination alone would lose orientation information. We fit all five images jointly for these seven beam quantities, with separate exposure and background per image. The full marginal covariance is retained, including correlations.

This is a paraxial intensity-transport experiment, not a diffraction calculation. It assumes a known positive specimen transmission pattern, round illumination and lenses, a known detector PSF and coordinate origin, and no unknown drift between views. Unknown free shifts per frame would remove information about the beam mean angle. There are no clipping apertures or coherent contrast effects. The 64 × 64 detector uses a 220 µm effective pixel pitch, corresponding to twofold binning of the earlier 110 µm detector.

## Lens and source model

Each finite solenoid uses $q=KL$ and radial matrix

$$
H(q,L)=\begin{pmatrix}\cos q & L\sin q/q\\-q\sin q/L & \cos q\end{pmatrix},
$$

with an additional transverse rotation $R(-q)$. In grouped coordinates every scalar radial entry multiplies that same 2 × 2 rotation. The three controlled strengths are $q_i=g_i u_i$; the static pre-field has a negative signed strength. Signs and the low-strength branch are part of this model choice. The angular spread is propagated; parallel illumination is not assumed.

The source is defined at an effective round waist: its position–angle covariance is zero there. Its axial position, transverse mean, mean angle, size and angular spread are unknown. This definition is needed to give a virtual source position a definite meaning. It does not locate the physical emitter. There are 19 unknown numbers: five gaps, four lens lengths, four strengths, four source means and two source widths. Lens centres and source z are calculated from the gaps and lengths, with the specimen at z = 0. Lens offsets are zero; source means supply the fitted beam offsets.

Starting lengths are 5, 5, 3 and 1 mm. The pre-field starts near −3 mm, with dimensionless strength −0.45; the generating example uses −2.5 mm, 1 mm and −0.48. These are **engineering guesses**, not measured NEOARM dimensions. Each fitted gap/length is bounded to 0.4–2.5 times its starting guess, and each strength magnitude to 1/1.8–1.8 times its guess. Source RMS size is bounded to 1–80 nm and angular RMS to 10–300 µrad. Source mean coordinates are solved without bounds. There are no soft penalties pulling the answer towards the guesses.

The radial lower-left entry gives $1/f=K\sin(KL)$ for the equivalent external focal length, rather than the thin-lens approximation $K^2L$. A hard-edge solenoid is a compact first-order model. A real specimen lies within the objective field, so a separate complete hard-edge lens plus drift is only an effective representation of the pre-field. It does not reproduce the actual immersed field profile.

For scale, the generating lenses at unit command have the following values. At 200 kV, the field follows $B_z=2p_e q/(|e|L)$, where $p_e$ is the relativistic electron momentum. These are consequences of the chosen ideal model, not measurements of the real lenses.

| Lens | Effective length | External focal length | Uniform field magnitude |
|---|---:|---:|---:|
| CL1 | 5 mm | 126 mm | 0.132 T |
| CL3 | 5 mm | 37.4 mm | 0.244 T |
| Mini | 3 mm | 19.3 mm | 0.440 T |
| Objective pre-field | 1 mm | 4.51 mm | 1.58 T |

## Why two freedoms remain

The ambiguity is exact, not a failed optimizer. Let the static section after the mini be
$F=D(g_4)H(q,L)D(g_3)$, with radial entries $a,b,c,d$. Then

$$
c=-q\sin q/L,\quad a=\cos q+g_4c,\quad d=\cos q+g_3c.
$$

For any nearby positive scale $\lambda$ and negative strength $q'$, construct

$$
c'=c/\sqrt\lambda,\quad L'=-q'\sin q'/c',
$$
$$
g_4'=(a/\sqrt\lambda-\cos q')/c',\qquad
g_3'=(d\sqrt\lambda-\cos q')/c'.
$$

Scale the first three gaps and first three lens lengths by $\lambda$. Multiply source positions and spatial RMS by $\sqrt\lambda$, divide source angles and angular RMS by $\sqrt\lambda$, and rotate both source mean vectors by $q'-q$. The controlled strength gains do not change. Provided the resulting lengths/gaps are positive, **every specimen mean and covariance remains identical at every condenser control setting**. Therefore every downstream image in this experiment also remains identical.

For example, the generating source can move from −325 mm to about −422 mm with different lengths and pre-field strength, without changing any predicted specimen beam. More repetitions or more downstream projector views cannot remove these freedoms. Bounds restrict the permitted family but do not turn its members into measured geometry.

For numerical fitting we choose one member by keeping the first source-to-CL1 entrance gap and static pre-field strength at their starting values. These are explicitly **coordinate choices**, not independently known quantities. The other 17 variables are fitted. Equivalent family members then release both choices. Source means and variances enter the second moments linearly, so they are solved by bounded linear least squares inside each nonlinear optical iteration. This improves the numerical problem without adding information.

The Jacobian of ideal beam observations has rank 17 of 19. A mathematical test adding a known 10 µrad input angle change at the **effective source plane** has local rank 19. This is a proposed extra measurement, not an axis used by this notebook. It has not yet been tested with noisy images, and rank does not guarantee small uncertainty or a unique global minimum. A calibrated tilt at the specimen is not a substitute for that upstream input constraint. Two suitable independent geometric measurements could also remove the two freedoms; not every pair of dimensions would do so.

## Result and limits

The saved run uses noise seeds 1901 (calibration) and 1902 (predictions). The fitted model has calibration measurement RMS 0.912 and prediction RMS 0.995. These residuals are normalized by the full image-derived measurement covariance; prediction RMS does not include uncertainty in the fitted model. The three controlled strength gains have a maximum error of about 0.033% in this synthetic run.

Five of twelve starts meet the optimizer stopping criteria; seven exhaust the 5,000-evaluation budget. All reach similar noise-level residuals. This is evidence for predictive agreement, **not a completed census of minima or robust recovery of every lens dimension**. Even after aligning the two exact freedoms with truth for scoring, finite-noise geometry errors remain appreciable. The fit has an additional weak direction. No single local error bar is reported for an unidentifiable physical parameter.

The notebook keeps generating truth out of image analysis and condenser fitting. Cache signatures cover the inputs and local model code; set `reuse_completed=False` to rerun. Raw generated arrays are ignored by git. The notebook outputs and `calibration_results/condenser_calibration_summary.json` preserve the reviewable result.

## Relation to the references

Pfeifer, Lupini and Hoglund, *A versatile generalized digital twin for Electron Microscopy*, supplied [arXiv manuscript](../lens_inversion/paper/2607.29411v1.pdf), pp. 3–5, use lens sweeps and crossover conditions. Their condenser calibration also uses aperture transmission. We borrow the finite-lens model and the idea of combining lens settings; we do not claim to reproduce their measurement procedure or have their internal aperture constraints. Their effective gun position is also an optical source position. Their discussion initially assumes a lens length, then uses further rotation information to adjust it.

JEOL describes the [condenser-objective pre-field and mini lens](https://www.jeol.com/words/emterms/20121023.034459.php) and the [specimen inside the objective field](https://www.jeol.com/words/emterms/20121023.060558.php). These support the optical roles, not the numerical thickness guesses. [EPFL's TEM ray diagrams](https://www.epfl.ch/labs/lsme/tem-ray-diagrams/) likewise distinguish the effective objective pre-field and post-field. No verified NEOARM pre-field field profile or effective thickness is supplied here.
