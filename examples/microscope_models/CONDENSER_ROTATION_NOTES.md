# Condenser rotation and aperture-focus experiment

**Current full-microscope workflow:** [one two-voltage notebook](microscope_calibration_pipeline.ipynb). The results and standalone procedures below are historical.

These notes record an earlier version of the condenser experiment. It makes synthetic measurements, analyses their pixels and counts, fits a separate microscope from earlier data-derived guesses, and checks predictions before comparing with hidden truth.

**Rotation does not remove the fixed-objective ambiguity.** Adding objective-current changes gives a well-defined local minimum for axial geometry in this model. Optimization is still fragile, and the lateral aperture centre remains uncertain. This is a simulation check, not a demonstration that every parameter of a real microscope is uniquely recoverable.

## What was added

- **24 rotation images:** temporarily prepared elliptical illumination, with no specimen, imaged through the calibrated projector. Each image is fitted independently for centre, two widths, orientation, dose and background. Only its measured orientation and marginal uncertainty enter the microscope fit. MAGICAL remains the specimen for the other imaging measurements.
- **492 current readings:** 123 condenser settings, each with three aperture readings and an open-aperture reference. CL1 sweeps through focus at three CL3 commands, 0.75, 1.0 and 1.3. Aperture diameters remain 4, 10 and 20 µm. Unlike the earlier monotonic scans, these curves fall on both sides of their high-transmission region.
- **Joint rotation fitting:** the three lens gains enter both rotation and focusing. One extra unknown absorbs the starting ellipse orientation and the constant pre-field rotation.
- **Profiles:** fix pre-field thickness and refit every other parameter; separately fix the direction of the aperture centre and refit its radius and all other parameters.

The simulation locates focus to choose acquisition settings. In an experiment, a preliminary current scan would choose the interval. The fitted crossover positions are not supplied to inference. Saturated transmission near the centre does not identify one exact focal command by itself.

The [paper](https://arxiv.org/html/2607.29411v1), section II.2.2, describes measuring an ellipse created by upstream stigmation; its preferred alternative registers defocused lacy-carbon Ronchigrams. We use the ellipse idea. This is not a reproduction of the paper's complete microscope calibration.

## Matrix and rotation

Using the requested ray order,

$$
\mathbf r=(x,y,\theta_x,\theta_y,1)^T,
\qquad \mathbf r_{\rm out}=M\mathbf r_{\rm in},
$$

$$
M=\begin{pmatrix}
 A_{xx}&A_{xy}&B_{xx}&B_{xy}&x_0\\
 A_{yx}&A_{yy}&B_{yx}&B_{yy}&y_0\\
 C_{xx}&C_{xy}&D_{xx}&D_{xy}&\theta_{x,0}\\
 C_{yx}&C_{yy}&D_{yx}&D_{yy}&\theta_{y,0}\\
 0&0&0&0&1
\end{pmatrix}.
$$

This is the general affine form. The experiment uses centred round lenses; it fits the source mean position and angle, and the aperture centre. It does not introduce separate decentring parameters for every lens.

For one finite solenoid of length $L$ and signed strength $q$,

$$
A=aR(-q),\quad B=bR(-q),\quad C=cR(-q),\quad D=dR(-q),
$$

$$
a=d=\cos q,\qquad b=L\frac{\sin q}{q},\qquad
c=-\frac{q\sin q}{L}.
$$

Here $R$ is the ordinary 2D rotation matrix. The controlled strengths satisfy $q_j=g_j u_j$. Thus rotation is linear in recorded relative current, while the effective focal power is

$$
\frac{1}{f}=-c=\frac{q\sin q}{L}\simeq\frac{q^2}{L}.
$$

It is **inverse focal length**, rather than focal length, that is approximately quadratic in current. The code uses the exact finite-solenoid expression.

For the rotation images the objective is fixed. After subtracting the calibrated projector rotation,

$$
\alpha_i=\beta-g_1u_{1,i}-g_3u_{3,i}-g_mu_{m,i}\pmod{\pi}.
$$

The shared intercept $\beta$ is fitted. An ellipse has no arrow, hence the period is $\pi$, not $2\pi$. The angle residual uses the wrapped difference divided by the measured angle uncertainty. Current ratios use their full covariance, including correlation from the shared open-aperture reference. These residuals are appended to the independently measured specimen-image residuals. There are no soft priors; positivity, ordering, the selected lens-polarity branch and engineering bounds still restrict the model.

The source ellipse is deliberately simple: its position and angular covariance share the same unknown principal directions, with no position-angle covariance at the source waist. Both covariance blocks are proportional to the same elliptical shape matrix. A round lens system then scales and rotates that ellipse without changing its principal directions in the rotating frame. We fit widths separately in every image; their generating values are not supplied to inference.

This requires a stable orientation marker **upstream of CL1**. A marker introduced farther down the column measures only the lenses after it. A real astigmatic beam whose axes change with focus may need a richer model before its measured ellipse angle can be treated as pure Larmor rotation.

## What the experiment found

The separate angle regression recovered gains 0.20047, 0.37150 and 0.40002, against generating values 0.20, 0.37 and 0.40. The joint fit uses the measured angles, not those generating values.

| Measurement set | Locally constrained directions | Calibration RMS | Converged starts |
|---|---:|---:|---:|
| Fixed objective, including rotation and focus sweeps | 22 / 23 | 0.982 | 0 / 4 |
| Also include objective changes | 26 / 26 | 1.032 | 2 / 4 |

RMS is measured in estimated noise units. The fixed-objective fits reach a small residual but exhaust their evaluation budget. They are numerical representatives of an ambiguous family, not accepted unique geometry estimates. For their numerical solution only, the pre-field strength is held at a different value in each start; the rank calculation includes that unknown.

For the objective-changing fit, two starts agree at the good solution. Two other starts exhaust 5,000 evaluations with RMS about 1,522 and are clearly rejected by the data. These are not equally good physical solutions. Starts use the earlier fitted objective model with 0.5% perturbations in log geometry/strength; this is not a blind global search. An earlier initialization based on replacing gains with the separate rotation regression also failed; its completed attempt is retained in [the initialization audit](calibration_results/condenser_rotation_initialization_audit.json).

At fixed objective current, an exact transformation changes pre-field thickness, nearby gaps and the unknown source/aperture orientation. It preserves the measured rotations because the common angle intercept is unknown. Tested pre-field thicknesses **0.508–1.665 mm** change the complete residual vector by less than **7 × 10⁻¹⁰ noise units**. More CL1/CL3/mini settings of these same types cannot remove that exact ambiguity.

With objective changes, the best pre-field thickness is **0.99972 mm**, against a hidden 1.00000 mm. Refitting all other parameters gives:

| Fixed pre-field thickness (mm) | Extra chi-squared |
|---:|---:|
| 0.670 | 2081 |
| 0.819 | 663 |
| 0.993 | 0.99 |
| 1.000 | 0 |
| 1.007 | 1.01 |
| 1.221 | 1143 |
| 1.491 | 6225 |

All nine thickness-profile fits converged. The local standard error is **0.00681 mm**. This is evidence against the old thickness ambiguity in the sampled region, not proof that there is no disconnected minimum elsewhere.

The source-waist position is **−324.988 mm**, against −325.000 mm. Its local standard error is **0.0804 mm**. Fitted axial lens positions are close to their generating values; the largest lens-thickness error in this noise realization is about **0.046%**. Those small realized errors should not be substituted for the uncertainty estimates.

### Which measurements helped?

The following comparison uses the same fitted geometry for every Jacobian. It isolates added information; these rows are not four separately optimized microscopes.

| Measurements | Source-position standard error (mm) | Pre-field thickness standard error (mm) |
|---|---:|---:|
| Previous images and currents, including objective changes | 1.315 | 0.006873 |
| Previous + rotation | 1.314 | 0.006867 |
| Previous + aperture-focus sweeps | 0.08198 | 0.006811 |
| All measurements | 0.08041 | 0.006809 |

**The focus sweeps supply nearly all the improvement here.** Rotation is measured successfully, but the other data already constrain the rotation gains strongly. These measurements do not materially improve pre-field thickness precision.

### Aperture centre and new images

The fitted aperture centre is approximately $(0.038,-0.246)$ µm, against $(0.200,-0.150)$ µm. The small local covariance ellipse gives a misleading impression of its nonlinear uncertainty. Profiling the centre direction while refitting its radius and all remaining parameters finds a much broader range of plausible positions along a curved path. Directions −90°, −67.5° and −45° all fall within 3.84 chi-squared units of the minimum near −81°. This finite grid is not an exact confidence interval. The radius is better constrained than the direction. All 17 sampled direction-profile fits converged.

The model predicts **30 new images at 97% and 103% objective current**, without refitting its geometry or response. Pixel RMS is **0.983**. Independently measured pattern RMS is **1.243**, or **1.133** after including the local fitted-parameter covariance and correlations across states. Exposure/background are taken from independent image analysis. Good predictions are useful evidence, but do not prove parameter uniqueness.

## Numerical and physical limits

The aperture calculation uses the output power of actual TemGym Gaussian beamlets after `SigmoidAperture`, including their changed width. Adaptive radial integration resolves narrow crossovers. Refining both integration and the tabulated power curve changes transmission by at most **8.2 × 10⁻⁹** over the new acquisition settings. The 0.5 nm beamlet waist is a numerical representation choice, not a known microscope source size. Tests check ellipse extraction without lens truth, rotation against full covariance propagation, crossover integration, its derivative and the exact remaining symmetry.

The source position is the model's **effective waist plane**, not an independently located emitter surface. Lens positions and lengths are physical coordinates of an ideal finite-solenoid model, not automatically the dimensions of a real pole piece. The simulation uses the same lens family for generation and fitting. The reference post-field/projector calibration is treated as known, while three response slopes away from reference are fitted. Reference-calibration uncertainty, changing source conditions, field-profile mismatch, current hysteresis and a moving ellipse orientation are not included in the reported errors.

The complete pipeline passed **17 targeted tests**. The notebook reports convergence failures and profiles alongside the good image predictions. A fit should not be accepted merely because the optimizer returned, or because one image looks right.
