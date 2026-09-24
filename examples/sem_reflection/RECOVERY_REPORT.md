The SEM reflection recovery estimated nine source, lens, and camera parameters from 129 synthetic diffraction images. We fitted 111 images and predicted 18 images acquired at withheld lens settings, using a local Gaussian reflection model for three Cu(001) diffraction orders. The recovered objective working distance was 40.154 ± 0.275 mm, compared with a generating value of 40 mm. All uncertainties below are local one-standard-deviation estimates conditional on the fixed calibrations and model assumptions.

This report describes the saved experiment in [sem_reflection_calibration_pipeline.ipynb](sem_reflection_calibration_pipeline.ipynb), with numerical results from [fit.json](output/calibration_gaussian/fit.json) and [checks.json](output/calibration_gaussian/checks.json). It is a reconstruction of the saved run; the acquisition and optimization were not rerun for this report.

The unknown parameter vector was

$$
\mathbf p=(z_s,z_1,z_2,z_o,f_{1,0},f_{2,0},f_{o,0},\sigma_s,L_0).
$$

The four axial positions refer to the source, first condenser, second condenser, and objective, with the specimen at column coordinate z = 0. The three focal lengths are reference values at unit lens-power command; the remaining unknowns are the source RMS intensity width in either transverse coordinate and the initial camera distance. This gives five geometry parameters, three focal lengths, and one source-width parameter. Working distance is derived as −z_o, so it adds no extra unknown. These column coordinates are distinct from laboratory Z, along which the camera is translated.

| Unknown | Generating value | Recovered value ± local standard error |
|---|---:|---:|
| Source position z_s (mm) | −350.000 | −347.80 ± 9.82 |
| First condenser position z_1 (mm) | −270.000 | −269.27 ± 5.83 |
| Second condenser position z_2 (mm) | −130.000 | −130.55 ± 1.56 |
| Objective position z_o (mm) | −40.000 | −40.154 ± 0.275 |
| First condenser reference focal length f_1,0 (mm) | 40.000 | 39.39 ± 1.76 |
| Second condenser reference focal length f_2,0 (mm) | 25.7143 | 25.694 ± 0.522 |
| Objective reference focal length f_o,0 (mm) | 21.1765 | 21.279 ± 0.205 |
| Source RMS width σ_s (nm) | 1.9100 | 1.8889 ± 0.0509 |
| Initial camera distance L_0 (mm) | 120.000000 | 119.999880 ± 0.000108 |

Neither the 40 mm generating working distance nor the 120 mm generating camera distance was supplied to the fitter. Broad hard bounds enforced component ordering and plausible ranges; for example, the objective lay between −75 and −15 mm and camera distance between 80 and 180 mm. The source was assumed to be a round, coherent, diffraction-limited Gaussian waist, so its angular spread followed from its fitted width and the known wavelength; source curvature, emittance, and partial coherence were not additional fitted variables.

The number of “knowns” has two useful meanings. Each image had four recorded scalar controls, comprising three relative lens powers and one camera translation. Each image supplied twelve measured scalar observables, comprising two centroids and two variances for each of three diffraction orders. Fixed calibrations and structural assumptions were additional inputs, as listed below; they do not have a unique scalar count because quantities such as pose and source model are structured assumptions.

| Fixed input or assumption | Value or specification |
|---|---|
| Electron energy | 30 keV |
| Surface lattice | Ideal Cu(001), primitive square period a/√2 = 0.255619 nm, with a = 0.3615 nm |
| Incidence and column inclination | Both 2°; on-axis beam |
| Specimen reference and illumination | Specimen at z = 0; known surface hit and surface orientation |
| Camera pose | Known orientation; initial centre on the specular ray; fixed orientation and laboratory height during translation |
| Detector | 256 × 256 pixels, 55 µm pitch |
| Detector point-spread function | Isotropic Gaussian, RMS width 0.6 pixels |
| Noise | Poisson counts, mean background 0.02 counts/pixel, Gaussian read noise RMS 0.2 counts/pixel |
| Lens control law | 1/f_i = u_i/f_i,0, with recorded relative powers u_i |
| Source and column structure | Round coherent Gaussian waist; three ideal thin round lenses; zero steering and objective aberrations |
| Retained diffraction orders | (0,m), m = −1, 0, +1 |
| Surface response used for image generation | Cu scattering factors; attractive potential step with 21.2 V inner potential and zero absorption |
| Simulated incident dose | 10^9 electrons per image; total order intensities were excluded from the geometry objective |

The acquisition used 43 distinct lens settings, each imaged at laboratory camera translations ΔZ = −25, 0, and +25 mm. The fit set contained one baseline setting, twelve single-lens perturbations (−20%, −10%, +10%, and +20% for each of three lenses), and twenty-four joint settings with each power sampled between 0.8 and 1.2. Six further joint settings were withheld from both initialization and refinement.

| Data quantity | Fit | Held-out prediction | Total |
|---|---:|---:|---:|
| Lens settings | 37 | 6 | 43 |
| Images | 111 | 18 | 129 |
| Measured order spots | 333 | 54 | 387 |
| Scalar moment observations | 1,332 | 216 | 1,548 |

The saved image array has shape (129, 256, 256), and the observation array has shape (129, 3, 4). The fit therefore had 1,332 residual components and nine fitted parameters, giving 1,323 nominal residual degrees of freedom. The four statistics for each order had a full 4 × 4 measurement covariance, so they were whitened jointly. The image count was the chosen acquisition design, rather than an established minimum for recovery; dividing nine unknowns by twelve observables per image would not establish identifiability.

The optical propagation can be written using ABCD matrices. For one transverse coordinate and its paraxial slope, a drift and thin lens are

$$
P(d)=\begin{pmatrix}1&d\\0&1\end{pmatrix},
\qquad
F_i(u_i)=\begin{pmatrix}1&0\\-u_i/f_{i,0}&1\end{pmatrix}.
$$

The source-to-specimen matrix is

$$
M_{s\to S}(\mathbf u)=
P(-z_o)F_o(u_o)P(z_o-z_2)F_2(u_2)
P(z_2-z_1)F_1(u_1)P(z_1-z_s)
=\begin{pmatrix}A&B\\C&D\end{pmatrix},
$$

with the rightmost element acting first. The same matrix applies to both transverse directions before reflection because the column contains round, centred lenses. Each elementary matrix has unit determinant. The ABCD entries are calculated from the nine physical parameters and recorded commands; they are not four further fitted parameters for every image.

For baseline powers u_1 = u_2 = u_o = 1, using millimetres for position and dimensionless slopes, direct multiplication gives

$$
M_{\mathrm{generating}}\simeq
\begin{pmatrix}
-0.666667&0\\-0.116667&-1.500000
\end{pmatrix},
\qquad
M_{\mathrm{recovered}}\simeq
\begin{pmatrix}
-0.675616&-0.0180462\\-0.117819&-1.483277
\end{pmatrix}.
$$

Here B has units of mm and C has units of mm⁻¹. The unrounded determinants are unity to floating-point precision. The generating baseline has B ≈ 0, corresponding to source-to-specimen imaging, while the lens sweeps explore defocused configurations.

For two transverse dimensions, the code propagates the complex inverse-curvature matrix H = Q_inv. If the 4 × 4 ray matrix has 2 × 2 blocks A, B, C, and D, then

$$
H'=(C+DH)(A+BH)^{-1},
\qquad
H_s=\frac{i}{2k\sigma_s^2}I,
\qquad
\Sigma=[2k\,\operatorname{Im}H]^{-1},
$$

where k = 2π/λ and Σ is the intensity covariance in physical transverse coordinates. This combines diffraction and geometrical focusing within the Gaussian model. The detector PSF adds σ_PSF² I to the covariance in pixel units; predicted centroids and variances are then calculated from pixel-integrated probabilities over the same strips used to measure the images.

At the specimen, the periodic sample produces discrete outgoing orders. The generic BraggSample component represents a thin periodic field multiplier

$$
t(\mathbf r)=\sum_m c_m\exp[i\mathbf G_m\cdot(\mathbf r-\mathbf r_0)],
$$

so each incident Gaussian generates an order with coefficient c_m and tangential wavevector change G_m. For the retained Cu row, |G_m| = 2π|m|/a_s. The ordinary thin-grating component changes transverse momentum in a forward or folded frame; physical reflection is supplied by the three-dimensional surface adapter. The calibration implements the corresponding reduced order geometry and flux functions directly.

In the laboratory coordinates used by the adapter, the incident chief direction is (sin α, 0, cos α). Defining η_m = mλ/a_s, the reflected chief direction is

$$
\hat{\mathbf d}_m=
\left(-\sqrt{\sin^2\alpha-\eta_m^2},\;\eta_m,\;\cos\alpha\right).
$$

Tangential momentum follows the reciprocal lattice, and the outgoing normal component is selected to preserve the wavevector magnitude and leave the surface. Retained orders must propagate and clear the 0.25° exit-angle cutoff. The Gaussian envelope is mapped through the surface intersection and outgoing-plane projection using an order-dependent Jacobian J_m frozen at the chief ray,

$$
H_{m,\mathrm{out}}=J_m^{-T}H_{\mathrm{in}}J_m^{-1}.
$$

Each order then undergoes free-space propagation to the detector. For the known laboratory translation, the propagation distance in the outgoing frame and the detector-centred mean are

$$
\ell=L_0+\Delta Z\cos\alpha,
\qquad
\boldsymbol\mu_m=
\frac{1}{p_{\mathrm{pixel}}}
\begin{pmatrix}
\ell s_{m,u}-\Delta Z\sin\alpha\\
\ell s_{m,v}
\end{pmatrix},
$$

where s_m denotes the chief-ray slopes in the outgoing frame. The camera's known motion changes both its axial distance and its transverse offset, since the camera does not follow the specular ray vertically.

Order amplitudes use projected neutral-Cu Lobato–Van Dyck scattering factors and a complex potential-step reflection amplitude r(α,E). The coefficient is c_m = w_m r, with weights normalized against the fixed reference row m = −256,…,+256. Thus the retained intensity is proportional to |w_m|²|r|², and omitted orders or detector losses are not renormalized away. Equivalently, a nonzero coefficient has complex action ΔS_m = −i log(c_m)/k. The saved expected integrated counts before detector losses were approximately 2.242 million, 6.433 million, and 2.242 million for the three retained orders.

For recovery, the response and envelope map were fixed at the chief ray. The detector image was the sum of three separated Gaussian order intensities, integrated over pixels and blurred by the calibrated PSF. The saved overlap bound was 3.89 × 10⁻¹⁶, supporting this separated-order approximation for the recorded settings. In the full angular-packet model, overlapping coherent packets require summation of complex fields.

The fit followed saved noisy pixels → measured moments → independent initialization → covariance-weighted least squares → held-out prediction. Peak finding located the three orders, and each order supplied its centroid and marginal variance along both detector axes after background subtraction. Total intensity normalized those statistics but was excluded as a fitted observable.

Camera distance was initialized from order separations, using the known lattice period, energy, detector pitch, and camera translations. Ten bounded starting parameter vectors then explored the source/lens geometry using a coarse shape objective, and the three best were refined against the original moments with their full measurement covariance. JAX supplied parameter derivatives to the bounded least-squares solver. The optional generating-geometry file was separate from the recorded inputs and was removed during the notebook's fit; it was restored for comparison afterwards.

Known diffraction angles give the order separation an absolute distance scale, which explains the tight camera-distance estimate. Lens-power perturbations change the dependence of the spot widths on lens positions and focal lengths, while the three detector planes sample outgoing expansion and curvature. Those effects collectively constrain the source and column, although the fitted source position remains weak and correlated with other geometric parameters.

The saved χ² was 1366.16 for 1323 degrees of freedom, giving reduced χ² ≈ 1.033. RMS whitened moment residuals were 1.013 for fitting images and 1.011 for held-out images. The scaled residual Jacobian had local rank nine, all three refinements reported success, and no fitted parameter lay within 1% of a hard bound. The maximum lateral strip-tail fraction was 1.04 × 10⁻¹⁶. The reported maximum pixel-quadrature discrepancy was 1.52 × 10⁻¹⁰ noise units on the three generation frames checked at increased quadrature.

These results support parameter recovery and prediction within the stated synthetic model. The 9.82 mm local uncertainty on source position illustrates that full local rank does not imply uniformly precise geometry, and multiple starts do not prove global uniqueness. The roughly 0.108 µm formal uncertainty on initial camera distance conditions on exact lattice spacing, energy, pitch, incidence, and camera pose; their calibration errors were not propagated into that number.

The separate [interactive reflection notebook](sem_reflection.ipynb) defaults to 4° incidence, a 2° mechanical column inclination, steering deflectors, objective aberrations, and thirteen nominal Cu orders m = −6,…,+6, subject to propagation and clipping. The saved recovery instead used 2° incidence, zero steering, ideal lenses, and three local Gaussian orders. It tested a reduced predictor against images generated with that same reduced model. Full angular-packet validation, real SEM measurements, dynamical surface diffraction, partial coherence, and uncertainty in steering or lens-current calibration remain outside this saved recovery.

Implementation references are [reflection_calibration.py](../../src/temgym_core/reflection_calibration.py), [components.py](../../src/temgym_core/components.py), [gaussian.py](../../src/temgym_core/gaussian.py), and [scattering.py](../../src/temgym_core/scattering.py). The saved [experiment specification](output/calibration_gaussian/experiment.json), [measurement record](output/calibration_gaussian/measurement_record.json), and [generation checks](output/calibration_gaussian/generation_checks.json) record the fixed inputs and numerical checks. Existing regression tests are in [test_reflection_calibration.py](../../tests/test_reflection_calibration.py).
