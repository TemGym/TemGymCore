# Joint condenser and aperture experiment

**Current full-microscope workflow:** [one two-voltage notebook](microscope_calibration_pipeline.ipynb). The results and standalone procedures below are historical.

These notes record an earlier version of the condenser experiment. It combines the existing MAGICAL image measurements with aperture-current scans. The result is **not unique**: an exact family of physically different models has identical predictions.

## Measurements and unknowns

- The projector is calibrated; specimen transmission, voltage, detector response and coordinates are known.
- 120 open-aperture MAGICAL images provide beam measurements at 24 condenser settings. Thirty separate images at six new settings check predictions.
- Known aperture diameters are **4, 10 and 20 µm**. These are synthetic choices, not NEOARM specifications. The sigmoid amplitude edge has fixed width 0.15 µm and sharpness one.
- A 5 × 5 grid of CL1/CL3 commands produces three transmitted-current readings and one open-aperture reference per setting: **100 calibration readings**. Nine new settings provide **36 prediction readings**.
- Each reading is a Poisson electron count. The source current varies between settings; the three aperture readings and reference share the same incident current within a setting. An exposure has about 20 million incident electrons. The detector collects the full transmitted current; other losses and dark current are excluded.

The fit has **22 free parameters**: five gaps, four finite lens lengths, four strength gains, four source mean coordinates, two source widths, and three aperture coordinates. Source z is derived from the gaps and lengths. The source is an effective round waist, as in the earlier model.

The aperture's axial coordinate is an unknown fraction of the drift beside CL3. Separate fits test the drift before and after CL3. The fraction is `sigmoid(w)` with −7 ≤ w ≤ 0, keeping the aperture within the adjoining half-drift. Its transverse centre is unknown within ±3 µm. The generating aperture is 2.5 mm after the CL3 exit, at z = −80 mm, with centre (0.20, −0.15) µm. Inference does not read these generating values.

The image-only fit supplies starting values. **The joint fit holds no physical fit parameter fixed**, including the first source drift and static pre-field strength. It uses broad hard bounds and no soft penalties. Lens polarities and the strength branch remain model assumptions. Current-to-strength gains and lengths are shared across every measurement, linking rotation and focusing.

## Existing Gaussian aperture machinery

`make_transmission_operator` constructs actual TemGym `GaussianBeam` objects and applies `SigmoidAperture`. The integrated power of each beamlet after the component is proportional to

$$
\frac{|a|^2}{k\sqrt{\det\operatorname{Im}Q^{-1}}}.
$$

The output/input ratio includes amplitude and width changes. Radial quadrature combines these responses with the incident round Gaussian intensity. Angular integration is analytical, including beam/aperture decentring. The beamlet waist is 5 nm; its spatial variance is subtracted from the centre distribution to retain the intended uncut spatial variance. This computes current at the aperture; subsequent lossless propagation conserves it. It does not synthesize clipped MAGICAL images or feed clipped moments into the uncut-Gaussian image analysis.

The implementation uses the existing sigmoid component, not a hard-disk substitute. Its transmission is an **amplitude** transmission, so current depends on its squared magnitude. Increasing radial order from 192 to 384 changes calibration transmissions by less than 2 × 10⁻¹⁴ in the tested generating case. Tests also compare parameter derivatives with finite differences.

For aperture counts Nᵢ and an open reference N₀, current analysis returns Tᵢ = Nᵢ/N₀ and the large-count covariance

$$
\operatorname{Cov}(T_i,T_j)
\simeq \delta_{ij}\frac{N_i}{N_0^2}
+\frac{N_iN_j}{N_0^3}.
$$

The shared-reference term is retained. Count arrays are converted to floating point before forming powers. There is no independent dose fit for each radius that could absorb the transmission signal.

## Result

The joint fit uses the full covariance of image-derived beam measurements and current ratios. Eight starts are tried for each aperture placement. In the saved run:

| Check | Result |
|---|---:|
| Preferred aperture placement | After CL3 |
| Joint calibration residual RMS | 0.913 |
| New image-measurement residual RMS | 0.995 |
| New aperture-transmission residual RMS | 0.805 |
| Maximum controlled strength-gain error | 0.037% |
| Local sensitivity rank | 21 of 22 |
| Converged starts after CL3 | 1 of 8 |

Residual RMS values use measurement uncertainty; prediction residuals exclude fitted-model uncertainty. Seven starts after CL3 reach the 3,000-evaluation limit, some well outside measurement noise. All before-CL3 starts also reach that limit. The saved report retains these outcomes. This experiment does not establish robust optimizer convergence or count all minima.

The incorrect placement can nevertheless be rejected independently of those optimizer failures. An aperture before CL3 cannot respond to CL3 excitation at fixed CL1 in this model. Even an unrestricted fit of one transmission vector per CL1 setting has a current residual RMS of at least **166.6**. The variation with CL3 therefore establishes the ordering within the two tested hypotheses.

Physical dimensions can still have substantial errors. The best fit has source z ≈ −359 mm; truth is −325 mm. Accurate predictions and controlled strength gains do not establish accurate source position or lens thicknesses.

## Exact surviving ambiguity

The [image-only notes](CONDENSER_CALIBRATION_NOTES.md) derive a two-parameter family for the uncut condenser. Calibrated aperture sizes constrain its upstream scaling freedom. A second freedom remains when the static pre-field and aperture coordinates are unknown.

Choose a different static pre-field strength q′. Keep upstream scale at one and adjust pre-field length and surrounding drifts using `equivalent_parameters`. Rotate the source mean positions and angles by q′−q. Keep the aperture at the same fractional distance beside CL3 and rotate its unknown transverse centre by the same angle.

Upstream of the pre-field, the round beam and circular aperture rotate together, leaving transmitted current unchanged. Downstream, the compensating pre-field transformation restores the original specimen beam. This preserves **all measured quantities at any control setting**, including settings outside the calibration dataset.

The notebook returns three such models with different pre-field strengths and lengths. Their whitened predictions differ by less than 10⁻⁸. This constructive ambiguity establishes non-uniqueness independently of optimizer convergence and numerical rank.

An independently calibrated constraint sensitive to this remaining freedom is needed for a unique physical interpretation. More repetitions or concentric aperture diameters at the same unknown plane cannot remove it. A known axial reference or independently measured pre-field quantity are possible constraints; their uncertainty would need to enter a subsequent experiment.

## Paper connection

The supplied [paper, §II.2.1 and Fig. 2](https://arxiv.org/html/2607.29411v1#S2.SS2.SSS1), uses aperture current versus preceding lens excitation to infer crossover conditions. Its [rotation calibration](https://arxiv.org/html/2607.29411v1#S2.SS2.SSS2) adds information about strength and effective length. This experiment follows the transmission idea while retaining our finite solenoid, unknown source and independent static pre-field. The paper's procedure is not a uniqueness proof for these 22 unknowns.
