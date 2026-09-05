"""Image -> measurement -> unknown-geometry fit for a finite solenoid column.

The first experiment uses Gaussian phase-space intensity transport, for which
finite-pixel images are analytic. It models a partially coherent calibration
feature, not coherent specimen interference. All ray matrices returned by the
public adapter use [x, y, theta_x, theta_y, 1].
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq, least_squares

from temgym_core.constants import energy2wavelength
from temgym_core.microscope_inversion import drift_matrix, hard_edge_solenoid_matrix
from temgym_core.gaussian_image_likelihood import normal_interval_probability


LENSES = ("Objective", "IL1", "IL2", "IL3", "PL1")
PARAMETERS = (tuple(f"log_gap_{i}_m" for i in range(6))
              + tuple(f"log_length_{n}_m" for n in LENSES)
              + ("objective_q",)
              + tuple(f"log_abs_excitation_gain_{n}" for n in LENSES[1:])
              + ("illumination_curvature_per_1e4_m_inv", "log_angular_spread_rad", "specimen_rotation_rad"))
SIGNS = jnp.array([1., 1., 1., -1.])


def public_specification():
    return {
        "voltage_v": 200000., "pixel_pitch_m": 110e-6, "shape_yx": [128, 128],
        "feature_sigma_m": 2e-9,
        "feature_centres_m": (np.array([[-6., -5.], [7., -4.], [-4., 7.]]) * 1e-9).tolist(),
        "scope": "Known-shape synthetic three-mark feature with one absolute size calibration; intensities are fitted from each image",
        "lens_order": list(LENSES), "coordinate_order": ["x", "y", "theta_x", "theta_y", "1"],
        "illumination_model": "Unknown common round angular spread and wavefront curvature",
        "control_law": "q_i = signed_gain_i * recorded_control_i; objective control fixed",
    }


def initial_guess():
    """Literal engineering guesses, independently specified from the generator."""
    return np.r_[np.log([.0025, .19, .08, .05, .06, .34]),
                 np.log([.0005, .006, .004, .006, .0012]),
                 -.40, np.log([.50, .60, .50, .70]), 0., np.log(.0012), 0.]


def decode_geometry(p):
    gaps, lengths = jnp.exp(p[:6]), jnp.exp(p[6:11])
    entrances = jnp.cumsum(gaps[:-1]) + jnp.r_[0., jnp.cumsum(lengths[:-1])]
    return gaps, lengths, entrances + lengths / 2, jnp.sum(gaps) + jnp.sum(lengths)


def matrix5(p, controls):
    """Finite solenoid map in grouped ray coordinates.

    Four controls mean IL1, IL2, IL3, PL1 with the objective held at one.
    Five controls mean Objective, IL1, IL2, IL3, PL1. The objective control
    multiplies its fitted reference strength, just as for the other lenses.
    """
    controls = jnp.asarray(controls)
    if controls.shape not in ((4,), (5,)):
        raise ValueError("Supply four projector controls or five controls including the objective first.")
    gaps, lengths, _, _ = decode_geometry(p)
    objective = controls[0] if controls.shape == (5,) else 1.
    q = jnp.r_[p[11]*objective, SIGNS * jnp.exp(p[12:16]) * controls[-4:]]
    matrix = drift_matrix(gaps[0])
    for i in range(5):
        matrix = drift_matrix(gaps[i + 1]) @ hard_edge_solenoid_matrix(q[i], lengths[i]) @ matrix
    order = jnp.array([0, 2, 1, 3])
    return jnp.eye(5).at[:4, :4].set(matrix[jnp.ix_(order, order)])


def predict_observables(p, controls, spec):
    """Measured log scale, pattern angle, and log Gaussian width in pixels.

    Input mean angles are h*x, so defocused pattern scale measures A+B*h.
    The angular width is independent of each feature's spatial coordinate;
    covariance transport supplies the total image width. Pattern translations
    and individual feature counts are profiled out by image analysis.
    """
    matrix = matrix5(p, controls)
    h = p[16] * 1e4
    effective_a = matrix[:2, :2] + h * matrix[:2, 2:4]
    scale2 = jnp.sum(effective_a[:, 0]**2)
    b2 = jnp.sum(matrix[:2, 2]**2)
    variance = scale2 * spec["feature_sigma_m"]**2 + b2 * jnp.exp(2 * p[17])
    return jnp.array([
        .5 * jnp.log(scale2), jnp.arctan2(effective_a[1, 0], effective_a[0, 0]) + p[18],
        .5 * jnp.log(variance) - jnp.log(spec["pixel_pitch_m"]),
    ])


def predict_tilt_displacements(p, controls, angle_changes_rad, pixel_pitch_m):
    """Image displacement in detector x/y pixels for calibrated specimen tilts.

    One fixed lens setting per tilt pair. A shared image offset cancels when
    subtracting the two images; input tilt axes use the matrix's x/y basis.
    """
    return jax.vmap(lambda d, angle: matrix5(p, d)[:2, 2:4] @ angle)(
        jnp.asarray(controls), jnp.asarray(angle_changes_rad)) / pixel_pitch_m


def predict_beam_shift_displacements(p, controls, position_changes_m,
                                    angle_changes_rad, pixel_pitch_m):
    """Response of a translated incoming beam: (A ds + B dtheta) / pixel size.

    Positions and mean angles are independently calibrated in specimen x/y
    axes. This is a beam displacement, not a specimen-stage translation.
    """
    def displacement(commands, position, angle):
        matrix = matrix5(p, commands)
        return matrix[:2, :2] @ position + matrix[:2, 2:4] @ angle
    return jax.vmap(displacement)(jnp.asarray(controls), jnp.asarray(position_changes_m),
                                  jnp.asarray(angle_changes_rad)) / pixel_pitch_m


def make_image_operator(spec):
    """Exact pixel-integrated sum of round Gaussian intensity features.

    Image-analysis parameters: log scale, angle, log width_px, translation
    x/y in pixels, log counts for each mark, and log background per pixel.
    The pixel integral uses Gaussian CDF differences, with no spatial mesh
    approximation or coherent beamlet truncation.
    """
    centres = jnp.asarray(spec["feature_centres_m"]) / spec["pixel_pitch_m"]
    ny, nx = spec["shape_yx"]
    xedge = jnp.arange(nx + 1) - nx / 2
    yedge = jnp.arange(ny + 1) - ny / 2

    def image(p):
        phi = p[1]
        rotation = jnp.array([[jnp.cos(phi), -jnp.sin(phi)], [jnp.sin(phi), jnp.cos(phi)]])
        xy = jnp.exp(p[0]) * (centres @ rotation.T) + p[3:5]
        width = jnp.exp(p[2])
        px = normal_interval_probability((xedge[:-1, None] - xy[:, 0]) / width,
                                         (xedge[1:, None] - xy[:, 0]) / width)
        py = normal_interval_probability((yedge[:-1, None] - xy[:, 1]) / width,
                                         (yedge[1:, None] - xy[:, 1]) / width)
        return jnp.einsum("yk,xk,k->yx", py, px, jnp.exp(p[5:8])) + jnp.exp(p[8])

    return jax.jit(image)


def _generator_parameters():
    centres = np.array([.0023, .2135239554, .2904801564, .3529801564, .4079834965])
    lengths = np.array([.0004, .005, .005, .005, .001])
    gaps = np.r_[centres[0] - lengths[0] / 2,
                 np.diff(centres) - (lengths[1:] + lengths[:-1]) / 2,
                 .71875 - centres[-1] - lengths[-1] / 2]
    # Focus the effective objective at a chosen synthetic intermediate plane.
    def radial(q, length):
        return np.array([[np.cos(q), length * np.sinc(q / np.pi)],
                         [-q / length * np.sin(q), np.cos(q)]])
    drift = lambda d: np.array([[1., d], [0., 1.]])
    q0 = brentq(lambda q: (drift(.1881936689 - centres[0] - lengths[0] / 2)
                           @ radial(q, lengths[0]) @ drift(gaps[0]))[0, 1], .3, .6)
    q4 = brentq(lambda q: (drift(gaps[-1]) @ radial(q, lengths[-1])
                           @ drift(centres[-1] - lengths[-1] / 2 - .4056802138))[0, 1], .4, .9)
    return np.r_[np.log(gaps), np.log(lengths), -q0, np.log([.55, .65, .45, .75]),
                 .2, np.log(.001), .12], q4


def generate_dataset(directory, seed=6201, *, magnifications=None, wobble_fraction=.025,
                     settings_only=False, specification=None):
    """Write images/metadata separately from the parameters reserved for scoring."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec = public_specification()
    if specification is not None:
        spec.update(specification)
    truth, final_q = _generator_parameters()
    controls, labels, nominal, train = [], [], [], []
    matrix_fn = jax.jit(matrix5)
    previous = np.array([.53 / .55, .50 / .65])
    reference_settings = [(40000, .03286, True), (80000, .04304, False),
                          (150000, .05906, True), (300000, .10484, True), (600000, .40885, False)]
    if magnifications is not None:
        mags = np.asarray(magnifications, dtype=float)
        if (mags.ndim != 1 or not len(mags) or not np.isfinite(mags).all()
                or np.any(mags < 40000) or np.any(mags > 600000)
                or np.any(np.diff(mags) <= 0)):
            raise ValueError("Supply increasing, distinct magnifications between 40,000 and 600,000.")
        reference_settings = [(float(m), float(np.interp(m, [40000, 80000, 150000, 300000, 600000],
                                                        [.03286, .04304, .05906, .10484, .40885])), True)
                              for m in mags]
    if not 0 < wobble_fraction < .1:
        raise ValueError("The synthetic wobble fraction must be between 0 and 0.1.")
    for mag, f3, is_train in reference_settings:
        base = np.array([*previous, np.sqrt(.005 / f3) / .45, final_q / .75])

        def residual(v):
            row = base.copy()
            row[:2] = v
            m = np.asarray(matrix_fn(truth, row))
            q = np.r_[truth[11], np.array([.55, .65, .45, -.75]) * row]
            rotation = np.array([[np.cos(sum(q)), -np.sin(sum(q))], [np.sin(sum(q)), np.cos(sum(q))]])
            a = rotation @ m[:2, :2]
            b = rotation @ m[:2, 2:4]
            return np.array([a[0, 0] / mag - 1, b[0, 0] / (mag * .0023)])

        result = least_squares(residual, base[:2], bounds=(.1, 3.),
                               xtol=1e-13, ftol=1e-13, gtol=1e-13)
        if np.linalg.norm(residual(result.x)) > 1e-8:
            raise RuntimeError("Synthetic finite-lens reference mode failed to focus.")
        base[:2] = result.x
        previous = result.x.copy()
        perturbations = [np.zeros(4)]
        if is_train:
            perturbations += [sign * wobble_fraction * unit for unit in np.eye(4) for sign in [-1, 1]]
        else:
            perturbations += [sign * np.array([.012, -.018, .015, -.01]) for sign in [-1, 1]]
        for i, perturbation in enumerate(perturbations):
            controls.append(base * (1 + perturbation))
            if magnifications is None:
                labels.append(f"{mag // 1000}k {'training' if is_train else 'withheld'} {i}")
            else:
                change = "reference" if i == 0 else f"{LENSES[1 + (i-1)//2]} {(-1 if i % 2 else 1)*wobble_fraction:+.1%}"
                labels.append(f"{mag:,.0f}× {change}")
            nominal.append(mag)
            train.append(is_train)
    controls = np.array(controls)
    if settings_only:
        manifest = dict(spec, labels=labels, nominal_magnifications=nominal,
                        training_mask=train, controls=controls.tolist(), read_noise_std_e=2.,
                        noise_model="Poisson plus independent Gaussian read noise")
        (directory / "acquisition.json").write_text(json.dumps(manifest, indent=2))
        (directory / "scoring_truth.json").write_text(json.dumps({"parameters": truth.tolist(), "parameter_names": PARAMETERS}, indent=2))
        (directory / "initial_guess.json").write_text(json.dumps({"parameters": initial_guess().tolist(), "parameter_names": PARAMETERS}, indent=2))
        return manifest
    operator = make_image_operator(spec)
    rng = np.random.default_rng(seed)
    images = []
    expected = []
    for row in controls:
        observables = np.asarray(predict_observables(truth, row, spec))
        # Alignment and individual feature brightness are independently fitted
        # nuisance quantities, not supplied to the image-analysis initial guess.
        image_p = np.r_[observables, rng.normal(0, .3, 2), np.log([8e5, 6e5, 1e6]), np.log(3.)]
        mean = np.asarray(operator(image_p))
        expected.append(mean)
        images.append(rng.poisson(mean).astype(float) + rng.normal(0, 2., mean.shape))
    manifest = dict(spec, labels=labels, nominal_magnifications=nominal,
                    training_mask=train, controls=controls.tolist(),
                    read_noise_std_e=2., noise_model="Poisson plus independent Gaussian read noise")
    (directory / "acquisition.json").write_text(json.dumps(manifest, indent=2))
    np.savez_compressed(directory / "images.npz", images=np.array(images))
    (directory / "scoring_truth.json").write_text(json.dumps({"parameters": truth.tolist(), "parameter_names": PARAMETERS}, indent=2))
    np.savez_compressed(directory / "scoring_expected_images.npz", expected=np.array(expected))
    (directory / "initial_guess.json").write_text(json.dumps({"parameters": initial_guess().tolist(), "parameter_names": PARAMETERS}, indent=2))
    return manifest


def analyse_images(directory):
    """Consume only images and public acquisition metadata; write measurements."""
    directory = Path(directory)
    spec = json.loads((directory / "acquisition.json").read_text())
    images = np.load(directory / "images.npz")["images"]
    operator = make_image_operator(spec)
    jacobian = jax.jit(jax.jacfwd(operator))
    measurements, covariances, costs, image_parameters = [], [], [], []
    for image, nominal in zip(images, spec["nominal_magnifications"]):
        sigma = np.sqrt(np.maximum(image, 1.) + spec["read_noise_std_e"]**2)
        total = max(float(np.sum(image - np.median(image))), 1.)
        # A coarse search in angle is an image-registration operation, not a
        # query of the generating optics. Magnification labels only seed scale.
        best = None
        for angle in np.linspace(-np.pi, np.pi, 13)[:-1]:
            p = np.r_[np.log(nominal), angle, np.log(max(nominal * spec["feature_sigma_m"] / spec["pixel_pitch_m"], 1.)),
                      0., 0., np.log([total / 3] * 3), np.log(max(np.median(image), .1))]
            cost = float(np.sum(((np.asarray(operator(p)) - image) / sigma)**2))
            if best is None or cost < best[0]:
                best = (cost, p)

        def residual(p):
            return ((np.asarray(operator(p)) - image) / sigma).ravel()

        def jac(p):
            return np.asarray(jacobian(p)).reshape(-1, 9) / sigma.ravel()[:, None]

        lower = np.r_[np.log(nominal / 4), -4 * np.pi, np.log(.15), -20., -20., [np.log(100)] * 3, np.log(.01)]
        upper = np.r_[np.log(nominal * 4), 4 * np.pi, np.log(60.), 20., 20., [np.log(1e9)] * 3, np.log(1000)]
        result = least_squares(residual, best[1], jac=jac, bounds=(lower, upper),
                               x_scale="jac", max_nfev=100, ftol=1e-10, xtol=1e-10, gtol=1e-5)
        if not result.success:
            raise RuntimeError("Image analysis did not converge.")
        information = result.jac.T @ result.jac
        covariance = np.linalg.inv(information)[:3, :3]
        measurements.append(result.x[:3])
        image_parameters.append(result.x)
        covariances.append(covariance)
        costs.append(np.sqrt(np.mean(result.fun**2)))
    np.savez_compressed(directory / "measurements.npz", values=np.array(measurements),
                        covariance=np.array(covariances), image_residual_rms=np.array(costs),
                        image_parameters=np.array(image_parameters))
    return np.array(measurements), np.array(covariances), np.array(costs)


def position_prior_coordinates(priors):
    """Map bounded centre/detector coordinates to the column's log gaps.

    Position coordinates are standardized by the stated prior standard
    deviations. Bounds must guarantee positive gaps even at their corners;
    this makes every optimizer iterate an ordered, non-overlapping column.
    """
    mean = np.asarray(priors["position_mean_mm"], dtype=float)
    sigma = np.asarray(priors["position_sigma_mm"], dtype=float)
    bounds = np.asarray(priors["position_bounds_mm"], dtype=float)
    lengths = np.asarray(priors["length_bounds_mm"], dtype=float)
    if mean.shape != (6,) or sigma.shape != (6,) or bounds.shape != (6, 2) or lengths.shape != (5, 2):
        raise ValueError("Expected six position priors and five lens-length bounds.")
    if (not all(np.isfinite(a).all() for a in (mean, sigma, bounds, lengths))
            or np.any(sigma <= 0) or np.any(bounds[:, 1] <= bounds[:, 0])
            or np.any(lengths[:, 0] <= 0) or np.any(lengths[:, 1] <= lengths[:, 0])
            or np.any(mean < bounds[:, 0]) or np.any(mean > bounds[:, 1])):
        raise ValueError("Prior means, widths, and bounds must be finite and consistent.")
    minimum_gaps = np.r_[bounds[0, 0] - lengths[0, 1] / 2,
                        bounds[1:5, 0] - bounds[:4, 1] - (lengths[1:, 1] + lengths[:-1, 1]) / 2,
                        bounds[5, 0] - bounds[4, 1] - lengths[4, 1] / 2]
    if np.any(minimum_gaps <= 0):
        raise ValueError("Position/length bounds must guarantee positive non-overlapping gaps.")

    def encode(p):
        _, _, centres, detector = decode_geometry(p)
        return jnp.asarray(p).at[:6].set((jnp.r_[centres, detector] * 1000 - mean) / sigma)

    def decode(u):
        z = (mean + sigma * u[:6]) / 1000
        length = jnp.exp(u[6:11])
        gaps = jnp.r_[z[0] - length[0] / 2,
                      jnp.diff(z[:5]) - (length[1:] + length[:-1]) / 2,
                      z[5] - z[4] - length[4] / 2]
        return jnp.asarray(u).at[:6].set(jnp.log(gaps))

    return encode, decode, (bounds[:, 0] - mean) / sigma, (bounds[:, 1] - mean) / sigma


def fit_unknown_column(directory, guess=None, max_nfev=8000, regularization=1., position_priors=None):
    """Fit 19 unknowns using the measurement table, never the scoring files."""
    directory = Path(directory)
    spec = json.loads((directory / "acquisition.json").read_text())
    data = np.load(directory / "measurements.npz")
    mask = np.array(spec["training_mask"])
    initial = np.array(json.loads((directory / "initial_guess.json").read_text())["parameters"] if guess is None else guess)
    return fit_column_from_measurements(
        lens_controls=np.asarray(spec["controls"])[mask], measured_values=data["values"][mask],
        measurement_covariances=data["covariance"][mask], specimen_and_detector=spec,
        starting_parameters=initial, position_priors=position_priors,
        prior_weight=regularization, max_evaluations=max_nfev,
    )


def fit_column_from_measurements(*, lens_controls, measured_values, measurement_covariances,
                                 specimen_and_detector, starting_parameters,
                                 position_priors=None, prior_weight=1., max_evaluations=8000,
                                 tilt_measurements=None, prior_parameters=None,
                                 beam_shift_measurements=None):
    """Fit once; keep the prior centre separate from the optimizer's start.

    Omitted prior_parameters preserves the historical single-fit behaviour.
    For multiple starts, prepare_column_fit compiles one fixed objective once.
    """
    problem = prepare_column_fit(
        lens_controls=lens_controls, measured_values=measured_values,
        measurement_covariances=measurement_covariances,
        specimen_and_detector=specimen_and_detector,
        prior_parameters=starting_parameters if prior_parameters is None else prior_parameters,
        position_priors=position_priors, prior_weight=prior_weight,
        tilt_measurements=tilt_measurements,
        beam_shift_measurements=beam_shift_measurements,
    )
    return problem.solve(starting_parameters, max_evaluations=max_evaluations)


def prepare_column_fit(*, lens_controls, measured_values, measurement_covariances,
                       specimen_and_detector, prior_parameters,
                       position_priors=None, prior_weight=1., tilt_measurements=None,
                       beam_shift_measurements=None, use_soft_priors=True):
    """Compile a fixed measurement/prior objective. This function reads no files.

    Controls have shape (frames, 4), ordered IL1, IL2, IL3, PL1, or (frames, 5)
    with Objective first. Values have
    shape (frames, 3): log scale, rotation in radians, log width in pixels.
    Covariances have shape (frames, 3, 3), in those same coordinates.
    Prior parameters use the 19-element vector defined by PARAMETERS and stay
    fixed across all optimizer starts and detector-position profiles.
    use_soft_priors=False removes ALL penalty residuals, including positions.
    The declared hard bounds and numerical coordinate scaling remain unchanged.
    Optional tilt_measurements supplies independent image-pair displacements
    (pixels), their 2x2 covariances, controls, and angle changes (radians).
    Optional beam_shift_measurements adds calibrated position changes (metres)
    and accompanying mean-angle changes (radians), with measured beam-centre
    differences (pixels) and their 2x2 covariance. Only these differences enter
    the column fit; texture shape and illumination coordinates are fitted
    jointly in the preceding image-analysis step.
    Calling the returned object's solve method gives a scipy result with the
    fitted vector, data residuals, and data-only Jacobian. The notebook adapter gives these named fields
    and translates the vector into positions, lengths, gains, and angles.
    """
    controls = jnp.asarray(lens_controls)
    observations = jnp.asarray(measured_values)
    covariances = np.asarray(measurement_covariances)
    if (controls.ndim != 2 or controls.shape[1] not in (4, 5) or not len(controls)
            or observations.shape != (len(controls), 3)
            or covariances.shape != (len(controls), 3, 3)):
        raise ValueError("Controls, measurements, and covariances must have matching frame counts.")
    if not all(np.isfinite(a).all() for a in (controls, observations, covariances)):
        raise ValueError("Measurements and controls must be finite.")
    whitening = jnp.asarray(np.linalg.inv(np.linalg.cholesky(covariances)))
    spec = specimen_and_detector
    initial = np.asarray(prior_parameters, dtype=float)
    if initial.shape != (len(PARAMETERS),) or not np.isfinite(initial).all():
        raise ValueError("Supply 19 finite starting parameters.")
    regularization = prior_weight

    if tilt_measurements is not None:
        tilt_controls = np.asarray(tilt_measurements["lens_controls"])
        tilt_angles = np.asarray(tilt_measurements["angle_changes_rad"])
        tilt_shifts = np.asarray(tilt_measurements["displacements_pixels"])
        tilt_covariance = np.asarray(tilt_measurements["covariance_pixels"])
        count = len(tilt_controls)
        if (not count or tilt_controls.ndim != 2 or tilt_controls.shape[1] not in (4, 5)
                or tilt_angles.shape != (count, 2) or tilt_shifts.shape != (count, 2)
                or tilt_covariance.shape != (count, 2, 2)):
            raise ValueError("Tilt controls, angles, displacements, and covariances must have matching pair counts.")
        if not all(np.isfinite(a).all() for a in (tilt_controls, tilt_angles, tilt_shifts, tilt_covariance)):
            raise ValueError("Tilt measurements must be finite.")
        if np.any(np.linalg.norm(tilt_angles, axis=1) == 0):
            raise ValueError("Each tilt pair needs a nonzero calibrated angle change.")
        tilt_whitening = jnp.asarray(np.linalg.inv(np.linalg.cholesky(tilt_covariance)))

    if beam_shift_measurements is not None:
        beam_controls = np.asarray(beam_shift_measurements["lens_controls"])
        beam_positions = np.asarray(beam_shift_measurements["position_changes_m"])
        beam_angles = np.asarray(beam_shift_measurements["angle_changes_rad"])
        beam_displacements = np.asarray(beam_shift_measurements["displacements_pixels"])
        beam_covariance = np.asarray(beam_shift_measurements["covariance_pixels"])
        count = len(beam_controls)
        if (not count or beam_controls.ndim != 2 or beam_controls.shape[1] not in (4, 5)
                or beam_positions.shape != (count, 2) or beam_angles.shape != (count, 2)
                or beam_displacements.shape != (count, 2) or beam_covariance.shape != (count, 2, 2)):
            raise ValueError("Beam-shift arrays must have matching pair counts and x/y coordinates.")
        if not all(np.isfinite(a).all() for a in
                   (beam_controls, beam_positions, beam_angles, beam_displacements, beam_covariance)):
            raise ValueError("Beam-shift measurements must be finite.")
        if np.any(np.linalg.norm(beam_positions, axis=1) == 0):
            raise ValueError("Each beam-shift pair needs a nonzero calibrated position change.")
        beam_whitening = jnp.asarray(np.linalg.inv(np.linalg.cholesky(beam_covariance)))

    def residual(p):
        predicted = jax.vmap(lambda d: predict_observables(p, d, spec))(controls)
        difference = predicted - observations
        difference = difference.at[:, 1].set(jnp.arctan2(jnp.sin(difference[:, 1]), jnp.cos(difference[:, 1])))
        image_residual = jnp.einsum("nij,nj->ni", whitening, difference).ravel()
        parts = [image_residual]
        if tilt_measurements is not None:
            predicted_shifts = predict_tilt_displacements(p, tilt_controls, tilt_angles, spec["pixel_pitch_m"])
            parts.append(jnp.einsum("nij,nj->ni", tilt_whitening, predicted_shifts-tilt_shifts).ravel())
        if beam_shift_measurements is not None:
            predicted_shifts = predict_beam_shift_displacements(
                p, beam_controls, beam_positions, beam_angles, spec["pixel_pitch_m"])
            parts.append(jnp.einsum("nij,nj->ni", beam_whitening, predicted_shifts-beam_displacements).ravel())
        return jnp.concatenate(parts)

    compiled = jax.jit(residual)
    jacobian = jax.jit(jax.jacfwd(residual))
    # Broad engineering bounds are part of the inference specification. Their
    # role in selecting an equivalent geometry is shown by the scale audit.
    base = initial_guess()
    lower = np.r_[base[:11] - 1.2, -.9, base[12:16] - .8, -10., np.log(.00015), -np.pi]
    upper = np.r_[base[:11] + 1.2, -.15, base[12:16] + .8, 10., np.log(.01), np.pi]
    # Minimum Gaussian emittance at the known wavelength and feature width.
    lower[17] = np.log(float(energy2wavelength(spec["voltage_v"])) / (4 * np.pi * spec["feature_sigma_m"]))
    # Some measurement sets leave exact or very weak parameter directions.
    # Keep the data-only residual/Jacobian so added controls can be checked
    # independently of the priors that select among poorly constrained models.
    if position_priors is None:
        encode = lambda p: p
        decode = lambda u: u
        solver_initial = initial
        prior_residual = lambda p: regularization * (p - initial)
    else:
        encode, decode, position_lower, position_upper = position_prior_coordinates(position_priors)
        solver_initial = np.asarray(encode(initial))
        lower[:6], upper[:6] = position_lower, position_upper
        length_bounds = np.asarray(position_priors["length_bounds_mm"]) / 1000
        lower[6:11], upper[6:11] = np.log(length_bounds[:, 0]), np.log(length_bounds[:, 1])
        prior_residual = lambda p: jnp.r_[encode(p)[:6], regularization * (p[6:] - initial[6:])]
    if np.any(solver_initial < lower) or np.any(solver_initial > upper):
        raise ValueError("The starting geometry is outside the declared prior bounds; revise it explicitly.")

    if not use_soft_priors:
        regularization = 0.
        prior_residual = lambda p: jnp.empty((0,), dtype=p.dtype)

    def full_residual(p):
        return jnp.r_[residual(p), prior_residual(p)]

    regularized_residual = jax.jit(lambda u: full_residual(decode(u)))
    regularized_jacobian = jax.jit(jax.jacfwd(regularized_residual))
    return PreparedColumnFit(encode, decode, lower, upper, initial, position_priors,
                             regularization, regularized_residual, regularized_jacobian,
                             compiled, jacobian, jax.jit(jax.jacfwd(full_residual)))


class PreparedColumnFit:
    """A fixed data set and priors, reusable across starts and constrained fits."""

    def __init__(self, encode, decode, lower, upper, prior_parameters, position_priors,
                 prior_weight, residual, jacobian, data_residual, data_jacobian, physical_jacobian):
        self.encode, self.decode = encode, decode
        self.lower, self.upper = lower, upper
        self.prior_parameters = prior_parameters.copy()
        self.position_priors, self.prior_weight = position_priors, prior_weight
        self.residual, self.jacobian = residual, jacobian
        self.data_residual, self.data_jacobian = data_residual, data_jacobian
        self.physical_jacobian = physical_jacobian

    def solve(self, starting_parameters, *, max_evaluations=8000, fixed_detector_position_mm=None):
        """Refit every free coordinate; pin detector position exactly if supplied."""
        start = np.asarray(starting_parameters, dtype=float)
        if start.shape != (19,) or not np.isfinite(start).all():
            raise ValueError("Supply 19 finite starting parameters.")
        initial = np.asarray(self.encode(start)).copy()
        free = np.ones(19, dtype=bool)
        if fixed_detector_position_mm is not None:
            if self.position_priors is None:
                raise ValueError("A detector profile requires position priors to define position coordinates.")
            mean = self.position_priors["position_mean_mm"][5]
            sigma = self.position_priors["position_sigma_mm"][5]
            initial[5] = (fixed_detector_position_mm-mean)/sigma
            free[5] = False
        if not np.isfinite(initial).all() or np.any(initial < self.lower) or np.any(initial > self.upper):
            raise ValueError("The starting geometry or fixed detector is outside the declared prior bounds.")

        def expand(v):
            u = initial.copy()
            u[free] = v
            return u

        result = least_squares(lambda v: np.asarray(self.residual(expand(v))), initial[free],
                               jac=lambda v: np.asarray(self.jacobian(expand(v)))[:, free],
                               bounds=(self.lower[free], self.upper[free]), x_scale="jac",
                               max_nfev=max_evaluations, ftol=1e-9, xtol=1e-10, gtol=1e-5)
        result.solver_x = expand(result.x)
        result.solver_jacobian = result.jac.copy()
        result.solver_active_mask = result.active_mask.copy()
        result.free_coordinate_mask = free
        result.fixed_detector_position_mm = fixed_detector_position_mm
        result.x = np.asarray(self.decode(result.solver_x))
        result.jac = np.asarray(self.physical_jacobian(result.x))
        result.grad = result.jac.T @ result.fun
        if self.position_priors is not None:
            del result.active_mask
        result.data_residual = np.asarray(self.data_residual(result.x))
        result.data_jacobian = np.asarray(self.data_jacobian(result.x))
        result.regularization = self.prior_weight
        result.position_priors = self.position_priors
        result.prior_parameters = self.prior_parameters.copy()
        return result


def axial_scale_family(p, scale):
    """Exact observational ambiguity for unknown specimen angular distribution."""
    transformed = np.array(p, copy=True)
    transformed[:11] += np.log(scale)
    transformed[16] /= scale
    transformed[17] -= np.log(scale)
    return transformed
