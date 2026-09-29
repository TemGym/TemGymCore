"""Synthetic SEM reflection calibration with an explicitly local Gaussian model.

Native thin lenses transport a coherent Gaussian source. Each retained surface
order uses the same envelope Jacobian as the reflection notebook, frozen at the
chief ray. Separated orders are integrated as Gaussian intensities, including a
calibrated detector PSF. This is a reduced, local-paraxial pilot, not the full
angular-packet reflection renderer or a dynamical surface-scattering model.

Analysis reads pixels, recorded relative commands and calibrations only. Truth
is a separate, optional file used by the browser and final geometry score.
"""
import hashlib
import inspect
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr
from scipy.signal import find_peaks

from .components import Lens, Plane
from .config import require_x64
from .coordinate_transforms import PlaneFrame
from .gaussian import FreeSpacePropagator
from .optimization import least_squares
from .run import run_to_end
from .scattering import PotentialStepResponse, copper_scattering_factor
from .source import gaussian_waist

NAMES = (
    "Source z (mm)", "C1 z (mm)", "C2 z (mm)", "Objective z (mm)",
    "C1 f (mm)", "C2 f (mm)", "Objective f (mm)", "Source RMS (nm)",
    "Initial camera distance (mm)",
)
# Disjoint axial bounds enforce source < C1 < C2 < objective < specimen.
LOWER = np.array([-500., -300., -180., -75., 20., 12., 10., .7, 80.])
UPPER = np.array([-310., -200., -85., -15., 80., 55., 50., 5., 180.])
INITIAL = np.array([-390., -250., -145., -52., 46., 30., 25., 2.5, 145.])
SCALE = np.array([100., 80., 60., 30., 30., 20., 20., 2., 100.])
FS = FreeSpacePropagator()


def default_specification():
    """Independent calibrations and declared assumptions, not fitted geometry."""
    return dict(voltage_v=30000., incidence_deg=2., surface_period_m=.3615e-9 / np.sqrt(2),
                pixel_pitch_m=55e-6, detector_shape=[256, 256], psf_sigma_pixels=.6,
                background_counts=.02, read_noise_counts=.2, incident_electrons=1e9,
                orders=[-1, 0, 1], inner_potential_v=21.2, absorption_ev=0.,
                control_law="1/f_i = relative_power_i / reference_f_i",
                detector_motion="laboratory Z; fixed normal and no height following",
                image_axes="rows: -outgoing u; columns: outgoing v",
                approximation="chief-ray local Gaussian reflection; separated orders",
                incidence_known=True, detector_orientation_known=True,
                surface_hit_known=True, sample_z_m=0.)


def demonstration_truth():
    """Only the image generator and optional truth score may use this."""
    return np.array([-350., -270., -130., -40., 40.,
                     1000 / (1 / .060 + 1 / .045),
                     1000 / (1 / .045 + 1 / .040), 1.91, 120.])


def acquisition(seed=42):
    """Recorded power ratios and known laboratory-Z translations, independent of truth."""
    rng = np.random.default_rng(seed)
    single = [np.ones(3)]
    for lens in range(3):
        for change in (-.2, -.1, .1, .2):
            command = np.ones(3)
            command[lens] += change
            single.append(command)
    commands = np.concatenate((single, rng.uniform(.8, 1.2, (24, 3)),
                               rng.uniform(.8, 1.2, (6, 3))))
    purpose = np.repeat(["fit"] * 37 + ["check"] * 6, 3)
    return dict(controls=np.repeat(commands, 3, axis=0),
                detector_dz_m=np.tile([-.025, 0., .025], len(commands)), purpose=purpose)


def column(parameters, controls):
    return tuple(Lens(z=parameters[i + 1] * 1e-3,
                      focal_length=parameters[i + 4] * 1e-3 / controls[i])
                 for i in range(3)) + (Plane(z=0.),)


def frames(spec):
    alpha = np.deg2rad(spec["incidence_deg"])
    incoming = PlaneFrame(origin_xyz=jnp.zeros(3),
        u_xyz=jnp.array([np.cos(alpha), 0., -np.sin(alpha)]), v_xyz=jnp.array([0., 1., 0.]))
    outgoing = PlaneFrame(origin_xyz=jnp.zeros(3),
        u_xyz=jnp.array([np.cos(alpha), 0., np.sin(alpha)]), v_xyz=jnp.array([0., 1., 0.]))
    return incoming, outgoing


def order_geometry(spec):
    """Chief directions and envelope Jacobians from the existing surface adapter."""
    incoming, outgoing = frames(spec)
    source = gaussian_waist(z=0., voltage=spec["voltage_v"], rms_size=1e-9)
    q = jnp.asarray(spec["orders"]) * 2 * jnp.pi / (spec["surface_period_m"] * source.k)
    vi, normal = incoming.normal_xyz, jnp.array([-1., 0., 0.])
    radicand = vi[0] ** 2 - q ** 2
    if np.any(np.asarray(radicand) <= np.sin(np.deg2rad(.25)) ** 2):
        raise ValueError("Every retained chief-ray order must clear the 0.25 degree exit cutoff.")
    directions = jnp.stack((-jnp.sqrt(radicand), q, jnp.full_like(q, vi[2])), axis=-1)
    forward = directions @ outgoing.normal_xyz
    pi = jnp.eye(3) - vi[:, None] * normal[None, :] / (vi @ normal)
    po = jnp.eye(3) - directions[:, :, None] * outgoing.normal_xyz[None, None, :] / forward[:, None, None]
    mapping = outgoing.basis_xyz[:, :2].T @ po @ pi @ incoming.basis_xyz[:, :2]
    slopes = (directions @ outgoing.basis_xyz[:, :2]) / forward[:, None]
    return np.asarray(directions), np.asarray(mapping), np.asarray(slopes)


def make_predictor(spec):
    """Differentiable pixel-centre moments; no generating parameters are captured."""
    _, mapping, slopes = order_geometry(spec)
    inverse = jnp.linalg.inv(jnp.asarray(mapping))
    slopes = jnp.asarray(slopes)
    alpha = np.deg2rad(spec["incidence_deg"])
    pitch = spec["pixel_pitch_m"]
    nr, nc = spec["detector_shape"]
    u = jnp.arange(nr) - (nr - 1) / 2
    v = jnp.arange(nc) - (nc - 1) / 2

    def gaussians(p, controls, dz):
        source = gaussian_waist(z=p[0] * 1e-3, voltage=spec["voltage_v"], rms_size=p[7] * 1e-9)
        incident = run_to_end(source, column(p, controls), propagator=FS)
        q = jnp.swapaxes(inverse, -1, -2) @ incident.Q_inv @ inverse
        count = len(spec["orders"])
        outgoing = incident.derive(x=jnp.zeros(count), y=jnp.zeros(count), z=jnp.zeros(count),
            dx=slopes[:, 0], dy=slopes[:, 1], Q_inv=q,
            amplitude=jnp.ones(count, dtype=complex), pathlength=jnp.zeros(count))
        beam = FS(outgoing, p[8] * 1e-3 + dz * np.cos(alpha))
        mean = jnp.stack((beam.x - dz * np.sin(alpha), beam.y), axis=-1) / pitch
        covariance = jnp.linalg.inv(2 * beam.k[..., None, None] * beam.Q_inv.imag) / pitch ** 2
        covariance += jnp.eye(2) * spec["psf_sigma_pixels"] ** 2
        return mean, covariance

    def marginal(grid, mean, variance, mask):
        sd = jnp.sqrt(variance)
        probability = jax.scipy.special.ndtr((grid + .5 - mean) / sd) - jax.scipy.special.ndtr((grid - .5 - mean) / sd)
        probability = probability * mask
        probability /= jnp.maximum(jnp.sum(probability, axis=-1, keepdims=True), 1e-100)
        centre = jnp.sum(probability * grid, axis=-1)
        variance = jnp.sum(probability * (grid - centre[..., None]) ** 2, axis=-1)
        return centre, variance

    def moments_one(p, controls, dz, strips):
        mean, cov = gaussians(p, controls, dz)
        mu, vu = marginal(u, mean[:, 0, None], cov[:, 0, 0, None], jnp.ones_like(u))
        mask = (v[None, :] >= strips[:, 0, None]) & (v[None, :] < strips[:, 1, None])
        mv, vv = marginal(v, mean[:, 1, None], cov[:, 1, 1, None], mask)
        # Marginal u ignores only exponentially small lateral clipping; checked
        # separately at accepted fits and during acquisition. Pixel integration
        # remains explicit, including when the spot is narrower than one pixel.
        return jnp.stack((mu, mv, vu, vv), axis=-1)

    return (jax.jit(jax.vmap(moments_one, in_axes=(None, 0, 0, 0))),
            jax.jit(jax.vmap(gaussians, in_axes=(None, 0, 0))))


def order_flux(spec):
    period = spec["surface_period_m"]
    reference = np.asarray(copper_scattering_factor(np.abs(np.arange(-256, 257)) / period * 1e-10))
    coefficient = np.asarray(copper_scattering_factor(np.abs(np.asarray(spec["orders"])) / period * 1e-10)) / np.linalg.norm(reference)
    response = PotentialStepResponse(inner_potential_v=spec["inner_potential_v"], absorption_ev=spec["absorption_ev"])
    amplitude = response.amplitude(jnp.deg2rad(spec["incidence_deg"]), spec["voltage_v"])
    return coefficient ** 2 * float(jnp.abs(amplitude) ** 2)


def integrated_image(mean, covariance, flux, spec, quadrature=12):
    """Pixel-integrated correlated Gaussian orders, with the PSF already in cov.

    Integrate one coordinate by Gauss-Legendre and the conditional normal in
    the other by its CDF. No packet intensities are summed: these are separated
    diffraction orders. Their overlap is checked before accepting the dataset.
    """
    nr, nc = spec["detector_shape"]
    u = (nr - 1) / 2 - np.arange(nr)
    v = np.arange(nc) - (nc - 1) / 2
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    result = np.zeros((nr, nc))
    for centre, cov, intensity in zip(mean, covariance, flux):
        vv = v[:, None] + nodes[None, :] / 2
        sigma_v = np.sqrt(cov[1, 1])
        density = np.exp(-.5 * ((vv - centre[1]) / sigma_v) ** 2) / (np.sqrt(2 * np.pi) * sigma_v)
        conditional = centre[0] + cov[0, 1] / cov[1, 1] * (vv - centre[1])
        sigma_u = np.sqrt(cov[0, 0] - cov[0, 1] ** 2 / cov[1, 1])
        prob = ndtr((u[:, None, None] + .5 - conditional) / sigma_u) - ndtr((u[:, None, None] - .5 - conditional) / sigma_u)
        result += intensity * np.sum(prob * density * weights / 2, axis=-1)
    return result


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _numerical_fingerprint():
    digest = hashlib.sha256()
    for function in (default_specification, acquisition, column, frames, order_geometry,
                     make_predictor, order_flux, integrated_image, measure_images,
                     fit_microscope, check_microscope):
        digest.update(inspect.getsource(function).encode())
    for values in (INITIAL, LOWER, UPPER, SCALE):
        digest.update(values.tobytes())
    here = Path(__file__).parent
    sources = sorted((here / "components").glob("*.py"))
    sources += [here / name for name in ("source.py", "gaussian.py", "run.py", "constants.py",
                                         "scattering.py", "optimization.py", "coordinate_transforms.py")]
    for path in sources:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _fingerprint(root, names):
    digest = hashlib.sha256()
    digest.update(_numerical_fingerprint().encode())
    for name in names:
        digest.update((Path(root) / name).read_bytes())
    return digest.hexdigest()


def simulate_experiment(directory, *, truth=None, specification=None, settings=None, seed=7):
    """Write pixels and acquisition records; never overwrite a nonempty directory."""
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Use a new output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    spec = default_specification() if specification is None else specification
    settings = acquisition() if settings is None else settings
    truth = demonstration_truth() if truth is None else np.asarray(truth)
    require_x64("the reflection calibration")
    _, predict = make_predictor(spec)
    mean, covariance = map(np.asarray, predict(truth, settings["controls"], settings["detector_dz_m"]))
    # There is no per-image normalization after clipping or rejecting orders.
    flux = order_flux(spec) * spec["incident_electrons"]
    rng = np.random.default_rng(seed)
    images, errors = [], []
    for i, (mu, cov) in enumerate(zip(mean, covariance)):
        expected = integrated_image(mu, cov, flux, spec)
        if i in (0, len(mean) // 2, len(mean) - 1):
            refined = integrated_image(mu, cov, flux, spec, quadrature=24)
            errors.append(float(np.max(np.abs(refined - expected) / np.sqrt(refined + spec["background_counts"] + spec["read_noise_counts"] ** 2))))
        counts = rng.poisson(expected + spec["background_counts"]).astype(float)
        counts += rng.normal(0., spec["read_noise_counts"], counts.shape)
        images.append(counts)
    np.savez_compressed(root / "images.npz", images=np.asarray(images))
    np.savez(root / "settings.npz", **settings)
    _write_json(root / "experiment.json", spec)
    _write_json(root / "scoring_truth.json", dict(parameters=truth.tolist()))
    _write_json(root / "generation_checks.json", dict(seed=seed, frames=len(images),
        max_pixel_quadrature_error_noise_units=max(errors), expected_order_counts=flux.tolist(),
        note="Checks numerical integration of the reduced Gaussian model, not full angular reflection."))
    return root


def _load(root):
    root = Path(root)
    spec = json.loads((root / "experiment.json").read_text())
    with np.load(root / "settings.npz") as data:
        settings = dict(data)
    return root, spec, settings


def measure_images(directory):
    """Extract order moments and their joint counting/read-noise covariance from pixels."""
    root, spec, settings = _load(directory)
    with np.load(root / "images.npz") as data:
        images = data["images"]
    nr, nc = spec["detector_shape"]
    u, v = np.meshgrid((nr - 1) / 2 - np.arange(nr), np.arange(nc) - (nc - 1) / 2, indexing="ij")
    observations, covariance, strips, doses = [], [], [], []
    for index, image in enumerate(images):
        signal = image - spec["background_counts"]
        profile = signal.sum(axis=0)
        peaks, properties = find_peaks(profile, distance=20, prominence=max(profile.max() * .03, 1.))
        if len(peaks) != 3:
            raise ValueError(f"Frame {index}: expected three separated orders, found {len(peaks)}.")
        edges = np.r_[0, (peaks[:-1] + peaks[1:]) // 2, nc]
        obs_i, cov_i, strips_i, dose_i = [], [], [], []
        for left, right in zip(edges[:-1], edges[1:]):
            sl = np.s_[:, left:right]
            intensity = signal[sl]
            total = intensity.sum()
            if total <= 0:
                raise ValueError("Nonpositive order signal.")
            uu, vv = u[sl], v[sl]
            mu, mv = (np.sum(intensity * grid) / total for grid in (uu, vv))
            vu, vvv = (np.sum(intensity * grid ** 2) / total for grid in (uu - mu, vv - mv))
            stats = np.array([mu, mv, vu, vvv])
            derivatives = np.stack((uu - mu, vv - mv, (uu - mu) ** 2 - vu, (vv - mv) ** 2 - vvv)).reshape(4, -1) / total
            # The raw count is an unbiased estimate of Poisson variance.
            # Clipping noisy dark pixels upward biases the moment uncertainty.
            noise = (image[sl] + spec["read_noise_counts"] ** 2).ravel()
            matrix = (derivatives * noise) @ derivatives.T
            np.linalg.cholesky(matrix)
            obs_i.append(stats); cov_i.append(matrix); dose_i.append(total)
            strips_i.append([left - (nc - 1) / 2, right - (nc - 1) / 2])
        observations.append(obs_i); covariance.append(cov_i); strips.append(strips_i); doses.append(dose_i)
    np.savez(root / "measurements.npz", observations=observations, covariance=covariance,
             strips=strips, detected_counts=doses)
    _write_json(root / "measurement_record.json", dict(frames=len(images),
        fingerprint=_fingerprint(root, ["images.npz", "settings.npz", "experiment.json"]),
        statistics=["mean u (pixel)", "mean v (pixel)", "variance u (pixel²)", "variance v (pixel²)"],
        note="No truth read; order intensity is a nuisance and is not fitted."))
    return root


def fit_microscope(directory, *, number_of_starts=10, max_iter=250, seed=19):
    """Bounded native JAX least squares; truth is never read here."""
    if number_of_starts < 1:
        raise ValueError("number_of_starts must be positive.")
    require_x64("the reflection calibration")
    root, spec, settings = _load(directory)
    record = json.loads((root / "measurement_record.json").read_text())
    if record["fingerprint"] != _fingerprint(root, ["images.npz", "settings.npz", "experiment.json"]):
        raise ValueError("Measurements are stale; run measure_images again.")
    with np.load(root / "measurements.npz") as data:
        observed, covariance, strips = (data[k] for k in ("observations", "covariance", "strips"))
    use = settings["purpose"] == "fit"
    predict, _ = make_predictor(spec)
    controls, dz, strips_fit = (jnp.asarray(a[use]) for a in (settings["controls"], settings["detector_dz_m"], strips))
    whiten = jnp.asarray(np.linalg.inv(np.linalg.cholesky(covariance[use])))
    target = jnp.asarray(observed[use])

    def residual(p):
        difference = predict(p, controls, dz, strips_fit) - target
        return jnp.einsum("...ij,...j->...i", whiten, difference).ravel()

    # First estimate camera distance from lateral order positions. This uses
    # the known reciprocal lattice and measured centroids, never the truth pose.
    _, _, slopes = order_geometry(spec)
    camera_response = slopes[:, 1] * 1e-3 / spec["pixel_pitch_m"]
    known_motion = (settings["detector_dz_m"][use, None] *
                    np.cos(np.deg2rad(spec["incidence_deg"])) * slopes[:, 1] /
                    spec["pixel_pitch_m"])
    weights = 1 / covariance[use, :, 1, 1]
    camera_initial = np.sum(weights * camera_response *
                            (observed[use, :, 1] - known_motion)) / np.sum(weights * camera_response ** 2)
    camera_initial = float(np.clip(camera_initial, LOWER[8], UPPER[8]))

    def coarse_residual(p):
        predicted = predict(p, controls, dz, strips_fit)
        return jnp.concatenate((.1 * (predicted[..., :2] - target[..., :2]).ravel(),
            (jnp.log(jnp.maximum(predicted[..., 2:], .1)) -
             jnp.log(jnp.maximum(target[..., 2:], .1))).ravel()))

    rng = np.random.default_rng(seed)
    starts = [INITIAL.copy()] + [rng.uniform(LOWER, UPPER) for _ in range(number_of_starts - 1)]
    coarse_runs = []
    for index, start in enumerate(starts):
        original = start.copy()
        start[8] = camera_initial
        estimate = least_squares(coarse_residual, start, bounds=(LOWER, UPPER), x_scale=SCALE,
            max_iter=max_iter, max_nfev=max_iter * 2, ftol=1e-8, geodesic=True)
        coarse_runs.append(dict(initial=original.tolist(), parameters=np.asarray(estimate.x).tolist(),
                               cost=estimate.cost, success=bool(estimate.success), status=estimate.status))
        print(f"Initialize {index + 1}/{number_of_starts}: shape cost={estimate.cost:.3f}; {estimate.status}", flush=True)
    runs = []
    # Refine the three best independent branches against the original measured
    # moments and their covariance. Initializer statistics are not counted twice.
    for index, initializer in enumerate(sorted(coarse_runs, key=lambda r: r["cost"])[:3]):
        start = np.asarray(initializer["parameters"])
        result = least_squares(residual, start, bounds=(LOWER, UPPER), x_scale=SCALE,
                               max_iter=max_iter, max_nfev=max_iter * 2,
                               ftol=1e-9, xtol=1e-9, gtol=1e-6, geodesic=True)
        entry = dict(initial=initializer["initial"], refinement_start=start.tolist(),
                     parameters=np.asarray(result.x).tolist(),
                     success=bool(result.success), status=result.status, message=result.message,
                     chi_squared=2 * result.cost, iterations=result.nit,
                     optimality=result.optimality)
        runs.append(entry)
        print(f"Refine {index + 1}: chi²={2 * result.cost:.3f}; {result.status}", flush=True)
    successful = [r for r in runs if r["success"]]
    best = min(successful or runs, key=lambda r: r["chi_squared"])
    p = np.asarray(best["parameters"])
    jacobian = np.asarray(jax.jacfwd(residual)(p))
    _, singular, vt = np.linalg.svd(jacobian * SCALE, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-8))
    pcov = (SCALE[:, None] * (vt.T / singular ** 2) @ vt) * SCALE[None, :] if rank == len(p) else None
    distances = np.minimum((p - LOWER) / (UPPER - LOWER), (UPPER - p) / (UPPER - LOWER))
    from scipy.stats import chi2
    noise_consistent = best["chi_squared"] < chi2.ppf(.999, target.size - len(p))
    result = dict(best=best, starts=runs, initialization=coarse_runs, camera_initial_mm=camera_initial,
        noise_consistent=bool(noise_consistent), names=list(NAMES), lower=LOWER.tolist(), upper=UPPER.tolist(),
        rank=rank, singular_values=singular.tolist(), weak_direction_scaled=vt[-1].tolist(),
        covariance=None if pcov is None else pcov.tolist(),
        standard_errors=None if pcov is None else np.sqrt(np.diag(pcov)).tolist(),
        near_bounds=[NAMES[i] for i in np.flatnonzero(distances < .01)],
        degrees_of_freedom=int(target.size - len(p)),
        fingerprint=_fingerprint(root, ["experiment.json", "settings.npz", "measurements.npz"]),
        successful_refinements=sum(r["success"] for r in runs),
        accepted=bool(best["success"] and noise_consistent and rank == len(p) and np.all(distances >= .01)))
    _write_json(root / "fit.json", result)
    return result


def check_microscope(directory):
    """Held-out prediction, numerical/model checks, and optional synthetic scoring."""
    root, spec, settings = _load(directory)
    result = json.loads((root / "fit.json").read_text())
    if result["fingerprint"] != _fingerprint(root, ["experiment.json", "settings.npz", "measurements.npz"]):
        raise ValueError("Fit is stale; refit the measured images.")
    with np.load(root / "measurements.npz") as data:
        observed, covariance, strips = (data[k] for k in ("observations", "covariance", "strips"))
    p = np.asarray(result["best"]["parameters"])
    predict, gaussians = make_predictor(spec)
    predicted = np.asarray(predict(p, settings["controls"], settings["detector_dz_m"], strips))
    residual = np.einsum("...ij,...j->...i", np.linalg.inv(np.linalg.cholesky(covariance)), predicted - observed)
    mu, cov = map(np.asarray, gaussians(p, settings["controls"], settings["detector_dz_m"]))
    # Gaussian probability outside each independently measured lateral strip.
    tails = ndtr((strips[..., 0] - .5 - mu[..., 1]) / np.sqrt(cov[..., 1, 1])) + ndtr((mu[..., 1] - strips[..., 1] + .5) / np.sqrt(cov[..., 1, 1]))
    # Bhattacharyya overlap bounds field overlap (before detector PSF would be smaller).
    overlaps = []
    for i in range(2):
        average = (cov[:, i] + cov[:, i + 1]) / 2
        delta = mu[:, i + 1] - mu[:, i]
        distance = np.einsum("...i,...ij,...j->...", delta, np.linalg.inv(average), delta)
        overlaps.append(np.exp(-distance / 8))
    checks = dict(held_out_frames=int(np.sum(settings["purpose"] == "check")),
        prediction_rms_noise_units=float(np.sqrt(np.mean(residual[settings["purpose"] == "check"] ** 2))),
        fit_rms_noise_units=float(np.sqrt(np.mean(residual[settings["purpose"] == "fit"] ** 2))),
        max_lateral_tail_fraction=float(tails.max()), max_order_field_overlap_bound=float(np.max(overlaps)),
        rank=result["rank"], near_bounds=result["near_bounds"],
        objective_working_distance_mm=float(-p[3]),
        note="Local errors condition on the known source model, incidence, detector pose/PSF and surface response.")
    checks["separated_order_approximation_passed"] = bool(tails.max() < 1e-7 and np.max(overlaps) < 1e-7)
    from scipy.stats import chi2
    held = residual[settings["purpose"] == "check"]
    checks["prediction_noise_consistent"] = bool(held.size and np.sum(held ** 2) < chi2.ppf(.999, held.size))
    checks["validated"] = bool(result["accepted"] and checks["separated_order_approximation_passed"]
                               and checks["prediction_noise_consistent"])
    truth_path = root / "scoring_truth.json"
    if truth_path.exists():
        truth = np.asarray(json.loads(truth_path.read_text())["parameters"])
        checks.update(truth=truth.tolist(), errors=(p - truth).tolist())
    _write_json(root / "checks.json", checks)
    np.savez(root / "predictions.npz", predicted=predicted, residual=residual)
    return checks


def _models(root):
    models = {"Initial guess": INITIAL.copy()}
    path = Path(root) / "fit.json"
    if path.exists():
        result = json.loads(path.read_text())
        models["Fitted model"] = np.asarray(result["best"]["parameters"])
    truth_path = Path(root) / "scoring_truth.json"
    if truth_path.exists():
        models["Ground truth"] = np.asarray(json.loads(truth_path.read_text())["parameters"])
    return models


def _draw_column(ax, parameters, controls, dz, spec, title):
    from .plotting import plot_reflection_column, trace_gaussian_basis

    components = column(parameters, controls)
    trace = trace_gaussian_basis(components, 2 * parameters[7] * 1e-9,
                                voltage=spec["voltage_v"], z=parameters[0] * 1e-3)
    incoming, outgoing = frames(spec)
    directions, _, _ = order_geometry(spec)
    centre = parameters[8] * 1e-3 * np.asarray(outgoing.normal_xyz) + np.array([0., 0., dz])
    screen = PlaneFrame(origin_xyz=jnp.asarray(centre), u_xyz=outgoing.u_xyz, v_xyz=outgoing.v_xyz)
    hits = np.asarray(screen.intersect_line(jnp.zeros(3), jnp.asarray(directions))[0])
    paths = np.stack((np.zeros_like(hits), hits), axis=1)[..., [0, 2]]
    half_height = spec["detector_shape"][0] * spec["pixel_pitch_m"] / 2
    ends = centre + np.array([-1., 1.])[:, None] * half_height * np.asarray(outgoing.u_xyz)
    plot_reflection_column(components, trace, sample_z=0., outgoing_paths_xz=paths,
        detector_segment_xz=ends[:, [0, 2]], column_frame=incoming,
        component_labels=("C1", "C2", "Objective", "Surface"), specimen_label="Cu(001)",
        envelope_scale=60., title=title, show_section_labels=False, ax=ax)
    ax.text(.015, .98,
        f"Working distance {(-parameters[3]):.3f} mm · initial camera {parameters[8]:.3f} mm\n"
        f"Reference f: {parameters[4]:.3f}, {parameters[5]:.3f}, {parameters[6]:.3f} mm\n"
        "Incident envelope ×60; component positions use physical axes",
        transform=ax.transAxes, va="top", fontsize=8, color="#425466")


def _draw_image(ax, image, mean, cov, spec, *, maximum):
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Ellipse
    nr, nc = image.shape
    pitch_mm = spec["pixel_pitch_m"] * 1e3
    ax.imshow(np.maximum(image, .1), origin="upper", cmap="gray", norm=LogNorm(1., maximum),
              extent=np.array([-nc, nc, -nr, nr]) * pitch_mm / 2)
    for m, matrix, order in zip(mean, cov, spec["orders"]):
        # Plot horizontal v and vertical u, both in detector millimetres.
        eig, vectors = np.linalg.eigh(matrix[np.ix_([1, 0], [1, 0])])
        angle = np.rad2deg(np.arctan2(vectors[1, 1], vectors[0, 1]))
        ax.add_patch(Ellipse(m[::-1] * pitch_mm, 4 * np.sqrt(eig[1]) * pitch_mm,
                            4 * np.sqrt(eig[0]) * pitch_mm, angle=angle,
                            fill=False, lw=1.1, color="#E69F00"))
        ax.annotate(f"{order:+d}", m[::-1] * pitch_mm, xytext=(5, 6),
                    textcoords="offset points", color="#E69F00", fontsize=8)
    ax.set(xlabel="Detector lateral v (mm)", ylabel="Detector u (mm)",
           title="Saved counts · model 2σ contours")


def plot_comparison(directory, *, row=0, models=("Initial guess", "Fitted model"), export=False):
    """Saved detector pixels beside native reflection ray diagrams; common axes."""
    import matplotlib.pyplot as plt
    from .plotting import export_ray_figure

    root, spec, settings = _load(directory)
    available = _models(root)
    with np.load(root / "images.npz") as data:
        image = data["images"][row]
    _, predict = make_predictor(spec)
    fig, axes = plt.subplots(len(models), 2, figsize=(15, 4.3 * len(models)),
                             squeeze=False, gridspec_kw={"width_ratios": [1, 2.25]}, layout="constrained")
    for ax, name in zip(axes, models):
        p = available[name]
        mean, cov = map(np.asarray, predict(p, settings["controls"][row:row + 1], settings["detector_dz_m"][row:row + 1]))
        _draw_image(ax[0], image, mean[0], cov[0], spec, maximum=max(image.max(), 2.))
        _draw_column(ax[1], p, settings["controls"][row], settings["detector_dz_m"][row], spec, name)
    xlim = (min(a.get_xlim()[0] for a in axes[:, 1]), max(a.get_xlim()[1] for a in axes[:, 1]))
    ylim = (min(a.get_ylim()[0] for a in axes[:, 1]), max(a.get_ylim()[1] for a in axes[:, 1]))
    for ax in axes[:, 1]:
        ax.set(xlim=xlim, ylim=ylim)
    controls = settings["controls"][row]
    fig.suptitle(f"Record {row + 1} · {settings['purpose'][row]} · relative powers "
                 f"{controls[0]:.3f}, {controls[1]:.3f}, {controls[2]:.3f} · "
                 f"known ΔZ = {settings['detector_dz_m'][row] * 1e3:+.1f} mm", fontsize=12)
    if export:
        export_ray_figure(fig, root / "initial_and_fitted_column", dpi=160,
                          metadata=dict(record=row, models=list(models), envelope_scale=60))
    return fig


def browse_images(directory):
    """The calibration notebook's saved-image/model-selector form in reflection geometry."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    root, spec, settings = _load(directory)
    available = _models(root)
    with np.load(root / "images.npz") as data:
        images = data["images"]
    _, predict = make_predictor(spec)
    with plt.ioff():
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained",
                                 gridspec_kw={"width_ratios": [1, 2.1]})
    if not isinstance(fig.canvas, widgets.Widget):
        plt.close(fig)
        raise RuntimeError("Use %matplotlib widget before browse_images; static plot_comparison also works headlessly.")
    purpose = widgets.Dropdown(description="Experiment", options=["All records", "fit", "check"])
    model = widgets.Dropdown(description="Ray model", options=list(available), value="Initial guess")
    row = widgets.SelectionSlider(description="Image", options=[(str(i + 1), i) for i in range(len(images))],
                                  continuous_update=False, layout=widgets.Layout(width="75%"))
    summary = widgets.HTML()
    busy = False

    def update(event=None):
        if busy:
            return
        i, p = row.value, available[model.value]
        for ax in axes:
            ax.clear()
        mu, cov = map(np.asarray, predict(p, settings["controls"][i:i + 1], settings["detector_dz_m"][i:i + 1]))
        _draw_image(axes[0], images[i], mu[0], cov[0], spec, maximum=max(float(images.max()), 2.))
        _draw_column(axes[1], p, settings["controls"][i], settings["detector_dz_m"][i], spec, model.value)
        axes[1].set(xlim=(-530, 230), ylim=(-30, 20))
        command = settings["controls"][i]
        summary.value = (f"<b>Record {i + 1}/{len(images)} · {settings['purpose'][i]}</b> · "
            f"C1/C2/objective power: {command[0]:.3f}, {command[1]:.3f}, {command[2]:.3f} · "
            f"known laboratory ΔZ: {settings['detector_dz_m'][i] * 1e3:+.1f} mm. "
            "The saved image stays fixed when the model changes.")
        fig.canvas.draw_idle()

    def filter_rows(event=None):
        nonlocal busy
        busy = True
        ids = np.flatnonzero((purpose.value == "All records") | (settings["purpose"] == purpose.value))
        row.options = [(str(i + 1), int(i)) for i in ids]
        busy = False
        update()

    purpose.observe(filter_rows, names="value")
    model.observe(update, names="value")
    row.observe(update, names="value")
    update()
    fig.canvas.draw()
    return widgets.VBox([widgets.HTML("<h3>Explore the reflection calibration</h3>"),
        widgets.HBox([purpose, model]), row, summary, fig.canvas],
        layout=widgets.Layout(width="100%", max_width="1400px", padding="16px", border="1px solid #dce3e8"))


def show_result(directory):
    """Reuse the microscope calibration notebook's geometry table formatter."""
    from .calibration.viewer import show_result as show_calibration_result

    root = Path(directory)
    result = json.loads((root / "fit.json").read_text())
    checks = json.loads((root / "checks.json").read_text())
    truth = checks.get("truth", [None] * len(NAMES))
    error = checks.get("errors", [None] * len(NAMES))
    std = result["standard_errors"] or [None] * len(NAMES)
    geometry = [dict(parameter=name, fitted=p, truth=t, error=e, standard_deviation=s)
                for name, p, t, e, s in zip(NAMES, result["best"]["parameters"], truth, error, std)]
    print(f"Local rank {result['rank']}/{len(NAMES)}; near bounds: {result['near_bounds'] or 'none'}")
    print(f"Fit accepted: {result['accepted']}; prediction/approximation checks: {checks['validated']}")
    return show_calibration_result(checks=dict(geometry=geometry,
        fit_residual_rms=checks["fit_rms_noise_units"], prediction_residual_rms=checks["prediction_rms_noise_units"]))


def plot_diagnostics(directory):
    import matplotlib.pyplot as plt
    root, _, settings = _load(directory)
    result = json.loads((root / "fit.json").read_text())
    with np.load(root / "predictions.npz") as data:
        residual = data["residual"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    axes[0].semilogy(np.arange(1, 10), result["singular_values"], "o-", color="#269F83")
    axes[0].set(xlabel="Sensitivity direction", ylabel="Scaled singular value", title="Local identifiability")
    for purpose, color in (("fit", "#269F83"), ("check", "#E69F00")):
        use = settings["purpose"] == purpose
        axes[1].plot(np.flatnonzero(use) + 1, np.sqrt(np.mean(residual[use] ** 2, axis=(1, 2))),
                     ".", label=purpose, color=color)
    axes[1].axhline(1., color=".5", lw=.7)
    axes[1].set(xlabel="Record", ylabel="RMS / measurement noise", title="Unused settings test prediction")
    axes[1].legend()
    if result["covariance"] is not None:
        cov = np.asarray(result["covariance"])
        correlation = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        artist = axes[2].imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
        axes[2].set(xticks=np.arange(9), yticks=np.arange(9),
                    xticklabels=["zs", "z1", "z2", "zo", "f1", "f2", "fo", "σs", "L"],
                    yticklabels=["zs", "z1", "z2", "zo", "f1", "f2", "fo", "σs", "L"], title="Parameter correlation")
        fig.colorbar(artist, ax=axes[2], shrink=.7)
    return fig
