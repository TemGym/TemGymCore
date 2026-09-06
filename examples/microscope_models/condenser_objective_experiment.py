"""Objective-current experiment with independently measured MAGICAL image features.

The projector model is calibrated at objective command one. At other commands
the image analysis measures the pattern in nominal specimen coordinates; it
does not receive the actual post-field response or condenser.
"""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_calibration_model import specimen_moments
from scipy.optimize import least_squares

from temgym_core.gaussian_image_likelihood import normal_interval_probability
from temgym_core.microscope_inversion import drift_matrix, hard_edge_solenoid_matrix


def round_matrix(vector):
    """A two-number representation of an isotropic scale and rotation."""
    x, y = vector
    return jnp.array([[x, -y], [y, x]])


def post_field_change(projector_parameters, objective_command, response):
    """Return E with P(u)=P(1) E(u), for unchanged downstream lens commands.

    response = [strength slope, length slope, extra rotation slope]. All changes
    vanish at u=1, preserving the previous calibration exactly. The nominal
    solenoid response is [1, 0, 0]. Length changes keep its centre fixed.
    """
    p = jnp.asarray(projector_parameters)
    delta = objective_command - 1.0
    length0, gap0 = jnp.exp(p[6]), jnp.exp(p[0])
    length = length0 * jnp.exp(response[1] * delta)
    entrance = gap0 + (length0 - length) / 2
    reference = hard_edge_solenoid_matrix(p[11], length0) @ drift_matrix(gap0)
    changed = (
        drift_matrix((length0 - length) / 2)
        @ hard_edge_solenoid_matrix(p[11] * (1 + response[0] * delta), length)
        @ drift_matrix(entrance)
    )
    order = jnp.array([0, 2, 1, 3])
    difference = jnp.linalg.solve(reference, changed)[jnp.ix_(order, order)]
    rotation = round_matrix(
        jnp.array([jnp.cos(response[2] * delta), jnp.sin(response[2] * delta)])
    )
    rotation4 = jnp.zeros((4, 4)).at[:2, :2].set(rotation).at[2:, 2:].set(rotation)
    return rotation4 @ difference


def predict_pattern(
    parameters,
    condenser_controls,
    objective_command,
    projector_parameters,
    kernel_sigma,
    response,
):
    """Fourteen image quantities; no exposure parameters or generating truth.

    Three illumination-envelope quantities, two complex pattern-map columns,
    four offsets and three covariance quantities describe the transmitted
    Gaussian-mixture specimen in nominal specimen coordinates.
    """
    p = jnp.asarray(parameters)
    changed = p[:19].at[12].add(jnp.log(objective_command))
    mean, covariance = specimen_moments(changed, condenser_controls)
    variance = covariance[0, 0]
    h = covariance[0, 2] / variance
    conditional = jnp.exp(2 * (p[17] + p[18])) / variance
    kernel_variance = kernel_sigma**2
    alpha = variance / (variance + kernel_variance)
    product_variance = alpha * kernel_variance
    nominal = post_field_change(
        projector_parameters, objective_command, jnp.array([1.0, 0.0, 0.0])
    )
    actual = post_field_change(projector_parameters, objective_command, response)
    e = jnp.linalg.solve(nominal, actual)
    column = e[:, :2] + h * e[:, 2:]
    mapping = alpha * column
    offset = e @ jnp.r_[(1 - alpha) * mean[:2], mean[2:] - alpha * h * mean[:2]]
    out_cov = product_variance * (column @ column.T) + conditional * (
        e[:, 2:] @ e[:, 2:].T
    )
    out_variance = out_cov[0, 0]
    # The round symplectic map preserves the radial covariance determinant.
    out_conditional = product_variance * conditional / out_variance
    return jnp.r_[
        mean[:2] / 1e-7,
        0.5 * jnp.log(variance),
        mapping[:2, 0],
        mapping[2:, 0] / 1e4,
        offset[:2] / 1e-7,
        offset[2:] / 1e-3,
        0.5 * jnp.log(out_variance),
        out_cov[0, 2] / out_variance / 1e4,
        0.5 * jnp.log(out_conditional),
    ]


def make_pattern_image_operator(*, projector_matrices, template, detector):
    """Render measured pattern quantities through reference maps, not hidden optics."""
    matrices = jnp.asarray(projector_matrices)
    a, b = matrices[:, :2, :2], matrices[:, :2, 2:4]
    axis = jnp.asarray(template["axis_m"])
    xx, yy = jnp.meshgrid(axis, axis)
    centres = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
    transmission = jnp.asarray(template["intensity"]).ravel()
    kernel_variance = detector["sample_kernel_sigma_m"] ** 2
    pitch = detector["pixel_pitch_m"]
    ny, nx = detector["shape_yx"]
    xe, ye = jnp.arange(nx + 1) - nx / 2, jnp.arange(ny + 1) - ny / 2

    def render(p, projector_matrices_override=None):
        if projector_matrices_override is None:
            a_used, b_used = a, b
        else:
            a_used = projector_matrices_override[:, :2, :2]
            b_used = projector_matrices_override[:, :2, 2:4]
        variance = jnp.exp(2 * p[2])
        log_weights = jnp.log(transmission) - jnp.sum(
            (centres - p[:2] * 1e-7) ** 2, axis=1
        ) / (2 * (variance + kernel_variance))
        weights = jax.nn.softmax(log_weights)
        maps = a_used @ round_matrix(p[3:5]) + b_used @ round_matrix(p[5:7] * 1e4)
        offsets = a_used @ (p[7:9] * 1e-7) + b_used @ (p[9:11] * 1e-3)
        xy = (jnp.einsum("nij,kj->nki", maps, centres) + offsets[:, None]) / pitch
        covariance_maps = a_used + p[12] * 1e4 * b_used
        detector_variance = (
            jnp.exp(2 * p[11]) * jnp.sum(covariance_maps[:, :, 0] ** 2, axis=1)
            + jnp.exp(2 * p[13]) * jnp.sum(b_used[:, :, 0] ** 2, axis=1)
        ) / pitch**2
        widths = jnp.sqrt(detector_variance + detector["psf_sigma_pixels"] ** 2)

        def frame(centres, width, exposure):
            px = normal_interval_probability(
                (xe[:-1, None] - centres[:, 0]) / width,
                (xe[1:, None] - centres[:, 0]) / width,
            )
            py = normal_interval_probability(
                (ye[:-1, None] - centres[:, 1]) / width,
                (ye[1:, None] - centres[:, 1]) / width,
            )
            return jnp.exp(exposure[0]) * ((py * weights) @ px.T) + jnp.exp(exposure[1])

        return jax.vmap(frame)(xy, widths, p[14:].reshape(-1, 2))

    return jax.jit(render)


def _signature(paths, settings):
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for path in paths:
        digest.update(Path(path).read_bytes())
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()


def choose_objective_projector_views(views, objective_command):
    """Keep the five useful nominal conjugates by adjusting downstream lenses.

    Setup uses the previously calibrated model with its nominal q(u) law. It
    never uses the hidden response slopes. Actual images may differ from these
    planned maps, which is why the image analysis measures a free pattern.
    """
    from finite_projector_pipeline import SIGNS, matrix5

    p = jnp.asarray(views["calibrated_projector_parameters"])
    matrix = jax.jit(matrix5)
    planned, commands = [], []
    for view in views["views"]:
        magnification, target_b = view["target"]
        start = np.asarray(view["controls"][1:])

        def error(v, magnification=magnification, target_b=target_b):
            command = jnp.r_[objective_command, v]
            m = matrix(p, command)
            angle = p[11] * objective_command + jnp.dot(SIGNS * jnp.exp(p[12:16]), v)
            rotation = round_matrix(jnp.array([jnp.cos(angle), jnp.sin(angle)]))
            aa = rotation @ m[:2, :2]
            bb = rotation @ m[:2, 2:4]
            return jnp.array(
                [
                    aa[0, 0] / magnification - 1,
                    (bb[0, 0] - target_b) / (magnification * 1e-4),
                ]
            )

        objective = jax.jit(error)
        derivative = jax.jit(jax.jacfwd(error))
        fit = least_squares(
            lambda v, objective=objective: np.asarray(objective(v)),
            start,
            jac=lambda v, derivative=derivative: np.asarray(derivative(v)),
            bounds=(0.05, 4.5),
            max_nfev=500,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        if np.linalg.norm(fit.fun) > 1e-7:
            # Two target coefficients leave freedom in four lens currents.
            # A nearby starting point can escape a turning point of one branch.
            for direction in np.r_[np.eye(4), -np.eye(4)]:
                trial = least_squares(
                    lambda v, objective=objective: np.asarray(objective(v)),
                    np.clip(start * (1 + 0.03 * direction), 0.050001, 4.499999),
                    jac=lambda v, derivative=derivative: np.asarray(derivative(v)),
                    bounds=(0.05, 4.5),
                    max_nfev=1000,
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12,
                )
                if np.linalg.norm(trial.fun) < np.linalg.norm(fit.fun):
                    fit = trial
                if np.linalg.norm(fit.fun) <= 1e-7:
                    break
            if np.linalg.norm(fit.fun) > 1e-7:
                raise RuntimeError(
                    f"Could not keep projector target A={magnification}, B={target_b}; residual {np.linalg.norm(fit.fun):.3g}."
                )
        command = np.r_[objective_command, fit.x]
        commands.append(command)
        planned.append(np.asarray(matrix(p, command)))
    return np.asarray(planned), np.asarray(commands)


def make_objective_images(
    *,
    reference_image_directory,
    projector_views_path,
    output_directory,
    reuse_completed=True,
):
    """Supply baseline acquisition/calibration; return noisy images at new currents.

    Only this generating function uses the hidden condenser and post response.
    Six condenser settings at 98%/102% are fitted; three other settings at
    97%/103% are reserved for a frozen prediction check.
    """
    from condenser_calibration_model import generating_parameters

    reference, folder = Path(reference_image_directory), Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    signature = _signature(
        [
            reference / "images.npz",
            reference / "template.npz",
            reference / "detector.json",
            projector_views_path,
        ],
        {},
    )
    record_path = folder / "generation.json"
    if (
        reuse_completed
        and record_path.exists()
        and json.loads(record_path.read_text())["signature"] == signature
    ):
        return folder
    views = json.loads(Path(projector_views_path).read_text())
    projector = np.asarray(views["calibrated_projector_parameters"])
    matrices = np.asarray([v["matrix"] for v in views["views"]])
    detector = json.loads((reference / "detector.json").read_text())
    template = dict(np.load(reference / "template.npz"))
    p = generating_parameters()
    # A modest hidden departure from the nominal extrapolation at u=1.
    response = np.array([1.08, 0.20, 0.03])
    planned_views = {
        u: choose_objective_projector_views(views, u) for u in [0.97, 0.98, 1.02, 1.03]
    }
    candidate_controls = np.load(reference / "images.npz")["controls"]
    patterns = jax.jit(
        lambda c, u: predict_pattern(
            p, c, u, projector, detector["sample_kernel_sigma_m"], response
        )
    )
    rng = np.random.default_rng(7301)

    def visible(currents):
        candidates = []
        for i, c in enumerate(candidate_controls):
            widths = [float(jnp.exp(patterns(c, u)[2])) for u in currents]
            if min(widths) > 12e-9 and max(widths) < 125e-9:
                candidates.append(i)
        return candidates

    prediction_indices = rng.permutation(visible([0.97, 1.03]))[:3]
    calibration_indices = rng.permutation(
        [i for i in visible([0.98, 1.02]) if i not in prediction_indices]
    )[:6]
    if len(calibration_indices) < 6 or len(prediction_indices) < 3:
        raise RuntimeError(
            "Not enough settings keep the specimen visible across the sweep."
        )
    operator = make_pattern_image_operator(
        projector_matrices=matrices, template=template, detector=detector
    )
    for split, indices, currents in [
        ("calibration", calibration_indices, [0.98, 1.02]),
        ("prediction", prediction_indices, [0.97, 1.03]),
    ]:
        (
            images,
            commands,
            objectives,
            reference_indices,
            nominal_maps,
            projector_commands,
        ) = [], [], [], [], [], []
        for i in indices:
            for u in currents:
                pattern = patterns(candidate_controls[i], u)
                exposures = np.tile(np.log([4e7, 3.0]), (len(matrices), 1)).ravel()
                maps, lens_commands = planned_views[u]
                mean = np.asarray(
                    operator(jnp.r_[pattern, exposures], jnp.asarray(maps))
                )
                images.append(
                    rng.poisson(mean).astype(float)
                    + rng.normal(0, detector["read_noise_e"], mean.shape)
                )
                commands.append(candidate_controls[i])
                objectives.append(u)
                reference_indices.append(i)
                nominal_maps.append(maps)
                projector_commands.append(lens_commands)
        np.savez_compressed(
            folder / f"{split}_images.npz",
            images=images,
            controls=commands,
            objective_commands=objectives,
            reference_indices=reference_indices,
            projector_matrices=nominal_maps,
            projector_controls=projector_commands,
        )
    np.savez_compressed(folder / "template.npz", **template)
    (folder / "detector.json").write_text(json.dumps(detector, indent=2))
    (folder / "reference_projector.json").write_text(json.dumps(views, indent=2))
    (folder / "scoring_truth.json").write_text(
        json.dumps(
            {
                "condenser_parameters": p.tolist(),
                "post_response": response.tolist(),
                "seed": 7301,
            },
            indent=2,
        )
    )
    record_path.write_text(
        json.dumps(
            {
                "signature": signature,
                "calibration_images": 60,
                "prediction_images": 30,
                "selection": "Simulation setup chooses visible beams; fitting receives only recorded commands.",
            },
            indent=2,
        )
    )
    return folder


def pattern_from_measured_beam(beam, kernel_sigma):
    """Initial pattern guess from a previously measured beam at u=1."""
    b = np.asarray(beam)
    variance = np.exp(2 * b[4])
    alpha = variance / (variance + kernel_sigma**2)
    return np.r_[
        b[:2],
        b[4],
        alpha,
        0.0,
        alpha * b[5],
        0.0,
        (1 - alpha) * b[:2],
        b[2:4] - alpha * b[5] * b[:2],
        0.5 * np.log(alpha * kernel_sigma**2),
        b[5],
        b[6],
    ]


def measure_objective_images(
    *,
    image_directory,
    reference_measurements_path,
    split="calibration",
    reuse_completed=True,
    max_evaluations=250,
):
    """Supply pixels/reference maps; return 14 measured pattern quantities and covariance.

    No condenser geometry, changed projector maps or hidden response are read.
    Reference image measurements supply initial guesses only.
    """
    folder = Path(image_directory)
    output = folder / f"{split}_measurements.npz"
    signature = _signature(
        [
            folder / f"{split}_images.npz",
            folder / "template.npz",
            folder / "detector.json",
            folder / "reference_projector.json",
            reference_measurements_path,
        ],
        {"max_evaluations": max_evaluations},
    )
    if (
        reuse_completed
        and output.exists()
        and str(np.load(output)["signature"]) == signature
    ):
        return output
    data = np.load(folder / f"{split}_images.npz")
    detector = json.loads((folder / "detector.json").read_text())
    views = json.loads((folder / "reference_projector.json").read_text())
    maps = np.asarray([v["matrix"] for v in views["views"]])
    operator = make_pattern_image_operator(
        projector_matrices=maps,
        template=dict(np.load(folder / "template.npz")),
        detector=detector,
    )
    derivative = jax.jit(jax.jacfwd(operator))
    reference = np.load(reference_measurements_path)["values"]
    lower = np.r_[
        [-3, -3],
        np.log(3e-9),
        [-10, -10, -40, -40],
        [-10, -10, -10, -10],
        np.log(0.02e-9),
        -100,
        np.log(1e-8),
        np.tile(np.log([1e5, 0.01]), len(maps)),
    ]
    upper = np.r_[
        [3, 3],
        np.log(250e-9),
        [10, 10, 40, 40],
        [10, 10, 10, 10],
        np.log(100e-9),
        100,
        np.log(0.02),
        np.tile(np.log([1e10, 100]), len(maps)),
    ]
    records, covariance, residual_rms, full_parameters = [], [], [], []
    for index, (frames, ref_index) in enumerate(
        zip(data["images"], data["reference_indices"])
    ):
        noise = np.sqrt(np.maximum(frames, 1.0) + detector["read_noise_e"] ** 2)
        current_maps = jnp.asarray(data["projector_matrices"][index])
        residual = lambda p, current_maps=current_maps, frames=frames, noise=noise: (
            (np.asarray(operator(p, current_maps)) - frames) / noise
        ).ravel()
        jacobian = lambda p, current_maps=current_maps, noise=noise: (
            np.asarray(derivative(p, current_maps)).reshape(-1, len(p))
            / noise.ravel()[:, None]
        )
        initial = pattern_from_measured_beam(
            reference[ref_index], detector["sample_kernel_sigma_m"]
        )
        initial = np.r_[initial, np.tile(np.log([4e7, 3.0]), len(maps))]
        # First register the specimen-focused view. This determines the
        # illumination envelope and spatial pattern without a condenser model.
        spatial_free = np.array([0, 1, 2, 3, 4, 7, 8, 11, 14, 15])
        pixel_count = frames[0].size
        spatial_seeds = []
        for size_nm in [20.0, 60.0, 120.0]:
            for scale in [0.85, 1.0, 1.15]:
                guess = initial.copy()
                guess[2] = np.log(size_nm * 1e-9)
                guess[3] = scale
                spatial_seeds.append(
                    (np.mean(residual(guess)[:pixel_count] ** 2), guess)
                )
        spatial_fits = []
        for _, guess in sorted(spatial_seeds, key=lambda item: item[0])[:2]:

            def unpack(v, guess=guess, spatial_free=spatial_free):
                p = guess.copy()
                p[spatial_free] = v
                return p

            fit = least_squares(
                lambda v, residual=residual, unpack=unpack, pixel_count=pixel_count: (
                    residual(unpack(v))[:pixel_count]
                ),
                guess[spatial_free],
                jac=lambda v, jacobian=jacobian, unpack=unpack, pixel_count=pixel_count, spatial_free=spatial_free: (
                    jacobian(unpack(v))[:pixel_count, spatial_free]
                ),
                bounds=(lower[spatial_free], upper[spatial_free]),
                x_scale="jac",
                max_nfev=150,
                ftol=1e-9,
                xtol=1e-10,
                gtol=1e-5,
            )
            spatial_fits.append((fit.cost, unpack(fit.x)))
        initial = min(spatial_fits, key=lambda item: item[0])[1]
        candidates = []
        for curvature in np.linspace(-10, 10, 801):
            guess = initial.copy()
            guess[5] = curvature * guess[3]
            guess[12] = curvature
            candidates.append((np.mean(residual(guess) ** 2), guess))
        fits = []
        for _, guess in sorted(candidates, key=lambda x: x[0])[:2]:
            fits.append(
                least_squares(
                    residual,
                    np.clip(guess, lower + 1e-9, upper - 1e-9),
                    jac=jacobian,
                    bounds=(lower, upper),
                    x_scale="jac",
                    max_nfev=max_evaluations,
                    ftol=1e-9,
                    xtol=1e-10,
                    gtol=1e-5,
                )
            )
        fit = min(fits, key=lambda f: f.cost)
        rms = float(np.sqrt(np.mean(fit.fun**2)))
        if not fit.success or rms > 1.5:
            (folder / f"{split}_failed_analysis.json").write_text(
                json.dumps(
                    {
                        "index": index,
                        "fit": fit.x.tolist(),
                        "initial": initial.tolist(),
                        "spatial_chi_squared": [float(2 * x[0]) for x in spatial_fits],
                        "candidate_rms": [
                            float(np.sqrt(x[0]))
                            for x in sorted(candidates, key=lambda x: x[0])[:5]
                        ],
                        "rms": rms,
                    },
                    indent=2,
                )
            )
            raise RuntimeError(
                f"Objective image analysis failed at {split} {index}: RMS {rms}, {fit.message}"
            )
        _, s, vt = np.linalg.svd(fit.jac, full_matrices=False)
        cov = (vt.T / s) @ (vt / s[:, None])
        records.append(fit.x[:14])
        covariance.append(cov[:14, :14])
        residual_rms.append(rms)
        full_parameters.append(fit.x)
        print(
            f"{split} pattern {index + 1}/{len(data['images'])}: pixel RMS {rms:.3f}, nfev {fit.nfev}",
            flush=True,
        )
        np.savez_compressed(
            folder / f"{split}_analysis_progress.npz",
            values=records,
            covariance=covariance,
        )
    np.savez_compressed(
        output,
        controls=data["controls"],
        objective_commands=data["objective_commands"],
        values=records,
        covariance=covariance,
        pixel_residual_rms=residual_rms,
        image_parameters=full_parameters,
        signature=signature,
    )
    return output
