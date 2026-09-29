"""Projector-only inversion: specimen → detector, from paired steering images.

Only the five projector solenoids (objective post-field, IL1, IL2, IL3, PL1)
and the detector are modelled. A small Gaussian beam starts at the specimen
with a calibrated shift or tilt; native TemGym rays carry it to the detector.
Equal and opposite kicks give a detector displacement that depends only on the
A and B blocks of the projector matrix, which the fit inverts for 16 lens
positions, lengths and strengths.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from temgym_core.calibration.inference import PRESETS, fit_projector_responses
from temgym_core.calibration.projector_model import (
    LENSES,
    SIGNS,
    at_voltage,
    decode_geometry,
    matrix5,
    model_to_parameters,
    parameters_to_model,
    projector_components,
    projector_coordinates,
)
from temgym_core.components import Detector
from temgym_core.moments import measure_detector_moments, moment_regions
from temgym_core.ray import Ray

HERE = Path(__file__).resolve().parent
ACQUISITION = HERE / "calibration_results" / "tilt_y_30_ray_acquisition"


@dataclass
class ProjectorData:
    """The projector rows of a saved acquisition and their paired responses."""

    spec: dict
    settings: dict
    truth: np.ndarray
    reference_controls: np.ndarray
    images: np.ndarray | None = None
    response: dict | None = None


def load_acquisition(directory=ACQUISITION):
    """Keep only projector-calibration rows; the condenser settings are unused.

    Each projector setting has three ± pairs (shift X, tilt X, tilt Y). The
    truth parameters are read for simulation and scoring, never for fitting.
    """
    root = Path(directory)
    spec = json.loads((root / "experiment.json").read_text())
    with np.load(root / "settings.npz") as data:
        selected = data["purpose"] == "projector"
        settings = {
            key: data[key][selected]
            for key in (
                "voltage_index", "projector_controls", "specimen_kick",
                "nominal_magnification",
            )
        }
        # The unperturbed imaging setting at the first voltage, for focal lengths.
        reference = np.flatnonzero(
            ~selected
            & (data["voltage_index"] == 0)
            & (data["nominal_magnification"] == spec["nominal_magnification"])
        )[0]
        reference_controls = data["projector_controls"][reference]
        # Each wobble perturbs the imaging setting of its voltage and nominal
        # magnification; the figure reports changes from that baseline.
        settings["baseline_controls"] = np.array([
            data["projector_controls"][np.flatnonzero(
                ~selected & (data["voltage_index"] == v)
                & (data["nominal_magnification"] == m)
            )[0]]
            for v, m in zip(settings["voltage_index"], settings["nominal_magnification"])
        ])
    kicks = settings["specimen_kick"]
    if not np.allclose(kicks[1::2], -kicks[::2]):
        raise ValueError("Projector rows must alternate between + and − kicks.")
    truth = json.loads((root / "scoring_truth.json").read_text())
    return ProjectorData(
        spec, settings, np.asarray(truth["projector_parameters"]), reference_controls
    )


def _render(shape, pixel_pitch_m, psf_sigma_pixels):
    """Native rays through projector components, binned with detector blur."""
    detector = Detector(z=0.0, shape=shape, pixel_size=(pixel_pitch_m, pixel_pitch_m))

    @jax.jit
    def render(parameters, controls, voltage, samples, blur, weights):
        from temgym_core.run import run_to_end

        rays = Ray(*samples.T, z=0.0, pathlength=0.0, voltage=voltage)
        out = run_to_end(rays, projector_components(parameters, controls))
        x = out.x + blur[:, 0] * psf_sigma_pixels * pixel_pitch_m
        y = out.y + blur[:, 1] * psf_sigma_pixels * pixel_pitch_m
        iy, ix = detector.metres_to_pixels((x, y))
        inside = (iy >= 0) & (iy < shape[0]) & (ix >= 0) & (ix < shape[1])
        index = jnp.where(inside, iy * shape[1] + ix, 0)
        counts = jnp.bincount(index, weights=inside * weights, length=shape[0] * shape[1])
        return counts.reshape(shape)

    return render


def simulate_images(
    data,
    *,
    beam_rms_m=10e-9,
    angle_rms_rad=0.03e-3,
    electrons_per_image=10_000_000,
    random_seed=31415,
):
    """Render one noisy detector image per projector row.

    The specimen beam is a round Gaussian of ``beam_rms_m`` position and
    ``angle_rms_rad`` angular spread, centred on the calibrated kick. Every
    ray is one detected electron: a Poisson number is traced per image and
    blurred individually by the detector PSF, so the image carries exactly
    counting noise. Background and read noise follow the saved acquisition.
    """
    spec, settings = data.spec, data.settings
    shape = tuple(spec["shape_yx"])
    render = _render(shape, spec["pixel_pitch_m"], spec["psf_sigma_pixels"])
    rng = np.random.default_rng(random_seed)
    spread = np.array([beam_rms_m, beam_rms_m, angle_rms_rad, angle_rms_rad])
    voltages = spec["voltages_v"]
    images = np.empty((len(settings["voltage_index"]), *shape))
    # A fixed padded ray count avoids recompiling for each Poisson draw.
    size = int(electrons_per_image + 8 * np.sqrt(electrons_per_image))
    for i, (vi, controls, kick) in enumerate(
        zip(settings["voltage_index"], settings["projector_controls"], settings["specimen_kick"])
    ):
        samples = kick + spread * rng.standard_normal((size, 4))
        electrons = np.arange(size) < rng.poisson(electrons_per_image)
        counts = np.asarray(
            render(
                jnp.asarray(at_voltage(data.truth, voltages[vi], voltages[0])),
                jnp.asarray(controls),
                voltages[vi],
                jnp.asarray(samples),
                jnp.asarray(rng.standard_normal((size, 2))),
                jnp.asarray(electrons, dtype=float),
            )
        )
        images[i] = counts + rng.poisson(spec["background_counts"], shape) + rng.normal(
            0, spec["read_noise_counts"], shape
        )
    data.images = images
    return data


def measure_responses(data):
    """Beam centres from pixels, then half the difference of each ± pair."""
    spec = data.spec
    regions = moment_regions(
        images=data.images,
        background_counts=spec["background_counts"],
        read_noise_counts=spec["read_noise_counts"],
        margin_pixels=spec.get("moment_margin_pixels", 16),
    )
    measured = measure_detector_moments(
        images=data.images,
        background_counts=spec["background_counts"],
        read_noise_counts=spec["read_noise_counts"],
        regions=regions,
    )
    centres, covariance = measured["values"][:, :2], measured["covariance"][:, :2, :2]
    settings = data.settings
    data.response = {
        "values": (centres[::2] - centres[1::2]) / 2,
        "covariance": (covariance[::2] + covariance[1::2]) / 4,
        "controls": settings["projector_controls"][::2],
        "voltage_index": settings["voltage_index"][::2],
        "specimen_kicks": settings["specimen_kick"][::2],
        "centres": centres,
    }
    return data


def fit_projector(data, *, number_of_starts=4, output_directory, starting_model=None,
                  name="projector"):
    """Fit 16 projector coordinates to the paired responses alone.

    ``starting_model`` replaces the preset engineering guess; extra starts
    are dispersed around whichever guess is used.
    """
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    fit = fit_projector_responses(
        response=data.response,
        voltages_v=data.spec["voltages_v"],
        pixel_pitch_m=data.spec["pixel_pitch_m"],
        number_of_starts=number_of_starts,
        output_path=output / f"{name}_starts.json",
        prior_model=starting_model,
    )
    _, decode, _, _ = projector_coordinates(PRESETS / "projector_position_priors.json")
    fit["projector_parameters"] = np.asarray(
        decode(jnp.asarray(fit["best"]["parameters"]))
    ).tolist()
    (output / f"{name}_fit.json").write_text(json.dumps(fit, indent=2) + "\n")
    return fit


def focus_guess(model, data):
    """Set a guess's four gains so the imaging setting is in focus at its label.

    An operator focuses the image at the unperturbed setting, and that
    setting's nominal magnification is known, so a guessed column can be made
    to agree with both before fitting: specimen tilts do not move the image
    (B = 0) and the magnification equals the label. The four gains change as
    little as possible; positions, lengths and objective strength are kept.
    """
    from scipy.optimize import least_squares

    p0 = jnp.asarray(model_to_parameters(model))
    controls = jnp.asarray(data.reference_controls)
    magnification = data.spec["nominal_magnification"]

    @jax.jit
    def residual(log_gains):
        m = matrix5(p0.at[12:16].set(log_gains), controls)
        scale = jnp.sqrt(jnp.abs(jnp.linalg.det(m[:2, :2])))
        return jnp.r_[scale / magnification - 1, m[:2, 2:4].ravel() / 1e-3,
                      1e-2 * (log_gains - p0[12:16])]

    jacobian = jax.jit(jax.jacfwd(residual))
    fit = least_squares(lambda g: np.asarray(residual(jnp.asarray(g))), np.asarray(p0[12:16]),
                        jac=lambda g: np.asarray(jacobian(jnp.asarray(g))))
    gains = np.exp(fit.x)
    return dict(model, excitation_gains=dict(zip(LENSES[1:], gains.tolist())))


# Alternative starting guesses: lens and detector positions as if read from
# drawings (truth: 2.3, 213.5, 290.5, 353.0, 408.0 and 718.75 mm), then gains
# set by ``focus_guess``.
_GUESS_OPTICS = {
    "lens_lengths_mm": {"Objective": 0.42, "IL1": 5.2, "IL2": 4.8, "IL3": 5.2, "PL1": 1.05},
    "objective_strength": -0.43,
    "excitation_gains": {"IL1": 0.56, "IL2": 0.64, "IL3": 0.46, "PL1": 0.74},
}
GUESS_POSITIONS = {
    "Close guess": {
        "lens_positions_mm": {"Objective": 2.4, "IL1": 215.0, "IL2": 292.0,
                              "IL3": 351.0, "PL1": 410.0},
        "detector_position_mm": 720.0,
    },
    "Poor positions": {
        "lens_positions_mm": {"Objective": 2.35, "IL1": 183.5, "IL2": 310.5,
                              "IL3": 328.0, "PL1": 438.0},
        "detector_position_mm": 758.75,
    },
}


def fit_guesses(data, *, output_directory, guesses=None):
    """Fit from each guess alone (one start), to show where each converges.

    By default: the preset engineering guess, and each ``GUESS_POSITIONS``
    entry focused at the imaging setting by ``focus_guess``.
    """
    if guesses is None:
        guesses = {
            "Engineering guess": json.loads(
                (PRESETS / "projector_starting_guess.json").read_text()
            ),
            **{name: focus_guess({**_GUESS_OPTICS, **positions}, data)
               for name, positions in GUESS_POSITIONS.items()},
        }
    return {
        name: {"starting_model": model,
               "fit": fit_projector(data, number_of_starts=1, starting_model=model,
                                    output_directory=output_directory,
                                    name=name.lower().replace(" ", "_"))}
        for name, model in guesses.items()
    }


def guess_table(data, fit, guesses):
    """How far each guess puts the beam, and where its fit ends up."""
    from IPython.display import Markdown

    settings, voltages = data.settings, data.spec["voltages_v"]
    maps = jax.jit(jax.vmap(lambda p, c, v: matrix5(at_voltage(p, v, voltages[0]), c),
                            in_axes=(None, 0, 0)))
    volts = jnp.asarray(np.array(voltages)[settings["voltage_index"]])
    controls = jnp.asarray(settings["projector_controls"])

    def centres(p):
        m = np.asarray(maps(jnp.asarray(p), controls, volts))
        return np.einsum("nij,nj->ni", m[:, :2, :4], settings["specimen_kick"]) * 1e3

    truth = centres(data.truth)
    half_width = data.spec["shape_yx"][1] * data.spec["pixel_pitch_m"] * 0.5e3
    true_model = parameters_to_model(data.truth)
    rows = {"Engineering guess": {
        "starting_model": json.loads((PRESETS / "projector_starting_guess.json").read_text()),
        "fit": fit,
    }, **guesses}
    lines = [
        "| Initial guess | Position error, RMS (mm) | Beam miss, median (mm) | Beam on camera |"
        " Fit RMS (σ) | Fitted detector error (mm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, entry in rows.items():
        start = entry["starting_model"]
        error = np.r_[
            [start["lens_positions_mm"][k] - true_model["lens_positions_mm"][k]
             for k in LENSES],
            start["detector_position_mm"] - true_model["detector_position_mm"],
        ]
        guess = centres(model_to_parameters(start))
        miss = np.hypot(*(guess - truth).T)
        on = (np.abs(guess).max(axis=1) < half_width).mean()
        fitted = parameters_to_model(np.asarray(entry["fit"]["projector_parameters"]))
        lines.append(
            f"| {name} | {np.sqrt(np.mean(error**2)):.1f} | {np.median(miss):.2f} | "
            f"{on:.0%} | {entry['fit']['best']['residual_rms']:.2f} | "
            f"{fitted['detector_position_mm'] - true_model['detector_position_mm']:+.2f} |"
        )
    return Markdown("\n".join(lines))


def _named(parameters, controls):
    """Positions, thick-lens focal lengths at ``controls``, strength and gains.

    A hard-edge solenoid of length l and strength q has 1/f = q sin q / l.
    """
    _, lengths, centres, detector = decode_geometry(parameters)
    q = jnp.r_[parameters[11] * controls[0], SIGNS * jnp.exp(parameters[12:16]) * controls[1:]]
    return jnp.r_[
        centres * 1000, detector * 1000, lengths / (q * jnp.sin(q)) * 1000,
        parameters[11], jnp.exp(parameters[12:16]),
    ]


NAMED_ROWS = (
    [f"{name} position (mm)" for name in LENSES]
    + ["Detector position (mm)"]
    + [f"{name} focal length (mm)" for name in LENSES]
    + ["Objective strength (rad)"]
    + [f"{name} gain" for name in LENSES[1:]]
)


def parameter_table(fit, data):
    """Fitted, truth, error and local standard deviation for 16 named quantities.

    Focal lengths are at the unperturbed 40,000× setting and first voltage;
    IL1–PL1 focal lengths change with every lens current.
    """
    from IPython.display import Markdown

    _, decode, _, _ = projector_coordinates(PRESETS / "projector_position_priors.json")
    u = jnp.asarray(fit["best"]["parameters"])
    controls = jnp.asarray(data.reference_controls)
    fitted = np.asarray(_named(decode(u), controls))
    expected = np.asarray(_named(jnp.asarray(data.truth), controls))
    guess = np.asarray(_named(jnp.asarray(model_to_parameters(
        json.loads((PRESETS / "projector_starting_guess.json").read_text())
    )), controls))
    sigma = np.full(len(fitted), np.nan)
    if fit["covariance"] is not None:
        jacobian = np.asarray(jax.jacfwd(lambda v: _named(decode(v), controls))(u))
        sigma = np.sqrt(np.diag(jacobian @ np.asarray(fit["covariance"]) @ jacobian.T))
    lines = [
        f"Fit RMS **{fit['best']['residual_rms']:.3f}** noise units · "
        f"rank {fit['best']['rank']}/{fit['best']['parameter_count']} · "
        f"acceptable **{fit['best']['acceptable']}**\n",
        "| Parameter | Initial guess | Fitted | Truth | Error | Std. dev. | Error / σ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, g, f, t, s in zip(NAMED_ROWS, guess, fitted, expected, sigma):
        lines.append(
            f"| {name} | {g:.5g} | {f:.6g} | {t:.6g} | {f - t:.3g} | {s:.3g} | {(f - t) / s:+.2f} |"
        )
    return Markdown("\n".join(lines))


def save_experiment(data, fit, directory, *, guesses=None, beam_rms_m=10e-9,
                    angle_rms_rad=0.03e-3):
    """Write images, settings, truth and fit in the calibration browser's format.

    ``calibration.viewer.browse_images`` then shows this projector-only
    experiment with the same controls and ray diagram as the full pipeline.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    spec = dict(data.spec, column="projector", aperture_diameters_um=[],
                specimen_beam_rms=[beam_rms_m, beam_rms_m, angle_rms_rad, angle_rms_rad])
    (root / "experiment.json").write_text(json.dumps(spec, indent=2) + "\n")
    settings = data.settings
    np.savez_compressed(
        root / "settings.npz",
        voltage_index=settings["voltage_index"],
        projector_controls=settings["projector_controls"],
        specimen_kick=settings["specimen_kick"],
        nominal_magnification=settings["nominal_magnification"],
        baseline_controls=settings["baseline_controls"],
        purpose=np.full(len(settings["voltage_index"]), "projector"),
    )
    np.savez_compressed(root / "images.npz", images=data.images[:, None])
    (root / "scoring_truth.json").write_text(
        json.dumps({"projector_parameters": np.asarray(data.truth).tolist()}) + "\n"
    )
    (root / "fit.json").write_text(json.dumps(fit, indent=2) + "\n")
    (root / "guesses.json").write_text(json.dumps({
        name: {"starting_parameters": model_to_parameters(g["starting_model"]).tolist(),
               "projector_parameters": g["fit"]["projector_parameters"]}
        for name, g in (guesses or {}).items()
    }, indent=2) + "\n")
    return root


MODELS = {  # panel title → colour
    "Ground truth": "#222222",
    "Initial guess": "#D55E00",
    "Fitted model": "#0072B2",
}
STEERING = ("Shift X", "Tilt X", "Tilt Y")
RAY_VIEWS = ("Specimen → objective", "Specimen → detector", "Projector lenses",
             "Detector")
FIGURE_RC = {"font.size": 16, "axes.titlesize": 19, "axes.labelsize": 17,
             "xtick.labelsize": 14, "ytick.labelsize": 14}


class Experiment:
    """A saved projector experiment, read once for figures."""

    def __init__(self, directory):
        from temgym_core.calibration.inference import PRESETS as presets

        root = Path(directory)
        self.spec = json.loads((root / "experiment.json").read_text())
        with np.load(root / "settings.npz") as data:
            self.settings = dict(data)
        with np.load(root / "images.npz") as data:
            self.images = data["images"][:, 0]
        self.truth = json.loads((root / "scoring_truth.json").read_text())
        # Each guess: its starting parameters and the fit that began there.
        self.guesses = {"Engineering guess": (
            model_to_parameters(
                json.loads((presets / "projector_starting_guess.json").read_text())
            ),
            json.loads((root / "fit.json").read_text())["projector_parameters"],
        )}
        if (root / "guesses.json").exists():
            for name, g in json.loads((root / "guesses.json").read_text()).items():
                self.guesses[name] = (g["starting_parameters"], g["projector_parameters"])
        kicks = self.settings["specimen_kick"]
        self.steering = np.array([STEERING[int(np.argmax(np.abs(k[[0, 2, 3]])))]
                                  for k in kicks])
        self.positive = kicks.sum(axis=1) > 0
        # Lens settings are numbered per voltage in acquisition order.
        self.setting = np.zeros(len(kicks), int)
        for v in np.unique(self.settings["voltage_index"]):
            rows = np.flatnonzero(self.settings["voltage_index"] == v)
            self.setting[rows] = np.arange(len(rows)) // (2 * len(STEERING))

    def row(self, voltage, setting, steering):
        """The + kick image for one voltage, lens setting and beam input."""
        return int(np.flatnonzero(
            (self.settings["voltage_index"] == voltage) & (self.setting == setting)
            & (self.steering == steering) & self.positive
        )[0])

    def models(self, guess):
        """Truth, the chosen initial guess, and the fit that started from it."""
        start, fitted = self.guesses[guess]
        return {"Ground truth": self.truth,
                "Initial guess": {"projector_parameters": start},
                "Fitted model": {"projector_parameters": fitted}}

    def centre_mm(self, parameters, row):
        """Model beam centre on the detector, from the native ray matrix."""
        voltages = self.spec["voltages_v"]
        v = voltages[self.settings["voltage_index"][row]]
        m = np.asarray(matrix5(
            jnp.asarray(at_voltage(parameters, v, voltages[0])),
            jnp.asarray(self.settings["projector_controls"][row]),
        ))
        return (m[:2, :4] @ self.settings["specimen_kick"][row] + m[:2, 4]) * 1e3

    def measured_mm(self, row):
        frame = self.images[row][None]
        regions = moment_regions(images=frame, background_counts=self.spec["background_counts"],
                                 read_noise_counts=self.spec["read_noise_counts"],
                                 margin_pixels=16)
        values = measure_detector_moments(
            images=frame, background_counts=self.spec["background_counts"],
            read_noise_counts=self.spec["read_noise_counts"], regions=regions,
        )["values"][0]
        return values[:2] * self.spec["pixel_pitch_m"] * 1e3


def draw_figure(fig, experiment, *, voltage=0, setting=0, steering="Tilt X", view=1,
                guess="Engineering guess"):
    """Recorded image, then truth, initial-guess and fitted ray diagrams.

    All three diagrams share the z axis and one transverse scale wide enough
    for every model's beam, so an offset guess is seen landing elsewhere. Dashed grey lines mark the true
    lens and detector planes. Drawn with the calibration browser's
    ``_draw_ray_section``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from temgym_core.calibration.viewer import (
        _draw_ray_section,
        _projector_sections,
        _ray_diagram,
    )

    e = experiment
    row = e.row(voltage, setting, steering)
    models = e.models(guess)
    s = e.settings
    fig.clear()
    with plt.rc_context(FIGURE_RC):
        axes = fig.subplots(1, 4, width_ratios=[1.15, 1, 1, 1],
                            gridspec_kw={"wspace": 0.08})
        image_ax, ray_axes = axes[0], axes[1:]

        # (a) Recorded beam, centred on the measured spot.
        frame = e.images[row]
        pitch = e.spec["pixel_pitch_m"] * 1e3
        h, w = frame.shape
        image_ax.imshow(np.maximum(frame, 1), cmap="gray", origin="upper",
                        norm=LogNorm(1, max(frame.max(), 2)),
                        extent=(-w / 2 * pitch, w / 2 * pitch, -h / 2 * pitch, h / 2 * pitch))
        measured = e.measured_mm(row)
        # Widen the view (up to ±8 mm) to include a nearby guessed centre.
        offsets = [np.abs(e.centre_mm(models[m]["projector_parameters"], row) - measured).max()
                   for m in ("Initial guess", "Fitted model")]
        near = [o for o in offsets if 1.25 * o <= 8]
        span = float(np.clip(1.25 * max(near, default=0), 2.5, 8))
        image_ax.set(xlim=(measured[0] - span, measured[0] + span),
                     ylim=(measured[1] - span, measured[1] + span),
                     xlabel="Detector x (mm)", ylabel="Detector y (mm)", aspect="equal")
        image_ax.plot(*measured, "+", color="#F0E442", ms=26, mew=3.5, label="Measured")
        for model in ("Initial guess", "Fitted model"):
            centre = e.centre_mm(models[model]["projector_parameters"], row)
            offset = centre - measured
            if np.abs(offset).max() < span:
                image_ax.plot(*centre, "o" if model == "Fitted model" else "x",
                              mfc="none", color=MODELS[model], ms=18, mew=3, label=model)
            else:
                tip = measured + offset / np.abs(offset).max() * 0.92 * span
                image_ax.annotate(
                    f"{np.hypot(*offset):.0f} mm", tip, xytext=measured + (tip - measured) * 0.45,
                    color=MODELS[model], fontsize=16, fontweight="bold", ha="center",
                    arrowprops={"arrowstyle": "-|>", "color": MODELS[model], "lw": 2.5,
                                "mutation_scale": 22},
                )
                image_ax.plot([], [], "x", color=MODELS[model], ms=14, mew=3, label=model)
        image_ax.set_facecolor("black")
        fig.legend(*image_ax.get_legend_handles_labels(), loc="outside lower left",
                   ncol=3, fontsize=15, frameon=False,
                   title="Beam centre in (a):", title_fontsize=15,
                   alignment="left")
        image_ax.set_title("(a)  Recorded beam", loc="left")

        # (b–d) One ray diagram per model at the same kick and currents.
        diagrams = {m: _ray_diagram(e.spec, s, models[m], row, 0) for m in MODELS}
        truth_column, _ = diagrams["Ground truth"]
        all_segments = [piece for _, segments in diagrams.values() for piece in segments]
        truth_planes = [float(c.z + getattr(c, "length", 0) / 2) for c in truth_column]
        # One z window and transverse scale for all three, from the truth.
        bounds = _projector_sections(truth_column)["section_bounds"]
        bottom = max(float(column[-1].z) for column, _ in diagrams.values())
        bounds[1] = (bounds[1][0], bottom)
        bounds[3] = (bounds[3][0], bottom)
        for ax, (letter, (model, colour)) in zip(ray_axes, zip("bcd", MODELS.items())):
            column, segments = diagrams[model]
            options = dict(_projector_sections(column), section_bounds=bounds)
            _draw_ray_section(ax, column, segments, view, scale_segments=all_segments,
                              label_fontsize=14, **options)
            if model != "Ground truth":
                lo, hi = sorted(ax.get_ylim())
                for z in truth_planes:
                    if lo <= z * 1000 <= hi:
                        ax.axhline(z * 1000, color="0.55", lw=1.3, ls="--", zorder=-1)
            title = f"({letter})  {model}"
            if model == "Initial guess":
                title += f"\n       {guess.lower()}"
            elif model == "Fitted model":
                title += "\n       from that guess"
            ax.set_title(title, loc="left", color=colour, fontweight="bold")
            if letter != "b":
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
        top, low = ray_axes[0].get_ylim()
        for ax in ray_axes:
            ax.set_ylim(top, low)

        kick = s["specimen_kick"][row]
        amount = (f"shift x = {kick[0] * 1e9:+.0f} nm" if steering == "Shift X"
                  else f"tilt {steering[-1].lower()} = {kick[2 + (steering == 'Tilt Y')] * 1e3:+.1f} mrad")
        change = 100 * (s["projector_controls"][row] / s["baseline_controls"][row] - 1)
        currents = ",  ".join(f"{n} {c:+.0f}%" for n, c in zip(LENSES[1:], change[1:]))
        fig.suptitle(
            f"{e.spec['voltages_v'][s['voltage_index'][row]] / 1000:g} kV  ·  lens setting "
            f"{setting + 1}:  {currents}  ·  specimen {amount}",
            x=0.01, ha="left", fontsize=17,
        )
    return fig


def figure_browser(experiment_directory):
    """A minimal control strip above the publication figure."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    e = Experiment(experiment_directory)
    with plt.ioff():
        fig = plt.figure(figsize=(18, 7.6), layout="constrained")
    fig.canvas.header_visible = fig.canvas.footer_visible = False
    fig.canvas.toolbar_visible = False
    fig.canvas.layout = widgets.Layout(width="100%", height="auto")
    style = {"description_width": "initial"}
    voltage = widgets.ToggleButtons(
        options=[(f"{v / 1000:g} kV", i) for i, v in enumerate(e.spec["voltages_v"])],
        style=style)
    steering = widgets.ToggleButtons(options=STEERING, value="Tilt X", style=style)
    setting = widgets.IntSlider(value=18, min=1, max=int(e.setting.max()) + 1,
                                description="Lens setting", continuous_update=False,
                                style=style)
    view = widgets.Dropdown(options=[(v, i) for i, v in enumerate(RAY_VIEWS)], value=1,
                            description="Ray view", style=style)
    guess = widgets.ToggleButtons(options=list(e.guesses), style=style)
    if "Poor positions" in e.guesses:
        guess.value = "Poor positions"
    save = widgets.Button(description="Save PDF", icon="download")
    status = widgets.Label()

    def state():
        return dict(voltage=voltage.value, setting=setting.value - 1,
                    steering=steering.value, view=view.value, guess=guess.value)

    def update(_=None):
        draw_figure(fig, e, **state())
        fig.canvas.draw_idle()

    def write(_):
        from matplotlib.figure import Figure

        path = HERE / "figures" / "projector_inversion.pdf"
        path.parent.mkdir(exist_ok=True)
        paper = Figure(figsize=(18, 7.6), layout="constrained")
        draw_figure(paper, e, **state())
        paper.savefig(path)
        paper.savefig(path.with_suffix(".png"), dpi=300)
        status.value = f"Saved figures/{path.name} and .png"

    for control in (voltage, steering, setting, view, guess):
        control.observe(update, names="value")
    save.on_click(write)
    update()
    return widgets.VBox([
        widgets.HBox([voltage, steering]),
        guess,
        widgets.HBox([setting, view, save, status],
                     layout=widgets.Layout(align_items="center", grid_gap="16px")),
        fig.canvas,
    ])


COLUMN_LABELS = ["Objective", "IL1", "IL2", "IL3", "PL1", "Detector"]


def _beam_rays(voltage, half_width_m):
    """The recorded beam's footprint: parallel rays across the specimen."""
    x = np.linspace(-half_width_m, half_width_m, 21)
    zero = np.zeros_like(x)
    return Ray(x=x, y=zero, dx=zero, dy=zero, z=0.0, pathlength=0.0, voltage=voltage)


def _length(value_m):
    for scale, unit in ((1e-3, "mm"), (1e-6, "µm"), (1e-9, "nm")):
        if value_m >= scale:
            return f"{value_m / scale:.2g} {unit}"
    return f"{value_m * 1e9:.2g} nm"


def _column_panel(ax, parameters, controls, voltage, reference, *, xmax, title, colour,
                  bottom, detector_half_width, fill="#43C78E", truth_planes=(),
                  half_width_m=20e-9):
    """TemGym's ray-model look for the recorded beam, magnified down the column.

    The transverse axis is linear within ±``half_width_m`` and logarithmic
    beyond, so a 40 nm beam at the specimen and its ~1.6 mm image on the
    detector are both visible; pinches are crossovers.
    """
    from matplotlib.patches import Ellipse
    from matplotlib.transforms import blended_transform_factory

    from temgym_core.plotting import legacy_beam_plot_params, plot_model, trace_ray_paths

    column = projector_components(jnp.asarray(at_voltage(parameters, voltage, reference)),
                                  jnp.asarray(controls))
    rays = _beam_rays(voltage, half_width_m)
    style = legacy_beam_plot_params(fixed_xmax=xmax, fill_color=fill, fill_alpha=0.6,
                                    interior_ray_alpha=0.0, center_ray_alpha=0.0,
                                    ray_alpha=0.6, edge_lw=1.2, show_side_guides=False)
    plot_model(column, rays=rays, ax=ax, ray_coordinate="x_rot", plot_params=style)
    # Keep only the beam; lenses and labels are redrawn for the non-linear axis.
    for artist in [*ax.texts, *ax.patches]:
        artist.remove()
    ax.set_xscale("asinh", linear_width=half_width_m)
    ax.set(xlim=(-xmax, xmax), ylim=(bottom, -0.03 * bottom), xticks=[], yticks=[])
    ax.set_xticks([], minor=True)
    ax.grid(False, which="both")
    across = blended_transform_factory(ax.transAxes, ax.transData)
    lens_z = [float(c.z + c.length / 2) for c in column[:-1]]
    detector = float(column[-1].z)
    for z in lens_z:
        ax.add_patch(Ellipse((0.5, z), 0.96, 0.018 * bottom, transform=across, fill=False,
                             lw=1.4, color="0.15", zorder=999))
    ax.plot([-detector_half_width, detector_half_width], [detector] * 2, color="0.35",
            lw=5, solid_capstyle="butt", zorder=999)  # the camera's true width
    for z in truth_planes:  # where the true lenses and detector are
        ax.plot([0.0, 1.0], [z] * 2, transform=across, color="0.35", lw=1.3,
                ls=(0, (4, 3)), zorder=998)
    # Labels to the right, kept apart where planes crowd.
    positions = [*lens_z, detector]
    for i in range(1, len(positions)):
        positions[i] = max(positions[i], positions[i - 1] + 0.05 * bottom)
    for z, name in zip(positions, COLUMN_LABELS):
        ax.text(1.02, z, name, transform=across, fontsize=17, va="center")
    ax.plot(0, 0, "o", color="#795548", ms=8, zorder=1001, clip_on=False)
    _, x = trace_ray_paths(column, rays, ray_coordinate="x_rot")
    for z, text in ((0.0, _length(2 * half_width_m)), (detector, _length(np.ptp(x[-1])))):
        ax.text(0.0, z, text, transform=across, fontsize=15, color="0.35", va="center",
                ha="right")
    ax.set_title(title, fontsize=21, fontweight="bold", color=colour, pad=16, loc="left")
    return float(np.abs(x[-1]).max())


def _kick_icon(ax, kind, label):
    """Schematic of one change: a lens made stronger and weaker, or the beam
    displaced or tilted (about a point) at the sample plane."""
    from matplotlib.patches import Ellipse, Polygon

    ax.set(xlim=(-1, 1), ylim=(1.05, -1.05), xticks=[], yticks=[])
    ax.axis("off")
    if kind == "lens":
        width = 0.35
        ax.add_patch(Polygon([(-width, -1), (width, -1), (width, -0.3), (-width, -0.3)],
                             facecolor="#43C78E", alpha=0.75, edgecolor="0.25", lw=0.8))
        for focus, alpha in ((0.45, 0.75), (0.95, 0.35)):  # stronger, weaker
            ax.add_patch(Polygon([(-width, -0.3), (width, -0.3), (0, focus)],
                                 facecolor="#43C78E", alpha=alpha, edgecolor="0.25",
                                 lw=0.8))
        ax.add_patch(Ellipse((0, -0.3), 1.5, 0.22, fill=False, lw=1.4, color="0.15"))
        ax.text(0, 1.3, label, ha="center", va="top", fontsize=15, color="0.2")
        return
    ax.plot([-0.95, 0.95], [0, 0], color="0.45", lw=1.6)  # sample plane
    half, angle = 0.1, np.radians(16)
    for sign, alpha in ((1, 0.75), (-1, 0.35)):
        if kind == "shift":
            direction, offset = np.array([0.0, 1.0]), np.array([0.3 * sign, 0.0])
        else:
            direction = np.array([np.sin(sign * angle), np.cos(sign * angle)])
            offset = np.zeros(2)
        normal = np.array([direction[1], -direction[0]])
        ends = [offset + t * direction for t in (-1.0, 1.0)]
        corners = [ends[0] + half * normal, ends[1] + half * normal,
                   ends[1] - half * normal, ends[0] - half * normal]
        ax.add_patch(Polygon(corners, facecolor="#43C78E", alpha=alpha, edgecolor="0.25",
                             lw=0.8))
    if kind == "tilt":
        ax.plot(0, 0, "o", color="#795548", ms=7, zorder=5)  # pivot on the sample plane
    ax.text(0, 1.3, label, ha="center", va="top", fontsize=15, color="0.2")


def draw_overview(fig, experiment, *, guess="Poor positions", voltage=0, grid=3):
    """Ground truth → recorded images → initial guess → fitted model.

    The columns show the recorded beam at the unperturbed imaging setting;
    the grid shows beam images at wobbled lens currents with shifts and
    tilts, all on one detector field of view. That is all the fit sees.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import FancyArrowPatch

    e = experiment
    s = e.settings
    voltages = e.spec["voltages_v"]
    v = voltages[voltage]
    start, fitted = e.guesses[guess]
    truth = e.truth["projector_parameters"]
    reference = s["baseline_controls"][np.flatnonzero(s["voltage_index"] == voltage)[0]]
    fig.clear()

    panels = {"(a)  Ground truth": (0.05, truth, "#222222", "#43C78E"),
              "(c)  Initial guess": (0.58, start, MODELS["Initial guess"], "#F5A35C"),
              "(d)  Fitted model": (0.82, fitted, MODELS["Fitted model"], "#6BAED6")}
    _, _, centres, detector = decode_geometry(jnp.asarray(truth))
    truth_planes = [*np.asarray(centres).tolist(), float(detector)]
    bottom = 1.03 * max(float(decode_geometry(jnp.asarray(p))[3])
                        for _, p, _, _ in panels.values())
    # One transverse scale, just wider than the camera.
    detector_half_width = e.spec["shape_yx"][1] * e.spec["pixel_pitch_m"] / 2
    for title, (x0, parameters, colour, fill) in panels.items():
        ax = fig.add_axes([x0, 0.08, 0.12, 0.8])
        _column_panel(ax, parameters, reference, v, voltages[0],
                      xmax=1.06 * detector_half_width, title=title, colour=colour,
                      bottom=bottom, detector_half_width=detector_half_width, fill=fill,
                      truth_planes=truth_planes if parameters is not truth else ())

    # (b) Recorded images on one detector field of view.
    fig.text(0.265, 0.905, "(b)  Recorded images", fontsize=21, fontweight="bold")
    plus = np.flatnonzero(e.positive & (s["voltage_index"] == voltage))
    spots = np.array([e.measured_mm(row) for row in plus])
    view = 6.0  # mm half-width of every thumbnail
    candidates = np.flatnonzero(np.abs(spots).max(axis=1) < 0.8 * view)
    chosen = [candidates[np.argmax(np.hypot(*spots[candidates].T))]]
    while len(chosen) < grid * grid:  # spread the spots across the detector
        distance = np.min([np.hypot(*(spots[candidates] - spots[c]).T) for c in chosen],
                          axis=0)
        chosen.append(candidates[np.argmax(distance)])
    pitch = e.spec["pixel_pitch_m"] * 1e3
    h, w = e.images.shape[1:]
    size = 0.072
    aspect = fig.get_figwidth() / fig.get_figheight()
    for k, c in enumerate(chosen):
        row = plus[c]
        card = fig.add_axes([0.268 + (k % grid) * (size + 0.008),
                             0.675 - (k // grid) * (size + 0.008) * aspect,
                             size, size * aspect])
        card.imshow(np.maximum(e.images[row], 1), cmap="gray", origin="upper",
                    norm=LogNorm(1, e.images[row].max()),
                    extent=(-w / 2 * pitch, w / 2 * pitch, -h / 2 * pitch, h / 2 * pitch))
        card.set(xlim=(-view, view), ylim=(-view, view), xticks=[], yticks=[])
        for spine in card.spines.values():
            spine.set_visible(False)
    kicks = s["specimen_kick"][plus]
    change = np.abs(s["projector_controls"][plus][:, 1:]
                    / s["baseline_controls"][plus][:, 1:] - 1).max()
    # Two groups: what changes in the projector, and what changes at the sample.
    for x0, kind, label in (
            (0.268, "lens", f"currents ±{100 * change:.0f}%"),
            (0.375, "shift", f"shift ±{np.abs(kicks[:, 0]).max() * 1e9:.0f} nm"),
            (0.46, "tilt", f"tilt ±{np.abs(kicks[:, 2:]).max() * 1e3:.0f} mrad")):
        _kick_icon(fig.add_axes([x0, 0.1, 0.07, 0.13]), kind, label)
    for x, text in ((0.303, "projector lenses"), (0.452, "beam at the sample plane")):
        fig.text(x, 0.255, text, ha="center", fontsize=15, fontweight="bold", color="0.25")
    fig.add_artist(plt.Line2D([0.355, 0.355], [0.06, 0.27], transform=fig.transFigure,
                              color="0.75", lw=1.2))

    for x0, x1, label in ((0.222, 0.258, "record"), (0.515, 0.57, "fit"),
                          (0.765, 0.8, "")):
        fig.patches.append(FancyArrowPatch((x0, 0.5), (x1, 0.5), transform=fig.transFigure,
                                           arrowstyle="simple", mutation_scale=36,
                                           color="0.6"))
        fig.text((x0 + x1) / 2, 0.46, label, ha="center", va="top", fontsize=15,
                 color="0.3")
    return fig
