"""Build the virtual NEOARM and run the microscope calibration through PyJEM calls.

The hidden optics are the selected synthetic acquisition's truth
(``calibration_results/tilt_y_30_ray_acquisition/scoring_truth.json``). The hidden
electronics map 16-bit words to lens controls and kicks, and the camera is ideal.
TEM MAG has the 40k and 45k rows; TEM DIFF has camera lengths 0.15-1 m designed on
the hidden optics (:func:`design_diffraction_table`).
Lens headroom is *relaxed* here: each lens's largest planned control maps to
``headroom_word`` (60000 by default), so the selected plan can be executed. The
recorded NEOARM words leave PL1 only 2.4% headroom; with that, the plan is
rejected (see ``src/temgym_core/virtual/README.md``).

The protocol (``temgym_core.calibration.protocol``) only uses PyJEM calls and the
vendor calibration; the truth is used afterwards, to score the fit.

Run from the repository root (GPU recommended for the full fidelity)::

    JAX_PLATFORMS=cpu .venv/bin/python examples/microscope_models/virtual_neoarm.py --quick
    .venv/bin/python examples/microscope_models/virtual_neoarm.py --rays 22 --subsamples 32
"""

import argparse
import json
import shutil
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from scipy.constants import elementary_charge  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from temgym_core.calibration import check_microscope, fit_microscope, measure_images  # noqa: E402
from temgym_core.calibration.projector_model import at_voltage, matrix5  # noqa: E402
from temgym_core.calibration.protocol import acquire_calibration  # noqa: E402
from temgym_core.virtual import (  # noqa: E402
    CameraModel,
    DeflectorDrive,
    Electronics,
    LensDrive,
    OpticalTruth,
    VirtualMicroscope,
    install,
    vendor_calibration,
)

HERE = Path(__file__).parent
ACQUISITION = HERE / "calibration_results" / "tilt_y_30_ray_acquisition"
PRESETS = HERE.parents[1] / "src" / "temgym_core" / "calibration" / "presets"
OUTPUT = HERE.parent / "generated" / "virtual_neoarm"
PROJECTOR = ("IL1", "IL2", "IL3", "PL1")
CONDENSER = ("CL1", "CL3", "CM")
REFERENCE_SPOT = 2                        # PyJEM index of "spot 3"
APERTURE_HOLES = {50.0: 1, 20.0: 2}       # condenser-aperture hole numbers
DOSE_PER_IMAGE = 1e8                      # incident electrons, as in the synthetic acquisition
EXPOSURE_S = 0.01
#: DIFF selector rows: camera length (m) at this camera, as JEOL's nominal steps.
CAMERA_LENGTHS_M = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


def load_plan():
    """The selected acquisition's settings and its reference (gauge) controls."""
    plan = {k: np.asarray(v) for k, v in np.load(ACQUISITION / "settings.npz").items()}
    spec = json.loads((ACQUISITION / "experiment.json").read_text())
    protocol = json.loads((PRESETS / "beam_measurement_protocol.json").read_text())
    baseline45 = plan["projector_controls"][(plan["purpose"] == "check") & (plan["voltage_index"] == 0)
                                            & (plan["nominal_magnification"] == 45000.0)][0]
    return plan, spec, {40000.0: np.asarray(protocol["projector_reference_controls"])[1:],
                        45000.0: baseline45[1:]}, np.asarray(spec["condenser_reference_controls"])


def design_diffraction_table(truth, drives, mag_words, camera_lengths_m=CAMERA_LENGTHS_M, *, max_word=63000):
    """TEM DIFF rows for the hidden optics: words that put the back focal plane on the camera.

    For each camera length L: PL1 stays at its MAG value; IL1, IL2 and IL3 are chosen so that
    A = 0 (a shift at the specimen does not move the beam) and |B| = L (m/rad, at 200 kV).
    That leaves one free direction, taken closest to the MAG row in IL2 and IL3, so IL1 does
    the switching, as in a JEOL DIFF mode. Rows are solved in increasing L, each starting
    from the last, and kept within ``max_word``.

    Parameters
    ----------
    truth : OpticalTruth
    drives : dict
        Lens name -> :class:`LensDrive`.
    mag_words : dict
        The MAG row the DIFF rows start from (words for IL1, IL2, IL3, PL1).
    camera_lengths_m : sequence of float
    max_word : int

    Returns
    -------
    dict
        Camera length (m) -> words for IL1, IL2, IL3, PL1.
    """
    pp = at_voltage(np.asarray(truth.projector_parameters), truth.voltages_v[0], truth.voltages_v[0])
    ray_map = jax.jit(lambda c: matrix5(pp, c))
    base = np.array([drives[n].control(mag_words[n]) for n in PROJECTOR])
    per_unit = np.array([drives[n].words_per_unit for n in PROJECTOR[:3]])

    def a_b(controls):
        m = np.asarray(ray_map(np.r_[1.0, controls]))
        return complex(m[0, 0], -m[0, 1]), complex(m[0, 2], -m[0, 3])

    def residuals(log_ratio, length):
        a, b = a_b(np.r_[base[:3] * np.exp(log_ratio), base[3]])
        signed_a = (a * np.conj(b)).real / abs(b)      # A and B share the Larmor rotation
        return np.r_[signed_a / 100, 50 * np.log(abs(b) / length), log_ratio[1:], 0.1 * log_ratio[:1]]

    bounds = np.log(np.array([2000.0, max_word])[:, None] / (per_unit * base[:3]))
    x, table = np.log([0.6, 1.0, 1.0]), {}
    for length in sorted(camera_lengths_m):
        x = least_squares(residuals, np.clip(x, *(bounds + [[1e-3], [-1e-3]])), args=(length,),
                          x_scale=0.1, bounds=bounds).x
        controls = np.r_[base[:3] * np.exp(x), base[3]]
        rounded = np.array([drives[n].word(c) for n, c in zip(PROJECTOR, controls)])
        # whole DAC values: the neighbour (within 2 of each of IL1-IL3) with the smallest |A|
        steps = np.stack(np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij"), -1).reshape(-1, 3)
        candidates = rounded + np.c_[steps, np.zeros(len(steps), int)]
        a_values = [abs(a_b(np.array([drives[n].control(w) for n, w in zip(PROJECTOR, c)]))[0]) for c in candidates]
        words = dict(zip(PROJECTOR, (int(w) for w in candidates[int(np.argmin(a_values))])))
        a, b = a_b(np.array([drives[n].control(words[n]) for n in PROJECTOR]))
        if abs(a) > 1.0 or abs(np.log(abs(b) / length)) > 1e-3:
            raise ValueError(f"no DIFF row for camera length {length} m within the DAC range (|A| {abs(a):.3g}, "
                             f"|B| {abs(b):.4g} m/rad)")
        table[float(length)] = words
    return table


def build_virtual_neoarm(*, headroom_word=60000, ray_sample_power=18, detector_samples_per_pixel=1,
                         camera=None, seed=0, diffraction_table=None):
    """Virtual NEOARM whose relaxed lens ranges fit the selected plan.

    ``diffraction_table`` (camera length in m -> IL1, IL2, IL3, PL1 words) replaces the
    designed TEM DIFF rows, e.g. with recorded DIFF values mapped onto this lens table.

    Returns
    -------
    scope : VirtualMicroscope
    plan : dict
        The selected acquisition's settings.
    gauge : dict
        Reference magnification/spot and the relative controls assigned to them.
    """
    plan, spec, baselines, condenser_reference = load_plan()
    truth = OpticalTruth.from_files(ACQUISITION / "experiment.json", ACQUISITION / "scoring_truth.json")
    largest = np.r_[plan["projector_controls"][:, 1:].max(axis=0), plan["condenser_controls"].max(axis=0)]
    drives = {name: LensDrive(headroom_word / top) for name, top in zip(PROJECTOR + CONDENSER, largest)}
    # 0.01 nm and 0.2 µrad per CLA word: 20 nm and 3 mrad are 2000 and 15000 words.
    deflector = DeflectorDrive(np.diag([1e-11, 1e-11, 2e-7, 2e-7]))
    electronics = Electronics(drives, deflector)
    magnification_table = {m: {n: drives[n].word(c) for n, c in zip(PROJECTOR, controls)}
                           for m, controls in baselines.items()}
    if diffraction_table is None:
        diffraction_table = design_diffraction_table(truth, drives, magnification_table[40000.0])
    spot_table = {REFERENCE_SPOT: {n: drives[n].word(c) for n, c in zip(CONDENSER, condenser_reference)}}
    scope = VirtualMicroscope(
        truth, electronics, magnification_table=magnification_table, spot_table=spot_table,
        diffraction_table=diffraction_table,
        camera=camera or CameraModel(), aperture_holes={hole: d for d, hole in APERTURE_HOLES.items()},
        beam_current_a=DOSE_PER_IMAGE * elementary_charge / EXPOSURE_S, ray_sample_power=ray_sample_power,
        detector_samples_per_pixel=detector_samples_per_pixel, seed=seed)
    gauge = dict(reference_magnification=40000.0, reference_spot=REFERENCE_SPOT,
                 projector_reference_ratios=baselines[40000.0], condenser_reference_ratios=condenser_reference)
    return scope, plan, gauge


def run(output, *, rows=None, number_of_starts=4, **build):
    """Acquire through PyJEM calls, fit, and score against the hidden truth."""
    scope, plan, gauge = build_virtual_neoarm(**build)
    if rows is not None:
        plan = {k: v[rows] for k, v in plan.items()}
    vendor = vendor_calibration(scope)
    spec = json.loads((ACQUISITION / "experiment.json").read_text())
    if output.exists():
        shutil.rmtree(output)
    with install(scope):
        from PyJEM import TEM3, detector  # the virtual microscope, through PyJEM's own names

        folder = acquire_calibration(
            TEM3, detector, plan=plan, vendor=vendor, output_directory=output,
            voltages_v=spec["voltages_v"], aperture_diameters_um=spec["aperture_diameters_um"],
            aperture_holes=APERTURE_HOLES, exposure_s=EXPOSURE_S, binning=8, projector_aperture_um=50.0,
            model_assumptions=dict(aperture_edge_width_m=spec["aperture_edge_width_m"], aperture_diffraction=False,
                                   crossover_fraction_bounds=[0.01, 0.99], moment_margin_pixels=16),
            **gauge)
    measure_images(experiment_directory=folder)
    microscope = fit_microscope(measurements_directory=folder, number_of_starts=number_of_starts,
                                projector_aperture_um=50.0)
    (folder / "scoring_truth.json").write_text(json.dumps(scope.truth.scoring_truth(), indent=2) + "\n")
    return folder, check_microscope(experiment_directory=folder, microscope=microscope, render_images=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rays", type=int, default=18, help="log2 of rays per image")
    parser.add_argument("--subsamples", type=int, default=1, help="detector sub-samples per pixel axis")
    parser.add_argument("--quick", action="store_true", help="2**14 rays, two starts")
    args = parser.parse_args()
    rays = 14 if args.quick else args.rays
    folder, checks = run(OUTPUT / f"rays{rays}_sub{args.subsamples}", ray_sample_power=rays,
                         detector_samples_per_pixel=args.subsamples, number_of_starts=2 if args.quick else 4)
    print(json.dumps({k: v for k, v in checks.items() if not isinstance(v, (list, dict))}, indent=2))
    print("saved", folder)
