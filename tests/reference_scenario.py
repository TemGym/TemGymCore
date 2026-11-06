"""Fast, deterministic scenarios whose numbers are frozen in ``tests/reference/``.

These are the safety net for the refactor to a public release: every scenario
runs on CPU in float64 in a few seconds and touches a different layer (physical
constants, the microscope column, Gaussian propagation through the biprism
column, the coherent renderer). ``tests/reference/make_baseline.py`` writes the
files; ``tests/test_reference_values.py`` recomputes and compares.
"""
from pathlib import Path

import jax
import numpy as np
from scipy.optimize import brentq

from temgym_core import constants
from temgym_core.components import Detector, PhaseBiprism, Plane
from temgym_core.evaluate import evaluate_gaussians
from temgym_core.microscope import Microscope, find_imaging_condition
from temgym_core.run import column_matrix, run_to_end
from temgym_core.source import rectangular_input_wave

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).parents[1]
TOML = ROOT / "examples/microscope_models/microscope.toml"
VOLTAGES = (30e3, 100e3, 200e3, 300e3)
CONSTANT_FUNCTIONS = (
    "energy2wavelength", "interaction_constant_CE", "relativistic_mass_correction",
    "relativistic_voltage_factor", "effective_accelerating_potential", "energy2sigma",
    "compute_Rc_from_voltage", "compute_Kv_from_voltage",
)
MAGNIFICATION = 150_000
LENSES = ("Obj_post", "IL1", "IL2", "IL3", "PL1")


def constants_scenario():
    """Every voltage-dependent constant at four voltages, plus round trips."""
    table = {
        name: [float(getattr(constants, name)(v)) for v in VOLTAGES]
        for name in CONSTANT_FUNCTIONS
    }
    table["wavelength2energy_round_trip"] = [
        float(constants.wavelength2energy(constants.energy2wavelength(v))) for v in VOLTAGES
    ]
    table["e_over_hbar"] = float(constants.e_over_hbar())
    return table


def focused_microscope():
    """The example column focused for 150,000x, as in the hologram notebook."""
    microscope = Microscope.from_toml(TOML)
    solution = find_imaging_condition(
        microscope.with_settings(magnification=MAGNIFICATION),
        magnification=MAGNIFICATION, vary=["IL1", "IL2", "IL3"],
    )
    return solution


def microscope_scenario(solution=None):
    solution = focused_microscope() if solution is None else solution
    microscope = solution.microscope
    voltage = microscope.voltage
    return {
        "voltage": float(voltage),
        "currents": {n: float(microscope.components[n].current) for n in LENSES},
        "focal_lengths_m": {n: float(microscope.components[n].focal_length(voltage))
                            for n in LENSES},
        "rotation_rad": {n: float(microscope.components[n].rotation_angle(voltage))
                         for n in LENSES},
        "specimen_to_detector_matrix": np.asarray(solution.matrix).tolist(),
        "residual_max_abs": float(np.max(np.abs(solution.residual))),
    }


def biprism_column(solution=None):
    """Specimen -> objective -> biprism -> IL1..IL3 -> PL1 -> camera, in vacuum."""
    solution = focused_microscope() if solution is None else solution
    microscope = solution.microscope
    voltage = microscope.voltage
    lenses = {n: microscope.components[n] for n in LENSES}
    z_sample = float(microscope.components["sample"].z)

    def matrix_to(z_end, last=np.inf):
        column = tuple(c for c in microscope.column
                       if z_sample < float(c.z) < z_end and float(c.z) <= last)
        return np.asarray(column_matrix(column + (Plane(z=z_end),),
                                        source_z=z_sample, voltage=voltage))

    z_obj = float(lenses["Obj_post"].z)
    z_bfp = brentq(lambda z: matrix_to(z, z_obj)[0, 0], z_obj + 1e-4, z_obj + 0.02)
    z_img = brentq(lambda z: matrix_to(z, z_obj)[0, 2], z_obj + 0.05,
                   float(lenses["IL1"].z) - 1e-3)
    first_image = matrix_to(z_img, z_obj)[:2, :2]
    magnification_objective = float(np.sqrt(abs(np.linalg.det(first_image))))
    image_angle = float(np.arctan2(first_image[1, 0], first_image[0, 0]))
    z_biprism = z_img - 100e-3
    biprism = PhaseBiprism(z=z_biprism, strength=3.5e-5, width=0.6e-6, theta=image_angle,
                           sharpness=4 / 100e-9, eps=0.02)
    column = (lenses["Obj_post"], Plane(z=z_biprism), biprism,
              lenses["IL1"], lenses["IL2"], lenses["IL3"], lenses["PL1"])
    planes = dict(z_sample=z_sample, z_bfp=float(z_bfp), z_first_image=float(z_img),
                  z_biprism=float(z_biprism), objective_magnification=magnification_objective,
                  image_angle_rad=image_angle)
    return microscope, column, planes


def illumination(z_sample, voltage):
    """41 x 41 beamlets (waist 12 nm, 6 nm lattice) covering 240 nm: fast, still overlaps."""
    return rectangular_input_wave(
        aperture_width=240e-9, aperture_height=240e-9, waist=12e-9,
        voltage=voltage, overlap_factor=2.0, z0=z_sample,
    )


def beam_subset(beam, count=4):
    """Head and tail of a propagated bundle plus bundle-wide sums."""
    idx = np.r_[np.arange(count), np.arange(beam.x.shape[0] - count, beam.x.shape[0])]
    take = lambda a: np.asarray(a)[idx]
    q = np.asarray(beam.Q_inv)[idx]
    return {
        "count": int(beam.x.shape[0]),
        "x": take(beam.x).tolist(), "y": take(beam.y).tolist(),
        "dx": take(beam.dx).tolist(), "dy": take(beam.dy).tolist(),
        "z": float(np.asarray(beam.z).ravel()[0]),
        "pathlength": take(beam.pathlength).tolist(),
        "amplitude_re": take(np.real(beam.amplitude)).tolist(),
        "amplitude_im": take(np.imag(beam.amplitude)).tolist(),
        "Q_inv_re": q.real.tolist(), "Q_inv_im": q.imag.tolist(),
        "sum_abs_amplitude": float(np.sum(np.abs(np.asarray(beam.amplitude)))),
        "sum_pathlength": float(np.sum(np.asarray(beam.pathlength))),
    }


def propagate_and_render(solution=None):
    """Run the bundle to the camera and render a centre line and a small patch."""
    microscope, column, planes = biprism_column(solution)
    voltage = microscope.voltage
    detector_z = float(microscope.components["detector"].z)
    beam = illumination(planes["z_sample"], voltage)
    at_camera = jax.jit(run_to_end)(beam, column + (Plane(z=detector_z),))
    line = Detector(z=detector_z, pixel_size=(14e-6, 14e-6), shape=(1, 1024))
    patch = Detector(z=detector_z, pixel_size=(14e-6, 14e-6), shape=(16, 16))
    line_field = np.asarray(evaluate_gaussians(at_camera, line, method="scan"))[0]
    patch_field = np.asarray(evaluate_gaussians(at_camera, patch, method="scan"))
    return planes, at_camera, line_field, patch_field


def fringe_numbers(line_field, pixel=14e-6):
    """Period (m) and contrast of the strongest non-DC line in the hologram row."""
    intensity = np.abs(line_field) ** 2
    spectrum = np.abs(np.fft.rfft(intensity - intensity.mean()))
    peak = int(np.argmax(spectrum[2:]) + 2)
    return {
        "peak_index": peak,
        "period_m": float(len(intensity) * pixel / peak),
        "modulation": float(2 * spectrum[peak] / (len(intensity) * intensity.mean())),
        "mean_intensity": float(intensity.mean()),
        "max_intensity": float(intensity.max()),
        "total_intensity": float(intensity.sum()),
    }
