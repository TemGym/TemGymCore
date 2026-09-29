"""Frozen numerical baseline: refactors must not change these numbers.

The scenarios live in ``golden_scenario.py``; the frozen values in
``tests/golden/baseline.{json,npz}`` were written by ``golden/make_baseline.py``.
Regenerate them only for an intended, reviewed numerical change.
"""
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest

import golden_scenario as scenario
from temgym_core.evaluate import evaluate_gaussians, gpu_kernel_available

GOLDEN = Path(__file__).parent / "golden"
BASELINE = json.loads((GOLDEN / "baseline.json").read_text())
ARRAYS = np.load(GOLDEN / "baseline.npz")

jax.config.update("jax_enable_x64", True)


def close(actual, expected, *, rtol=1e-9, atol=0.0):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol)


# Fields are compared after removing their (physically meaningless) global phase.
# ``Ray.pathlength`` is the excess optical path (0.7 mm here), so the phase
# k * pathlength is about 2e9 rad and float64 resolves it to ~2e-7 rad per beamlet.
# Measured on this column, the GPU ``scan`` renderer and the Pallas kernels differ
# from the frozen CPU baseline by 2e-8 and 7e-9 of the peak field. 1e-6 leaves a
# margin for other platforms and stays far below any real physics change.
FIELD_RTOL = 1e-6


def close_field(actual, expected, rtol=FIELD_RTOL):
    """Complex fields agree to ``rtol`` of their peak amplitude, up to a global phase."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    actual = actual * np.exp(-1j * np.angle(np.vdot(expected, actual)))
    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol * scale)


@pytest.fixture(scope="module")
def solution():
    return scenario.focused_microscope()


@pytest.fixture(scope="module")
def rendered(solution):
    return scenario.propagate_and_render(solution)


def test_constants_match_baseline():
    now, then = scenario.constants_scenario(), BASELINE["constants"]
    assert now.keys() == then.keys()
    for name in then:
        close(now[name], then[name], rtol=1e-13)


def test_focused_microscope_matches_baseline(solution):
    now, then = scenario.microscope_scenario(solution), BASELINE["microscope"]
    close(now["voltage"], then["voltage"], rtol=0)
    for group in ("currents", "focal_lengths_m", "rotation_rad"):
        assert now[group].keys() == then[group].keys()
        for lens in then[group]:
            close(now[group][lens], then[group][lens], rtol=1e-8, atol=1e-12)
    close(now["specimen_to_detector_matrix"], then["specimen_to_detector_matrix"],
          rtol=1e-8, atol=1e-8)
    assert now["residual_max_abs"] < 1e-12


def test_biprism_column_planes_match_baseline(rendered):
    planes, *_ = rendered
    assert planes.keys() == BASELINE["planes"].keys()
    for name, value in BASELINE["planes"].items():
        close(planes[name], value, rtol=1e-9, atol=1e-12)


def test_propagated_bundle_matches_baseline(rendered):
    _, at_camera, *_ = rendered
    now, then = scenario.beam_subset(at_camera), BASELINE["beam"]
    assert now["count"] == then["count"] == 41 * 41
    for name in then:
        if name == "count":
            continue
        close(now[name], then[name], rtol=1e-9, atol=1e-15)


def test_hologram_line_and_patch_match_baseline(rendered):
    _, _, line_field, patch_field = rendered
    close_field(line_field, ARRAYS["line_field"])
    close_field(patch_field, ARRAYS["patch_field"])
    now, then = scenario.fringe_numbers(line_field), BASELINE["fringes"]
    assert now["peak_index"] == then["peak_index"]
    close(now["period_m"], then["period_m"], rtol=1e-12)
    for name in ("modulation", "mean_intensity", "max_intensity", "total_intensity"):
        close(now[name], then[name], rtol=FIELD_RTOL)


def test_saved_calibration_results_are_unchanged():
    """The tracked outputs of the SEM and TEM calibration threads are the M5 targets."""
    saved = BASELINE["saved_results"]
    for key, path in {
        "sem_fit": "examples/sem_reflection/output/calibration_gaussian/fit.json",
        "sem_checks": "examples/sem_reflection/output/calibration_gaussian/checks.json",
        "tem_validation": "examples/microscope_models/calibration_results/tilt_y_30_ray_acquisition/validation.json",
        "tem_repeats": "examples/microscope_models/calibration_results/tilt_y_30_ray_acquisition/repeat_validation.json",
    }.items():
        digest = hashlib.sha256((scenario.ROOT / path).read_bytes()).hexdigest()
        assert digest == saved["sha256"][key], f"{path} changed"
    assert saved["sem"]["rank"] == 9 and saved["sem"]["accepted"]
    close(saved["sem"]["objective_working_distance_mm"], 40.15395340430102, rtol=1e-12)
    assert saved["tem"]["used_images"] == 356 and saved["tem"]["rank"] == 47
    assert saved["tem"]["repeat_seeds"] == list(range(262510, 262516))
    assert saved["tem"]["repeat_accepted"] == 6


@pytest.mark.gpu
@pytest.mark.skipif(not gpu_kernel_available(),
                    reason="needs a verified Pallas GPU backend (run with JAX_PLATFORMS=cuda)")
@pytest.mark.parametrize("method", ["spatial", "gpu_kernel"])
def test_gpu_renderers_match_the_frozen_cpu_hologram(solution, method):
    microscope, column, planes = scenario.biprism_column(solution)
    detector_z = float(microscope.components["detector"].z)
    beam = scenario.illumination(planes["z_sample"], microscope.voltage)
    from temgym_core.components import Detector, Plane
    from temgym_core.run import run_to_end

    at_camera = jax.jit(run_to_end)(beam, column + (Plane(z=detector_z),))
    line = Detector(z=detector_z, pixel_size=(14e-6, 14e-6), shape=(1, 1024))
    field = np.asarray(evaluate_gaussians(at_camera, line, method=method))[0]
    close_field(field, ARRAYS["line_field"])
    # The GPU kernels and the portable scan renderer are the same sum.
    scan = np.asarray(evaluate_gaussians(at_camera, line, method="scan"))[0]
    close_field(field, scan)
