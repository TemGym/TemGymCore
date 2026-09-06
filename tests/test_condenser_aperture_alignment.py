"""Recover a physical aperture scan without knowing actuator distances or skew."""

import json
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "examples/microscope_models"))
jax.config.update("jax_enable_x64", True)
from condenser_aperture_alignment import make_alignment_operator, measure_alignment_scan
from condenser_aperture_experiment import aperture_specification


def test_scan_recovers_unknown_peak_size_and_movement_metric_without_truth_file(
    tmp_path,
):
    x, y = np.meshgrid(np.linspace(-4, 4, 5), np.linspace(-4, 4, 5))
    commands = np.stack([x.ravel(), y.ravel()], axis=1)
    spec = aperture_specification()
    spec["radii_m"] = [2e-6]
    actual = np.r_[-0.4, 0.25, np.log([1.8, 0.6, 0.4]), -0.08]
    values = np.asarray(
        make_alignment_operator(aperture=spec, commands=commands)(actual)
    )
    reference = np.full(len(values), 1e12)
    np.savez(
        tmp_path / "scan.npz",
        commands=commands,
        reference=reference,
        transmitted=np.rint(values * reference),
    )
    (tmp_path / "aperture.json").write_text(json.dumps(spec))
    result = measure_alignment_scan(input_directory=tmp_path)
    np.testing.assert_allclose(result["peak_command"], actual[:2], atol=1e-6)
    np.testing.assert_allclose(result["beam_rms_um"], 1.8, rtol=1e-6)
    movement = np.array([[0.6, -0.08], [0.0, 0.4]])
    np.testing.assert_allclose(
        result["movement_metric_um2"], movement.T @ movement, rtol=1e-6, atol=1e-8
    )
    assert result["residual_rms"] < 0.001
