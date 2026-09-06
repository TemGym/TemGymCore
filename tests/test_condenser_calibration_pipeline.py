"""Check information content, physical transport, and the truth-free fitting boundary."""

import json
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

EXAMPLES = Path(__file__).parents[1] / "examples/microscope_models"
sys.path.insert(0, str(EXAMPLES))
jax.config.update("jax_enable_x64", True)

from condenser_calibration_model import (
    equivalent_parameters,
    generating_parameters,
    matrix5,
    named_parameters,
    specimen_moments,
)
from condenser_calibration_workflow import check_identifiability
from condenser_fitting import beam_to_moments, fit_condenser, model_moments
from condenser_image_pipeline import make_image_operator, predict_beam_coordinates
from finite_projector_pipeline import matrix5 as projector_matrix5


def test_affine_ray_order_and_symplectic_transport():
    p = generating_parameters()
    m = np.asarray(matrix5(p, np.array([0.8, 1.1, 1.2])))
    np.testing.assert_array_equal(m[4], [0, 0, 0, 0, 1])
    np.testing.assert_array_equal(m[:4, 4], 0)
    symplectic = np.block(
        [[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]]
    )
    np.testing.assert_allclose(
        m[:4, :4].T @ symplectic @ m[:4, :4], symplectic, atol=1e-11
    )
    assert named_parameters(p)["source_z_mm"] == pytest.approx(-325.0)


def test_different_geometries_are_exactly_indistinguishable_at_new_controls():
    p = generating_parameters()
    alternative = equivalent_parameters(p, upstream_scale=1.3, objective_strength=-0.6)
    controls = np.random.default_rng(59).uniform(0.65, 1.45, (31, 3))
    moments = jax.jit(jax.vmap(specimen_moments, in_axes=(None, 0)))
    mean, cov = moments(p, controls)
    other_mean, other_cov = moments(alternative, controls)
    np.testing.assert_allclose(other_mean, mean, rtol=1e-10, atol=1e-15)
    widths = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    scale = widths[:, :, None] * widths[:, None, :]
    np.testing.assert_allclose(other_cov / scale, cov / scale, rtol=1e-10, atol=1e-12)
    assert abs(named_parameters(alternative)["source_z_mm"] + 325) > 90
    with pytest.raises(ValueError):
        equivalent_parameters(p, upstream_scale=-1)


def test_known_source_angle_adds_two_missing_local_directions():
    controls = np.random.default_rng(27).uniform(0.65, 1.45, (40, 3))
    audit = check_identifiability(parameters=generating_parameters(), controls=controls)
    assert audit["current_measurements"]["rank"] == 17
    assert audit["hypothetical_source_angle_change"]["rank"] == 19


def test_calibrated_views_are_real_projector_maps_with_fixed_objective():
    record = json.loads((EXAMPLES / "condenser_projector_views.json").read_text())
    for view in record["views"]:
        assert view["controls"][0] == 1.0
        expected = projector_matrix5(
            np.asarray(record["calibrated_projector_parameters"]),
            np.asarray(view["controls"]),
        )
        np.testing.assert_allclose(view["matrix"], expected, rtol=2e-12, atol=1e-8)


def test_image_operator_conserves_dose_on_a_large_detector_and_rejects_astigmatism():
    template = {
        "axis_m": np.array([-5e-9, 5e-9]),
        "intensity": np.array([[1.0, 2.0], [3.0, 4.0]]),
    }
    detector = {
        "shape_yx": [64, 64],
        "sample_kernel_sigma_m": 1e-9,
        "pixel_pitch_m": 1e-6,
        "psf_sigma_pixels": 0.5,
    }
    matrices = np.eye(5)[None].copy()
    matrices[0, :2, :2] *= 1000
    operator = make_image_operator(
        projector_matrices=matrices, template=template, detector=detector
    )
    beam = np.array(
        [0, 0, 0, 0, np.log(10e-9), 0, np.log(1e-5), np.log(1e5), np.log(2)]
    )
    frame = np.asarray(operator(beam))[0]
    assert (frame - 2).sum() == pytest.approx(1e5, rel=1e-8)
    changed = matrices.copy()
    changed[0, :2, :2] *= 1.01
    rebuilt = make_image_operator(
        projector_matrices=changed, template=template, detector=detector
    )
    np.testing.assert_allclose(operator(beam, changed), rebuilt(beam), rtol=1e-12)
    matrices[0, 0, 0] *= 2
    with pytest.raises(ValueError, match="round"):
        make_image_operator(
            projector_matrices=matrices, template=template, detector=detector
        )


def test_source_is_fitted_from_measurements_without_a_truth_file(tmp_path, monkeypatch):
    import condenser_calibration_model

    p = generating_parameters()
    controls = np.asarray(
        json.loads((EXAMPLES / "condenser_measurement_protocol.json").read_text())[
            "calibration_controls"
        ]
    )
    values = np.asarray(
        jax.vmap(predict_beam_coordinates, in_axes=(None, 0))(p, controls)
    )
    np.testing.assert_allclose(
        jax.vmap(beam_to_moments)(values),
        jax.vmap(model_moments, in_axes=(None, 0))(p, controls),
        rtol=1e-9,
        atol=1e-12,
    )
    covariance = np.repeat(
        np.diag([0.001] * 4 + [0.01, 0.1, 0.1])[None] ** 2, len(controls), axis=0
    )
    measured = tmp_path / "measurements.npz"
    np.savez(measured, controls=controls, values=values, covariance=covariance)

    def forbidden():
        raise AssertionError("Inference must not read the generating microscope")

    monkeypatch.setattr(condenser_calibration_model, "generating_parameters", forbidden)
    guess = p.copy()
    guess[13:17] = 0.0
    guess[17:] += np.log(1.5)
    fitted = fit_condenser(
        measurements_path=measured,
        starting_parameters=guess,
        output_directory=tmp_path / "fit",
        number_of_starts=1,
        max_evaluations=100,
    )
    assert fitted["best_fit"]["data_rms"] < 1e-6
    np.testing.assert_allclose(fitted["best_fit"]["parameters"][13:], p[13:], atol=1e-6)
    assert not fitted["use_soft_priors"]
    assert fitted["data_rank"] == 17
