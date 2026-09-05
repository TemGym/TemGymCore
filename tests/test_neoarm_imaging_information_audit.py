"""Physical measurement contracts behind the imaging-only condenser audit."""

import importlib.util
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


PATH = Path(__file__).parents[1] / "examples/lens_inversion/neoarm_imaging_information_audit.py"
SPEC = importlib.util.spec_from_file_location("neoarm_imaging_information_audit", PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
jax.config.update("jax_enable_x64", True)


def test_angle_is_already_encoded_in_spatial_covariance():
    covariance = jnp.array(((4.0, 0.6), (0.6, 1.0)))
    summary = AUDIT.spatial_summary(covariance)
    expected = 0.5 * np.arctan2(1.2, 3.0)
    np.testing.assert_allclose(AUDIT.ellipse_angle(summary), expected)
    jacobian = np.asarray(jax.jacfwd(
        lambda x: jnp.concatenate((x, AUDIT.ellipse_angle(x)[None]))
    )(summary))
    assert np.linalg.matrix_rank(jacobian @ jacobian.T) == 3


def test_zero_larmor_drift_can_rotate_a_general_incident_ellipse():
    # State ordering x, theta_x, y, theta_y.  Misaligned position/angle
    # ellipses are allowed by the existing ten-coordinate entrance chart.
    covariance = jnp.array(((4., 0., 0., 0.), (0., 1., 0., .8),
                            (0., 0., 1., 0.), (0., .8, 0., 1.)))
    assert np.linalg.eigvalsh(covariance).min() > 0
    drift = jnp.eye(4).at[0, 1].set(1.).at[2, 3].set(1.)
    propagated = drift @ covariance @ drift.T
    xy = jnp.array((0, 2))
    before = AUDIT.ellipse_angle(AUDIT.spatial_summary(covariance[jnp.ix_(xy, xy)]))
    after = AUDIT.ellipse_angle(AUDIT.spatial_summary(propagated[jnp.ix_(xy, xy)]))
    assert abs(float(after - before)) > 0.2


def test_profiled_null_is_not_reported_as_finite_parameter_precision():
    # Second target and nuisance have the same derivative.
    matrix = np.array(((1., 0., 0.), (0., 1., 1.), (0., 2., 2.)))
    result = AUDIT.profile_targets(matrix, target_count=2)
    assert result["target_rank"] == 1
    assert result["maximum_scaled_standard_uncertainty"] is None


def test_duplicate_same_image_observable_does_not_create_a_new_direction():
    matrix = np.array(((1., 0., 0.), (0., 1., 1.), (0., 2., 2.)))
    augmented = np.vstack((matrix, .3 * matrix[0] + .7 * matrix[1]))
    assert AUDIT.profile_targets(augmented, target_count=2)["target_rank"] == 1


def test_stored_design_needs_angular_or_calibrated_diversity_information():
    report = AUDIT.run_audit(include_diversity=False)
    cases = report["cases"]
    spatial = cases["spatial_only_no_independent_tag"]
    angular = cases["spatial_and_angular_no_independent_tag"]
    assert spatial["target_rank"] == 11
    assert spatial["maximum_scaled_standard_uncertainty"] is None
    assert angular["target_rank"] == 12
    np.testing.assert_allclose(
        angular["maximum_scaled_standard_uncertainty"], 0.010122515743,
        rtol=1e-6,
    )
    assert cases["spatial_plus_separate_physical_ellipse_angle"]["target_rank"] == 11
    mismatch = report["tag_law_mismatch_under_released_entrance_covariance"][1]
    assert mismatch["angle_residual_rms_rad"] > 0.002
    assert min(report["entrance_symplectic_emittances_m_rad"]) > report["quantum_minimum_emittance_m_rad"]
    assert report["hardware_ready"] is False
    assert report["source_sha256"] == hashlib.sha256(PATH.read_bytes()).hexdigest()
    for name in ("astigmatic", "branch_b"):
        provenance = report["design_provenance"]
        design = Path(provenance[f"{name}_design_path"])
        assert provenance[f"{name}_design_sha256"] == hashlib.sha256(design.read_bytes()).hexdigest()


def test_centered_throughput_has_no_direct_cmini_information():
    likelihood = AUDIT.make_astigmatic_condenser_likelihood()
    matrix, contract = AUDIT.centered_throughput_matrix(likelihood)
    np.testing.assert_allclose(matrix[:, 8:12], 0.0, atol=1e-10)
    assert 0.0 < contract["transmission_range"][0] <= contract["transmission_range"][1] <= 1.0
    assert contract["inserted_frames"] == contract["independent_dose_reference_frames"]


def test_cli_preserves_default_stdout(monkeypatch, capsys):
    report = {"hardware_ready": False}
    monkeypatch.setattr(AUDIT, "run_audit", lambda **kwargs: report)
    assert AUDIT.main([]) == 0
    assert json.loads(capsys.readouterr().out) == report


def test_cli_writes_requested_json_path(monkeypatch, tmp_path, capsys):
    report = {"source_sha256": "test", "design_provenance": {}, "hardware_ready": False}
    options = {}

    def fake_audit(**kwargs):
        options.update(kwargs)
        return report

    monkeypatch.setattr(AUDIT, "run_audit", fake_audit)
    destination = tmp_path / "nested" / "report.json"
    assert AUDIT.main(["--output", str(destination), "--centered-throughput"]) == 0
    assert json.loads(destination.read_text()) == report
    assert options == {"include_diversity": True, "include_centered": True}
    assert str(destination) in capsys.readouterr().out
