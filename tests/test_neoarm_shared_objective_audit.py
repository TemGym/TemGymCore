"""Small-matrix checks; no CPU detector-image simulation is performed."""

import sys
from pathlib import Path
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples/lens_inversion"))
from neoarm_shared_objective_audit import (
    ImageDesign, JointImagePredictor, SOURCE_COVARIANCE, finite_pixel_information,
    fixed_tangent_field, make_design, profile_information, remove_column_space,
    information_artifact_metadata, provenance_snapshot, verify_provenance_snapshot,
)

jax.config.update("jax_enable_x64", True)


class AnalyticSharedColumn:
    """Deliberately small stand-in testing data plumbing, not microscope physics."""
    parameter_names = tuple(f"p{i}" for i in range(34))
    nominal_parameters = np.zeros(34)

    def matrix(self, p, u, *, segment="full", coordinates="mechanical"):
        assert coordinates == "mechanical"
        m = jnp.eye(4).at[0, 1].set(.01).at[2, 3].set(.01)
        if segment == "full":
            return m.at[0, 0].set(jnp.exp(p[2]+p[30]))
        assert segment == "sample_to_detector"
        return m.at[0, 0].set(jnp.exp(p[14]+p[30]))

    def matrix_to_plane(self, p, u, z):
        return jnp.eye(4).at[0, 1].set(z-p[0]).at[2, 3].set(z-p[0])


def design():
    return ImageDesign(np.zeros((5, 7)), np.arange(-1, 4), np.zeros(5, dtype=int),
        np.diag([5e-9, .0002, 5e-9, .0002]),
        np.broadcast_to(np.diag([1e-11, 4e-7, 1e-11, 4e-7]), (4, 4, 4)))


def test_invalid_frame_registration_is_rejected():
    with pytest.raises(ValueError, match="Kick indices"):
        ImageDesign(np.zeros((1, 7)), np.array([4]), np.array([0]), np.eye(4), np.zeros((4, 4, 4)))


def test_shared_objective_changes_both_beam_shape_and_calibrated_response():
    predictor = JointImagePredictor(AnalyticSharedColumn(), design())
    zero = jnp.zeros(75)
    jac = np.asarray(jax.jacfwd(predictor)(zero))
    assert jac.shape == (5, 6, 75)
    assert abs(jac[0, 2, 30]) > 0  # common objective changes source-image covariance
    assert abs(jac[1, 0, 30]) > 0  # and the independent sample-position response
    assert abs(jac[1, 0, 14]) > 0  # downstream is not held at synthetic truth
    assert abs(jac[0, 2, 2]) > 0   # condenser and source are fit together
    assert np.linalg.norm(jac[:, :, 34:44]) > 0


def test_calibration_errors_shared_across_settings_not_reference_image():
    predictor = JointImagePredictor(AnalyticSharedColumn(), design())
    values = jnp.zeros(75)
    reference = np.asarray(predictor(values))
    shifted = np.asarray(predictor(values.at[48].set(1.)))
    assert shifted[1, 0] != reference[1, 0]
    np.testing.assert_array_equal(shifted[[0, 2, 3, 4]], reference[[0, 2, 3, 4]])
    prior = predictor.prior_rows()
    np.testing.assert_array_equal(prior[:16, 48:64], np.eye(16))
    assert np.all(prior[:, :48] == 0)  # no hidden objective/source/optical prior
    assert np.all(prior[:, 68] == 0)   # aperture plane genuinely free by default


def test_source_covariance_is_physical_and_all_ten_local_directions_exist():
    predictor = JointImagePredictor(AnalyticSharedColumn(), design())
    np.testing.assert_allclose(np.asarray(predictor.decode(jnp.zeros(75))[1]), SOURCE_COVARIANCE)
    jac = jax.jacfwd(lambda x: predictor.chart.covariance(x).reshape(-1))(jnp.zeros(10))
    # Normalize rows because metres and radians have very different scales.
    scales = np.sqrt(np.diag(SOURCE_COVARIANCE))
    assert np.linalg.matrix_rank(np.asarray(jac)/(scales[:, None]*scales[None, :]).reshape(-1, 1)) == 10


def test_information_profiles_alignment_and_preserves_basis_differences():
    common_alignment = np.ones((2, 1))
    derivative = np.array([[0., 1.], [2., 1.]])
    retained = remove_column_space(derivative, common_alignment)
    np.testing.assert_allclose(retained[:, 0], [-1., 1.], atol=1e-14)
    np.testing.assert_allclose(retained[:, 1], 0., atol=1e-14)
    result = profile_information(np.array([[1., 0.], [0., .1]]), ["a", "b"], [0, 1])
    assert result["numerical_rank"] == 2
    assert result["noise_resolved_rank"] == 0  # threshold is strictly > one
    np.testing.assert_allclose(result["scaled_standard_deviations"], [1., 10.])
    assert not result["global_uniqueness_proved"]


def test_cpu_forward_image_gate(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="GPU"):
        finite_pixel_information(JointImagePredictor(AnalyticSharedColumn(), design()))


def test_design_keeps_alignment_common_only_for_calibrated_groups():
    class Initialization:
        nominal_controls = np.zeros((2, 7))
    result = make_design(Initialization(), np.zeros((2, 7)), np.zeros((3, 7)))
    assert len(result.controls) == 6+2*9*5
    for group in np.unique(result.alignment_group[6:]):
        np.testing.assert_array_equal(result.kick_index[result.alignment_group == group], np.arange(-1, 4))


def test_narrow_calibration_probe_does_not_change_condenser_image_design():
    class Initialization:
        nominal_controls = np.zeros((2, 7))
    calibration = np.arange(7)/10
    result = make_design(Initialization(), np.zeros((2, 7)), np.zeros((3, 7)),
                         calibration_controls=calibration)
    np.testing.assert_array_equal(result.controls[:6, :3], 0.)
    np.testing.assert_allclose(result.controls[6:, :3], np.broadcast_to(calibration[:3], (90, 3)))


def test_tangent_field_has_fixed_exact_endpoints_and_sample_edge():
    field = fixed_tangent_field(32)
    edges = np.asarray(field.z_edges_m)
    np.testing.assert_array_equal(edges[[0, 32, -1]], [-.05, 0., .05])
    assert len(edges) == 65
    assert np.all(np.diff(edges) > 0)
    assert np.diff(edges).min() < np.diff(edges).max()/10


def test_provenance_closure_includes_transitive_optics_and_actual_inputs():
    import neoarm_shared_objective  # initializer import also loads its dependencies
    import neoarm_measured_information_audit
    hashes, inputs = provenance_snapshot()
    required = {
        "examples/lens_inversion/neoarm_shared_objective_audit.py",
        "examples/lens_inversion/neoarm_shared_objective.py",
        "examples/lens_inversion/neoarm_measured_initialization.py",
        "examples/lens_inversion/neoarm_measured_information_audit.py",
        "examples/lens_inversion/neoarm_imaging_recipe.py",
        "src/temgym_core/microscope_inversion.py",
        "src/temgym_core/critical_line_calibration.py",
        "src/temgym_core/physical_nuisance.py",
        "src/temgym_core/gaussian_image_likelihood.py",
        "src/temgym_core/constants.py",
        "src/temgym_core/jax_least_squares.py",
        "src/temgym_core/neoarm_calibration.py",
    }
    assert required.issubset(hashes)
    assert set(inputs) == {"examples/lens_inversion/neoarm_inversion_config.toml",
                          "examples/microscope_models/measure_microscope_geometry.ipynb"}
    verify_provenance_snapshot(hashes)
    corrupted = dict(hashes); corrupted[next(iter(corrupted))] = "0"*64
    with pytest.raises(RuntimeError, match="changed during"):
        verify_provenance_snapshot(corrupted)


def test_information_artifact_digest_matches_saved_npz(tmp_path):
    path = tmp_path/"information.npz"
    np.savez_compressed(path, rows=np.eye(3), parameter_names=np.array(["a", "b", "c"]))
    metadata = information_artifact_metadata(path)
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["size_bytes"] == path.stat().st_size
    assert information_artifact_metadata(None) is None
