from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.condenser_observations import (
    BranchADeflectorObservations,
    BranchAObservationDataset,
    BranchATomographyObservations,
    GaussianWhitening,
    ObjectivePrefieldAnchorObservation,
    ObservationProvenance,
    branch_a_parameter_order,
    deflector_gain_parameter_names,
)


def _provenance(kind: str = "instrument", dataset_id: str = "branch-a"):
    return ObservationProvenance(
        evidence_kind=kind,
        dataset_id=dataset_id,
        manifest_sha256=hashlib.sha256(dataset_id.encode()).hexdigest(),
        source_uri=f"file:///data/{dataset_id}",
        extraction_method="registered five-screen reconstruction",
        covariance_method="joint image-level bootstrap",
    )


def _full_blocks():
    tomography_covariance = np.broadcast_to(np.eye(10), (12, 10, 10)).copy()
    tomography_covariance[:, 0, 1] = 0.25
    tomography_covariance[:, 1, 0] = 0.25
    deflector_covariance = np.broadcast_to(np.eye(4), (12, 4, 4)).copy()
    deflector_covariance[:, 0, 2] = 0.15
    deflector_covariance[:, 2, 0] = 0.15
    tomography_uncertainty = GaussianWhitening.from_per_group_covariance(
        tomography_covariance
    )
    deflector_uncertainty = GaussianWhitening.from_per_group_covariance(
        deflector_covariance
    )
    return (
        BranchATomographyObservations(np.zeros((12, 10)), tomography_uncertainty),
        BranchADeflectorObservations(np.zeros((3, 4, 4)), deflector_uncertainty),
    )


def test_instrument_contract_accepts_full_block_covariances_and_whitens():
    tomography, deflector = _full_blocks()
    data = BranchAObservationDataset(
        tomography, deflector, _provenance("instrument")
    )
    tomography_prediction = np.arange(120, dtype=float).reshape(12, 10) / 100.0
    deflector_prediction = np.arange(48, dtype=float).reshape(3, 4, 4) / 200.0

    residual = np.asarray(
        data.whitened_residual(tomography_prediction, deflector_prediction)
    )

    expected_tomography = np.linalg.solve(
        tomography.uncertainty.cholesky, tomography_prediction.reshape(-1)
    )
    expected_deflector = np.linalg.solve(
        deflector.uncertainty.cholesky, deflector_prediction.reshape(-1)
    )
    np.testing.assert_allclose(
        residual, np.concatenate((expected_tomography, expected_deflector))
    )
    assert residual.shape == (168,)
    assert data.instrument_certified
    assert not data.conditional_route_complete
    assert data.residual_block_slices == {
        "tomography": slice(0, 120),
        "deflector": slice(120, 168),
    }


def test_global_covariance_preserves_cross_state_correlations():
    covariance = np.eye(120)
    covariance[0, 10] = covariance[10, 0] = 0.2
    uncertainty = GaussianWhitening.from_global_covariance(
        covariance, group_shape=(12, 10)
    )
    difference = np.zeros((12, 10))
    difference[0, 0] = 1.0
    difference[1, 0] = 1.0

    residual = np.asarray(uncertainty.whiten(difference))

    np.testing.assert_allclose(
        residual, np.linalg.solve(np.linalg.cholesky(covariance), difference.ravel())
    )
    assert uncertainty.representation == "global_covariance"


def test_whitened_residual_is_jax_differentiable():
    tomography, deflector = _full_blocks()
    data = BranchAObservationDataset(tomography, deflector, _provenance())

    def residual(scale):
        return data.whitened_residual(
            jnp.ones((12, 10)) * scale,
            jnp.ones((3, 4, 4)) * (2.0 * scale),
        )

    jacobian = np.asarray(jax.jacfwd(residual)(jnp.asarray(0.0)))
    assert jacobian.shape == (168,)
    assert np.all(np.isfinite(jacobian))
    assert np.linalg.norm(jacobian) > 0.0


@pytest.mark.parametrize("diagonal_block", ["tomography", "deflector"])
def test_instrument_contract_rejects_diagonal_standard_deviation_shorthand(
    diagonal_block,
):
    tomography, deflector = _full_blocks()
    if diagonal_block == "tomography":
        tomography = BranchATomographyObservations(
            np.zeros((12, 10)),
            GaussianWhitening.from_standard_deviations(
                np.ones(10), group_shape=(12, 10)
            ),
        )
    else:
        deflector = BranchADeflectorObservations(
            np.zeros((3, 4, 4)),
            GaussianWhitening.from_standard_deviations(
                np.ones(4), group_shape=(12, 4)
            ),
        )

    with pytest.raises(ValueError, match="diagonal-only"):
        BranchAObservationDataset(tomography, deflector, _provenance("instrument"))


def test_missing_provenance_is_rejected():
    tomography, deflector = _full_blocks()
    with pytest.raises(ValueError, match="provenance is required"):
        BranchAObservationDataset(tomography, deflector, None)


def test_synthetic_constructor_is_deterministic_and_distinctly_labelled():
    tomography_uncertainty = GaussianWhitening.from_standard_deviations(
        np.full(10, 0.2), group_shape=(12, 10)
    )
    deflector_uncertainty = GaussianWhitening.from_standard_deviations(
        np.full(4, 0.1), group_shape=(12, 4)
    )

    first = BranchAObservationDataset.synthetic_from_predictions(
        np.ones((12, 10)),
        tomography_uncertainty,
        np.ones((3, 4, 4)),
        deflector_uncertainty,
        seed=11,
        simulation_id="matched-demo",
    )
    second = BranchAObservationDataset.synthetic_from_predictions(
        np.ones((12, 10)),
        tomography_uncertainty,
        np.ones((3, 4, 4)),
        deflector_uncertainty,
        seed=11,
        simulation_id="matched-demo",
    )

    assert first.evidence_kind == "synthetic"
    assert not first.instrument_certified
    assert first.provenance.source_uri.startswith("synthetic://")
    np.testing.assert_array_equal(first.tomography.values, second.tomography.values)
    np.testing.assert_array_equal(first.deflector.values, second.deflector.values)


def test_independent_objective_prefield_anchor_is_appended_and_whitened():
    tomography, deflector = _full_blocks()
    anchor_uncertainty = GaussianWhitening.from_global_covariance(
        np.diag((4.0, 9.0, 16.0)), group_shape=(1, 3)
    )
    anchor = ObjectivePrefieldAnchorObservation(
        np.asarray((1.0, 2.0, 3.0)),
        anchor_uncertainty,
        _provenance("instrument", "independent-obj-prefield"),
    )
    data = BranchAObservationDataset(
        tomography, deflector, _provenance("instrument", "branch-a"), anchor
    )

    residual = np.asarray(
        data.whitened_residual(
            np.zeros((12, 10)),
            np.zeros((3, 4, 4)),
            np.asarray((3.0, 5.0, 7.0)),
        )
    )

    np.testing.assert_allclose(residual[-3:], (1.0, 1.0, 1.0))
    assert data.residual_block_slices["objective_prefield_anchor"] == slice(168, 171)
    assert data.conditional_route_complete


def test_anchor_must_have_independent_instrument_provenance():
    tomography, deflector = _full_blocks()
    uncertainty = GaussianWhitening.from_global_covariance(
        np.eye(3), group_shape=(1, 3)
    )
    same_source_anchor = ObjectivePrefieldAnchorObservation(
        np.zeros(3), uncertainty, _provenance("instrument", "branch-a")
    )
    with pytest.raises(ValueError, match="independent provenance"):
        BranchAObservationDataset(
            tomography,
            deflector,
            _provenance("instrument", "branch-a"),
            same_source_anchor,
        )

    synthetic_anchor = ObjectivePrefieldAnchorObservation(
        np.zeros(3), uncertainty, _provenance("synthetic", "synthetic-anchor")
    )
    with pytest.raises(ValueError, match="instrument anchor"):
        BranchAObservationDataset(
            tomography,
            deflector,
            _provenance("instrument", "branch-a"),
            synthetic_anchor,
        )


def test_parameter_order_matches_global_likelihood_contract_exactly():
    order = branch_a_parameter_order(
        ("source_z_m", "detector_z_m", "CL1.z_center_m"),
        ("detector-scale:z0",),
        ("model-discrepancy-0",),
    )

    assert order.block_slices == {
        "optical": slice(0, 3),
        "calibration": slice(3, 4),
        "discrepancy": slice(4, 5),
        "entrance_covariance": slice(5, 15),
        "deflector_gain": slice(15, 31),
    }
    assert len(deflector_gain_parameter_names()) == 16
    assert order.parameter_names[5] == (
        "actual_hardware.entrance.entrance_covariance_whitened_log_0"
    )
    assert order.parameter_names[15] == (
        "actual_hardware.upper.kick_gain_x_rad_per_shift_x"
    )
    assert order.parameter_names[-1] == (
        "actual_hardware.lower.kick_gain_y_rad_per_tilt_y"
    )


def test_observation_order_is_state_then_command_then_coordinate():
    tomography, deflector = _full_blocks()
    data = BranchAObservationDataset(tomography, deflector, _provenance())

    assert data.observation_names[0] == "tomography_state_0.log_rms_x_m"
    assert data.observation_names[119] == (
        "tomography_state_11.correlation_y_theta_y"
    )
    assert data.observation_names[120] == (
        "deflector_state_0.shift_x.sample_central_half_difference_x_m"
    )
    assert data.observation_names[-1] == (
        "deflector_state_2.tilt_y.sample_central_half_difference_theta_y_rad"
    )
