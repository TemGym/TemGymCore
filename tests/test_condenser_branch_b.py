from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import least_squares

from temgym_core.condenser_branch_b import (
    EXPECTED_BRANCH_B_PAYLOAD_SHA256,
    NORMALIZED_CONTROL_WARNING,
    BranchBLikelihood,
    BranchBObservations,
    BranchBProvenanceError,
    load_branch_b_experiment,
    make_synthetic_branch_b_observations,
    signed_poisson_deviance_residual,
)
from temgym_core.critical_line_calibration import (
    DeflectorCommandKickMap,
    deflector_command_phase_space_moments,
)


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).parents[1]
PAYLOAD = (
    ROOT
    / "examples"
    / "lens_inversion"
    / "neoarm_condenser_branch_b_design.json"
)


@pytest.fixture(scope="module")
def experiment():
    return load_branch_b_experiment(PAYLOAD, repository_root=ROOT)


@pytest.fixture(scope="module")
def observations(experiment):
    return make_synthetic_branch_b_observations(experiment)


@pytest.fixture(scope="module")
def likelihood(experiment, observations):
    return BranchBLikelihood(experiment, observations)


@pytest.fixture(scope="module")
def truth_jacobian(likelihood):
    return np.asarray(likelihood.jacobian(jnp.asarray(likelihood.nominal_parameters)))


def test_payload_drives_exact_actual_geometry_layout(experiment, likelihood):
    assert experiment.payload_sha256 == EXPECTED_BRANCH_B_PAYLOAD_SHA256
    assert experiment.coordinate_status == NORMALIZED_CONTROL_WARNING
    assert experiment.aperture_z_m == pytest.approx(0.250)
    assert experiment.aperture_radius_m == pytest.approx(10.0e-6)
    assert experiment.upper_deflector_z_m == pytest.approx(0.290)
    assert experiment.lower_deflector_z_m == pytest.approx(0.295)
    assert experiment.sample_z_m == pytest.approx(0.325)
    assert experiment.edge_controls.shape == (4, 7)
    assert experiment.tomography_controls.shape == (32, 7)
    assert experiment.deflector_controls.shape == (3, 7)
    assert experiment.edge_commands_xy_m.shape == (4, 21, 2)
    assert experiment.raw_frame_count == 296

    assert len(likelihood.parameter_names) == 56
    assert likelihood.parameter_slices == {
        "target": slice(0, 12),
        "entrance_covariance": slice(12, 22),
        "source_and_obj_prefield": slice(22, 26),
        "K16": slice(26, 42),
        "shared_log_hole_radius": slice(42, 43),
        "shared_log_stage_xy_scale": slice(43, 44),
        "edge_local": slice(44, 56),
    }
    assert len(likelihood.residual_names) == 458
    assert likelihood.residual_slices == {
        "edge_poisson": slice(0, 88),
        "tomography_whitened": slice(88, 408),
        "deflector_whitened": slice(408, 456),
        "shared_priors": slice(456, 458),
    }
    assert len(set(likelihood.parameter_names)) == 56
    assert len(set(likelihood.residual_names)) == 458
    assert all("nuisance.K16" in name for name in likelihood.parameter_names[26:42])
    assert all(
        "response_per_rad" in name for name in likelihood.residual_names[408:456]
    )


def test_matched_synthetic_constructor_is_explicit_and_zero_residual(
    experiment,
    observations,
    likelihood,
):
    assert observations.kind == "synthetic_expectation"
    assert "synthetic" in observations.provenance.lower()
    prediction = likelihood.predict(likelihood.nominal_parameters)
    assert prediction.edge_expected_counts.shape == (4, 22)
    assert prediction.tomography_observables.shape == (32, 10)
    assert prediction.deflector_response_per_rad.shape == (3, 16)
    np.testing.assert_allclose(
        prediction.edge_expected_counts[:, -1],
        experiment.edge_incident_electrons_per_frame,
        rtol=0.0,
        atol=0.0,
    )
    residual = np.asarray(likelihood.residual(likelihood.nominal_parameters))
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=2.0e-12)
    assert float(likelihood.objective(likelihood.nominal_parameters)) < 1.0e-20
    assert np.isfinite(
        float(likelihood.negative_log_likelihood(likelihood.nominal_parameters))
    )


def test_raw_signed_deflector_pairs_reduce_to_stored_response_per_rad(
    experiment,
):
    prediction = experiment.predict(experiment.nominal_parameters)
    signed = experiment.predict_deflector_signed_displacements(
        experiment.nominal_parameters
    )
    assert signed.shape == (3, 4, 2, 4)
    half_difference = 0.5 * (signed[:, :, 1, :] - signed[:, :, 0, :])
    recovered = jnp.swapaxes(
        half_difference / experiment.deflector_drive_rad,
        1,
        2,
    ).reshape((3, 16))
    np.testing.assert_allclose(
        recovered,
        prediction.deflector_response_per_rad,
        rtol=2.0e-14,
        atol=2.0e-14,
    )

    command_map = DeflectorCommandKickMap(
        plane_z_m=np.asarray(
            (experiment.upper_deflector_z_m, experiment.lower_deflector_z_m)
        ),
        plane_names=("upper", "lower"),
        command_names=tuple(f"command_{index}" for index in range(4)),
        reference_command=np.zeros(4),
        kick_bias_xy_rad=np.asarray(((0.3e-6, -0.2e-6), (0.1e-6, 0.4e-6))),
        gain_xy_rad_per_command=np.asarray(
            (
                ((1.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 1.0)),
                ((1.0, 0.0, -1.0, 0.0), (0.0, 1.0, 0.0, -1.0)),
            )
        ),
    )
    signed_commands = []
    for command in range(4):
        delta = np.zeros(4)
        delta[command] = experiment.deflector_drive_rad
        signed_commands.extend((-delta, delta))
    signed_commands = np.asarray(signed_commands)
    raw_recovered = []
    for dacs in experiment.deflector_controls:
        means, covariances = deflector_command_phase_space_moments(
            experiment.column,
            experiment.column.nominal_parameters,
            dacs,
            np.asarray((2.0e-9, 3.0e-6, -1.0e-9, -4.0e-6)),
            experiment.entrance_reference_covariance,
            signed_commands,
            command_map,
            observation_z_m=experiment.sample_z_m,
        )
        raw_recovered.append(
            np.asarray(
                (means[1::2] - means[0::2])
                / (2.0 * experiment.deflector_drive_rad)
            ).T.reshape(-1)
        )
        np.testing.assert_allclose(
            covariances[1::2], covariances[0::2], rtol=0.0, atol=0.0
        )
    np.testing.assert_allclose(
        raw_recovered,
        prediction.deflector_response_per_rad,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert experiment.deflector_noise_cholesky.shape == (48, 48)
    # Four commands x two signs x image/diffraction = 16 frames per state.
    assert 4 * 2 * 2 == 16


def test_dense_cholesky_whitening_is_not_reduced_to_diagonal(
    experiment,
    observations,
):
    tomography_cholesky = experiment.tomography_noise_cholesky.copy()
    tomography_cholesky[np.arange(1, 320), np.arange(319)] = (
        0.17 * experiment.tomography_summary_sigma
    )
    deflector_cholesky = experiment.deflector_noise_cholesky.copy()
    deflector_cholesky[np.arange(1, 48), np.arange(47)] = (
        0.11 * np.diag(deflector_cholesky)[1:]
    )
    tomography_standard = np.linspace(-0.3, 0.4, 320)
    deflector_standard = np.linspace(0.25, -0.2, 48)
    shifted = BranchBObservations(
        edge_counts=observations.edge_counts,
        tomography_observables=(
            observations.tomography_observables
            + (tomography_cholesky @ tomography_standard).reshape((32, 10))
        ),
        tomography_noise_cholesky=tomography_cholesky,
        deflector_response_per_rad=(
            observations.deflector_response_per_rad
            + (deflector_cholesky @ deflector_standard).reshape((3, 16))
        ),
        deflector_noise_cholesky=deflector_cholesky,
        design_payload_sha256=experiment.payload_sha256,
        coordinate_status=experiment.coordinate_status,
        provenance="Synthetic correlated-whitener unit test.",
        kind="synthetic_expectation",
    )
    residual = np.asarray(
        BranchBLikelihood(experiment, shifted).data_residual(
            experiment.nominal_parameters
        )
    )
    np.testing.assert_allclose(residual[:88], 0.0, atol=2.0e-12)
    np.testing.assert_allclose(
        residual[88:408], -tomography_standard, rtol=2.0e-12, atol=2.0e-12
    )
    np.testing.assert_allclose(
        residual[408:], -deflector_standard, rtol=2.0e-12, atol=2.0e-12
    )


def test_poisson_residual_is_exact_and_has_fisher_derivative_at_truth():
    observed = jnp.asarray((3.0, 20.0, 1.0e5))
    expected = jnp.asarray((4.2, 17.0, 1.01e5))
    residual = signed_poisson_deviance_residual(expected, observed)
    deviance = 2.0 * (
        expected - observed + observed * jnp.log(observed / expected)
    )
    np.testing.assert_allclose(residual**2, deviance, rtol=2.0e-12)
    assert np.all(np.sign(np.asarray(residual)) == np.sign(expected - observed))

    derivative = jax.jacfwd(
        lambda means: signed_poisson_deviance_residual(means, observed)
    )(observed)
    np.testing.assert_allclose(
        derivative,
        np.diag(1.0 / np.sqrt(np.asarray(observed))),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_jit_gradients_and_full_jacobian_are_finite(likelihood, truth_jacobian):
    parameters = jnp.asarray(likelihood.nominal_parameters)
    compiled = jax.jit(likelihood.residual)(parameters)
    assert compiled.shape == (458,)
    assert np.all(np.isfinite(np.asarray(compiled)))
    # XLA reassociation leaves sub-nanostandard-deviation roundoff after the
    # very small deflector Cholesky factors are solved.
    assert np.max(np.abs(np.asarray(compiled))) < 5.0e-10
    assert truth_jacobian.shape == (458, 56)
    assert np.all(np.isfinite(truth_jacobian))
    assert np.all(np.linalg.norm(truth_jacobian[:, :12], axis=0) > 0.0)
    gradient = np.asarray(jax.jit(jax.grad(likelihood.objective))(parameters))
    assert gradient.shape == (56,)
    assert np.all(np.isfinite(gradient))


def test_raw_likelihood_reproduces_committed_profiled_rank_and_crlb(
    likelihood,
    truth_jacobian,
):
    diagnostics = likelihood.local_profiled_target_diagnostics(
        likelihood.nominal_parameters,
        jacobian=truth_jacobian,
    )
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    expected = payload["designs"]["design_296"]["fisher_audit"]

    assert diagnostics.target_rank == expected["profiled_target_rank"] == 12
    # The committed audit first projects 12 local edge nuisances, then reports
    # rank 31 for the remaining global block.  Simultaneous profiling reports
    # their combined rank, 31 + 12 = 43, with the same target projection.
    assert diagnostics.nuisance_rank == expected["profiled_nuisance_rank"] + 12
    np.testing.assert_allclose(
        diagnostics.target_singular_values,
        expected["profiled_target_singular_values"],
        rtol=2.0e-10,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(
        diagnostics.scaled_parameter_standard_uncertainties,
        expected["scaled_parameter_standard_uncertainties"],
        rtol=2.0e-9,
        atol=2.0e-11,
    )


def test_bounded_nonlinear_target_recovery_smoke(likelihood):
    """Exercise the SciPy/JAX seam; global nuisance release is tested by rank."""

    base = jnp.asarray(likelihood.nominal_parameters)

    def active_residual(active: jnp.ndarray) -> jnp.ndarray:
        return likelihood.residual(base.at[:12].set(active))

    compiled_residual = jax.jit(active_residual)
    compiled_jacobian = jax.jit(jax.jacfwd(active_residual))
    zero = jnp.zeros(12)
    compiled_residual(zero).block_until_ready()
    compiled_jacobian(zero).block_until_ready()
    initial = np.asarray(
        (
            4.0e-4,
            -6.0e-4,
            5.0e-4,
            -4.0e-4,
            -3.0e-4,
            5.0e-4,
            -6.0e-4,
            4.0e-4,
            2.0e-4,
            -5.0e-4,
            4.0e-4,
            6.0e-4,
        )
    )
    fit = least_squares(
        lambda values: np.asarray(compiled_residual(jnp.asarray(values))),
        initial,
        jac=lambda values: np.asarray(
            compiled_jacobian(jnp.asarray(values))
        ),
        bounds=(-2.0e-3, 2.0e-3),
        x_scale="jac",
        max_nfev=30,
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-8,
    )
    assert fit.success
    assert fit.nfev <= 15
    assert fit.cost < 1.0e-14
    assert np.max(np.abs(fit.x)) < 1.0e-8


def test_payload_tampering_and_unregistered_observations_fail_closed(
    experiment,
    observations,
    tmp_path,
):
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    payload["designs"]["design_296"]["edge_controls"][0][0] += 1.0e-6
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BranchBProvenanceError, match="checksum committed"):
        load_branch_b_experiment(tampered, repository_root=ROOT)

    with pytest.raises(BranchBProvenanceError, match="not registered"):
        BranchBLikelihood(
            experiment,
            BranchBObservations(
                edge_counts=observations.edge_counts,
                tomography_observables=observations.tomography_observables,
                tomography_noise_cholesky=(
                    observations.tomography_noise_cholesky
                ),
                deflector_response_per_rad=(
                    observations.deflector_response_per_rad
                ),
                deflector_noise_cholesky=observations.deflector_noise_cholesky,
                design_payload_sha256="0" * 64,
                coordinate_status=NORMALIZED_CONTROL_WARNING,
                provenance="Synthetic wrong-payload test.",
                kind="synthetic_expectation",
            ),
        )


def test_instrument_constructor_rejects_noninteger_raw_poisson_counts(
    experiment,
    observations,
):
    counts = observations.edge_counts.copy()
    counts[0, 0] += 0.25
    with pytest.raises(ValueError, match="must be integers"):
        BranchBObservations(
            edge_counts=counts,
            tomography_observables=observations.tomography_observables,
            tomography_noise_cholesky=observations.tomography_noise_cholesky,
            deflector_response_per_rad=observations.deflector_response_per_rad,
            deflector_noise_cholesky=observations.deflector_noise_cholesky,
            design_payload_sha256=experiment.payload_sha256,
            coordinate_status=experiment.coordinate_status,
            provenance="Instrument validation unit test.",
            kind="instrument",
        )
