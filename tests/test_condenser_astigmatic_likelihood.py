from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest

from temgym_core.condenser_astigmatic_design import (
    build_astigmatic_design_matrix,
    load_astigmatic_condenser_design,
)
from temgym_core.condenser_astigmatic_likelihood import (
    AstigmaticCondenserObservations,
    make_astigmatic_condenser_likelihood,
    make_synthetic_astigmatic_observations,
)


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).parents[1]
MANIFEST = (
    ROOT
    / "examples"
    / "lens_inversion"
    / "neoarm_condenser_astigmatic_design.json"
)


@pytest.fixture(scope="module")
def design():
    return load_astigmatic_condenser_design(MANIFEST, repository_root=ROOT)


@pytest.fixture(scope="module")
def likelihood():
    return make_astigmatic_condenser_likelihood(
        MANIFEST,
        repository_root=ROOT,
    )


def test_likelihood_has_exact_no_tilt_ordering_and_zero_expectation(likelihood):
    parameters = likelihood.nominal_parameters
    prediction = likelihood.predict(parameters)
    residual = np.asarray(likelihood.residual(parameters))

    assert len(likelihood.parameter_names) == 43
    assert len(likelihood.residual_names) == 428
    assert prediction.edge_expected_counts.shape == (4, 22)
    assert prediction.sample_observables.shape == (48, 7)
    assert residual.shape == (428,)
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=0.0)
    assert likelihood.data_residual(parameters).shape == (424,)
    assert likelihood.prior_residual(parameters).shape == (4,)
    assert likelihood.residual_slices == {
        "edge_poisson": slice(0, 88),
        "shared_priors": slice(88, 92),
        "sample_whitened": slice(92, 428),
    }
    assert not any(
        "K16" in name or "deflector" in name.lower()
        for name in (*likelihood.parameter_names, *likelihood.residual_names)
    )
    assert not any("tilt_command" in name for name in likelihood.residual_names)
    assert likelihood.hardware_ready is False


def test_nonlinear_jacobian_exactly_reproduces_committed_local_design(likelihood):
    expected, _ = build_astigmatic_design_matrix(likelihood.design)
    actual = np.asarray(likelihood.jacobian(likelihood.nominal_parameters))

    assert actual.shape == (428, 43)
    np.testing.assert_allclose(actual, expected.matrix, rtol=2.0e-10, atol=2.0e-8)
    assert np.linalg.matrix_rank(actual, tol=1.0e-7 * np.linalg.svd(actual)[1][0]) == 41

    target = actual[:, :12]
    nuisance = actual[:, 12:]
    norms = np.linalg.norm(nuisance, axis=0)
    active = norms > np.finfo(float).tiny
    left, singular, _ = np.linalg.svd(
        nuisance[:, active] / norms[active][None, :],
        full_matrices=False,
    )
    nuisance_rank = int(np.sum(singular > 1.0e-10 * singular[0]))
    projected = target - left[:, :nuisance_rank] @ (
        left[:, :nuisance_rank].T @ target
    )
    target_singular = np.linalg.svd(projected, compute_uv=False)
    assert nuisance_rank == 29
    assert np.sum(target_singular > 1.0e-7 * target_singular[0]) == 12

    diagnostics = likelihood.local_rank_diagnostics()
    assert diagnostics.full_rank == 41
    assert diagnostics.full_nullity == 2
    assert diagnostics.nuisance_rank == 29
    assert diagnostics.profiled_target_rank == 12
    assert diagnostics.targets_locally_identifiable
    assert diagnostics.full_null_directions.shape == (2, 43)
    np.testing.assert_allclose(
        diagnostics.full_null_directions @ actual.T,
        0.0,
        atol=1.0e-7 * diagnostics.full_singular_values[0],
    )


def test_synthetic_draw_is_reproducible_and_bound_to_both_payloads(design):
    first = make_synthetic_astigmatic_observations(
        design, sample_noise=True, seed=19
    )
    second = make_synthetic_astigmatic_observations(
        design, sample_noise=True, seed=19
    )

    assert first.kind == "synthetic_draw"
    assert first.astigmatic_design_sha256 == design.sha256
    assert first.branch_b_payload_sha256 == design.branch_b_payload_sha256
    np.testing.assert_array_equal(first.edge_counts, second.edge_counts)
    np.testing.assert_array_equal(first.sample_observables, second.sample_observables)
    np.testing.assert_array_equal(
        first.sample_noise_cholesky, second.sample_noise_cholesky
    )
    assert np.all(first.sample_observables[:, 6] >= -0.5 * np.pi)
    assert np.all(first.sample_observables[:, 6] <= 0.5 * np.pi)


def test_astigmatic_orientation_residual_is_pi_periodic(likelihood):
    observations = likelihood.observations
    shifted_sample = np.asarray(observations.sample_observables).copy()
    shifted_sample[:, 6] += np.pi
    shifted = AstigmaticCondenserObservations(
        edge_counts=observations.edge_counts,
        sample_observables=shifted_sample,
        sample_noise_cholesky=observations.sample_noise_cholesky,
        astigmatic_design_sha256=observations.astigmatic_design_sha256,
        branch_b_payload_sha256=observations.branch_b_payload_sha256,
        coordinate_status=observations.coordinate_status,
        provenance="pi-shifted synthetic tag orientation",
        kind="synthetic_expectation",
    )
    periodic = replace(likelihood, observations=shifted)
    np.testing.assert_allclose(
        np.asarray(periodic.sample_data_residual(periodic.nominal_parameters)),
        0.0,
        rtol=0.0,
        atol=2.0e-12,
    )


def test_observation_payload_mismatch_fails_closed(likelihood):
    observations = likelihood.observations
    mismatched = replace(
        observations,
        astigmatic_design_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="not registered to this astigmatic design"):
        replace(likelihood, observations=mismatched)


def test_instrument_edge_counts_must_be_integer(likelihood):
    observations = likelihood.observations
    noninteger = np.asarray(observations.edge_counts).copy()
    noninteger[0, 0] += 0.25
    with pytest.raises(ValueError, match="must be integers"):
        AstigmaticCondenserObservations(
            edge_counts=noninteger,
            sample_observables=observations.sample_observables,
            sample_noise_cholesky=observations.sample_noise_cholesky,
            astigmatic_design_sha256=observations.astigmatic_design_sha256,
            branch_b_payload_sha256=observations.branch_b_payload_sha256,
            coordinate_status=observations.coordinate_status,
            provenance="invalid instrument fixture",
            kind="instrument",
        )
