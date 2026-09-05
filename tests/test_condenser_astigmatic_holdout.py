from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.condenser_astigmatic_design import (
    build_astigmatic_design_matrix,
)
from temgym_core.condenser_astigmatic_holdout import (
    DEFAULT_ASTIGMATIC_HELD_OUT_STATE_INDICES,
    astigmatic_held_out_residual,
    astigmatic_training_residual,
    audit_astigmatic_held_out_training,
    evaluate_astigmatic_held_out,
    make_astigmatic_held_out_split,
)
from temgym_core.condenser_astigmatic_likelihood import (
    make_astigmatic_condenser_likelihood,
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
def likelihood_and_split():
    likelihood = make_astigmatic_condenser_likelihood(
        MANIFEST,
        repository_root=ROOT,
    )
    split = make_astigmatic_held_out_split(likelihood)
    return likelihood, split


def test_default_split_reserves_three_complete_singleton_states(
    likelihood_and_split,
):
    _likelihood, split = likelihood_and_split

    assert DEFAULT_ASTIGMATIC_HELD_OUT_STATE_INDICES == (2, 3, 7)
    assert split.held_out_state_indices == (2, 3, 7)
    assert split.held_out_acquisition_indices == (4, 5, 19)
    assert len(split.training_acquisition_indices) == 45
    assert split.held_out_observation_count == 21
    assert split.training_residual_count == 407
    np.testing.assert_array_equal(
        split.held_out_sample_indices,
        np.r_[28:42, 133:140],
    )
    np.testing.assert_array_equal(
        split.held_out_full_residual_indices,
        np.r_[120:134, 225:232],
    )
    assert split.maximum_train_held_out_noise_correlation == 0.0


def test_training_and_held_out_residuals_have_disjoint_shapes_and_jit(
    likelihood_and_split,
):
    likelihood, split = likelihood_and_split
    nominal = jnp.asarray(likelihood.nominal_parameters)

    training = astigmatic_training_residual(likelihood, nominal, split)
    held_out = astigmatic_held_out_residual(likelihood, nominal, split)
    assert training.shape == (407,)
    assert held_out.shape == (21,)
    np.testing.assert_allclose(training, 0.0, rtol=0.0, atol=2.0e-8)
    np.testing.assert_allclose(held_out, 0.0, rtol=0.0, atol=2.0e-8)
    np.testing.assert_allclose(
        jax.jit(
            lambda parameters: astigmatic_training_residual(
                likelihood, parameters, split
            )
        )(nominal),
        training,
        rtol=0.0,
        atol=2.0e-8,
    )


def test_training_jacobian_is_exact_design_subset(likelihood_and_split):
    likelihood, split = likelihood_and_split
    actual = np.asarray(
        jax.jacfwd(
            lambda parameters: astigmatic_training_residual(
                likelihood, parameters, split
            )
        )(jnp.asarray(likelihood.nominal_parameters))
    )
    matrix, _ = build_astigmatic_design_matrix(likelihood.design)
    expected = matrix.matrix[split.training_full_residual_indices]
    np.testing.assert_allclose(actual, expected, rtol=2.0e-7, atol=2.0e-7)


def test_held_out_values_cannot_change_training_residual(
    likelihood_and_split,
):
    likelihood, split = likelihood_and_split
    changed_sample = np.asarray(
        likelihood.observations.sample_observables
    ).copy()
    changed_flat = changed_sample.reshape(-1)
    changed_flat[split.held_out_sample_indices] += np.tile(
        (0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.1), 3
    )
    changed_observations = replace(
        likelihood.observations,
        sample_observables=changed_sample,
        provenance="held-out leakage unit test",
    )
    changed_likelihood = replace(
        likelihood, observations=changed_observations
    )
    changed_split = make_astigmatic_held_out_split(changed_likelihood)
    nominal = likelihood.nominal_parameters

    np.testing.assert_array_equal(
        astigmatic_training_residual(likelihood, nominal, split),
        astigmatic_training_residual(
            changed_likelihood, nominal, changed_split
        ),
    )
    assert np.linalg.norm(
        np.asarray(
            astigmatic_held_out_residual(
                changed_likelihood, nominal, changed_split
            )
        )
    ) > 1.0


def test_dense_cross_covariance_uses_gaussian_conditional_without_leakage(
    likelihood_and_split,
):
    likelihood, _split = likelihood_and_split
    factor = np.asarray(
        likelihood.observations.sample_noise_cholesky
    ).copy()
    # Sample coordinate 28 is held out and coordinate 0 is training.  Adding
    # a lower-triangular entry creates a genuine cross-acquisition covariance.
    factor[28, 0] = 7.5e-4
    observations = replace(
        likelihood.observations,
        sample_noise_cholesky=factor,
        provenance="dense covariance held-out unit test",
    )
    dense_likelihood = replace(likelihood, observations=observations)
    split = make_astigmatic_held_out_split(dense_likelihood)

    assert split.maximum_train_held_out_noise_correlation > 0.0
    assert np.linalg.norm(split.held_out_from_training_regression) > 0.0
    nominal = dense_likelihood.nominal_parameters
    np.testing.assert_allclose(
        astigmatic_training_residual(dense_likelihood, nominal, split),
        0.0,
        rtol=0.0,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        astigmatic_held_out_residual(dense_likelihood, nominal, split),
        0.0,
        rtol=0.0,
        atol=2.0e-8,
    )

    # Modifying only held-out values still cannot alter the training loss.
    sample = np.asarray(observations.sample_observables).copy()
    sample.reshape(-1)[split.held_out_sample_indices] += 0.005
    modified = replace(
        dense_likelihood,
        observations=replace(
            observations,
            sample_observables=sample,
            provenance="dense covariance held-out value unit test",
        ),
    )
    modified_split = make_astigmatic_held_out_split(modified)
    np.testing.assert_array_equal(
        astigmatic_training_residual(dense_likelihood, nominal, split),
        astigmatic_training_residual(modified, nominal, modified_split),
    )


def test_registered_training_audit_retains_rank_and_sub_percent_precision(
    likelihood_and_split,
):
    likelihood, split = likelihood_and_split
    audit = audit_astigmatic_held_out_training(likelihood, split)

    assert audit.training_matrix_shape == (407, 43)
    assert audit.held_out_matrix_shape == (21, 43)
    assert audit.full_rank == 41
    assert audit.full_nullity == 2
    assert audit.nuisance_rank == 29
    assert audit.profiled_target_rank == 12
    assert audit.held_out_response_rank == 17
    assert audit.conditional_precision_pass
    assert audit.checksum_contract_verified
    assert audit.maximum_scaled_target_standard_uncertainty == pytest.approx(
        0.009980278123994601,
        rel=2.0e-10,
    )
    assert audit.target_names[1].endswith("CL1.log_length_m")
    assert int(np.argmax(audit.scaled_target_standard_uncertainties)) == 1


def test_post_fit_evaluation_is_21_dimensional_and_accepts_prediction_covariance(
    likelihood_and_split,
):
    likelihood, split = likelihood_and_split
    addition = np.eye(21) * 1.0e-7
    evaluation = evaluate_astigmatic_held_out(
        likelihood,
        likelihood.nominal_parameters,
        split,
        additional_predictive_covariance=addition,
    )

    assert evaluation.state_indices == (2, 3, 7)
    assert evaluation.acquisition_indices == (4, 5, 19)
    assert evaluation.residual.shape == (21,)
    assert evaluation.raw_difference.shape == (21,)
    assert evaluation.conditional_innovation.shape == (21,)
    assert evaluation.predictive_covariance.shape == (21, 21)
    assert evaluation.degrees_of_freedom == 21
    assert evaluation.chi_square == pytest.approx(0.0, abs=1.0e-12)
    assert evaluation.rms_standardized_residual == pytest.approx(
        0.0, abs=1.0e-12
    )


def test_invalid_holdout_state_sets_fail_closed(likelihood_and_split):
    likelihood, _split = likelihood_and_split
    with pytest.raises(ValueError, match="unique, non-empty, and sorted"):
        make_astigmatic_held_out_split(
            likelihood, held_out_state_indices=(3, 2)
        )
    with pytest.raises(IndexError, match="outside"):
        make_astigmatic_held_out_split(
            likelihood, held_out_state_indices=(99,)
        )
