from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from temgym_core.identifiability_reporting import assess_matched_model_recovery
from temgym_core.microscope_inversion import IdentifiableSubspaceFit
from temgym_core.nuisance_profiled_subspace import (
    compare_scaled_active_subspaces,
    fit_nuisance_profiled_polytope,
    fit_nuisance_profiled_subspace,
    nuisance_profiled_active_bounds,
    nuisance_profiled_identifiable_subspace,
    profile_nuisance_profiled_polytope_direction,
)
import temgym_core.nuisance_profiled_subspace as profiled_module


def test_scaled_active_subspace_stability_is_basis_invariant():
    first = np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)))
    # Same column space under a nonsingular change of basis, permutation, and
    # sign change.  The helper accepts spanning matrices, not just pristine Qs.
    second = first @ np.asarray(((2.0, -1.0), (3.0, 4.0)))

    stability = compare_scaled_active_subspaces(first, second)

    assert stability.ranks_match
    assert stability.ambient_dimension == 4
    assert stability.first_rank == stability.second_rank == 2
    assert np.allclose(stability.principal_cosines, 1.0, atol=1e-15)
    assert np.allclose(stability.principal_angles_rad, 0.0, atol=2e-8)
    assert np.isclose(stability.maximum_principal_angle_rad, 0.0, atol=2e-8)
    assert np.isclose(stability.spectral_projector_distance, 0.0, atol=1e-15)
    assert not stability.principal_cosines.flags.writeable
    assert not stability.principal_angles_rad.flags.writeable


def test_scaled_active_subspace_stability_recovers_known_principal_angle():
    angle = 0.37
    first = np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)))
    second = np.asarray(
        ((np.cos(angle), 0.0), (0.0, 1.0), (np.sin(angle), 0.0))
    )

    stability = compare_scaled_active_subspaces(first, second)

    assert np.allclose(stability.principal_cosines, (1.0, np.cos(angle)))
    assert np.allclose(stability.principal_angles_rad, (0.0, angle), atol=2e-8)
    assert np.isclose(stability.maximum_principal_angle_rad, angle)
    assert np.isclose(stability.spectral_projector_distance, np.sin(angle))


def test_scaled_active_subspace_stability_handles_zero_rank_and_rank_change():
    empty = np.empty((3, 0))
    zero_rank = compare_scaled_active_subspaces(empty, empty)

    assert zero_rank.ranks_match
    assert zero_rank.principal_cosines.shape == (0,)
    assert zero_rank.principal_angles_rad.shape == (0,)
    assert zero_rank.maximum_principal_angle_rad == 0.0
    assert zero_rank.spectral_projector_distance == 0.0

    rank_change = compare_scaled_active_subspaces(
        np.asarray(((1.0,), (0.0,), (0.0,))),
        np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
    )
    assert not rank_change.ranks_match
    assert np.allclose(rank_change.principal_cosines, (1.0, 0.0))
    assert np.allclose(rank_change.principal_angles_rad, (0.0, np.pi / 2.0))
    assert rank_change.maximum_principal_angle_rad == np.pi / 2.0
    assert np.isclose(rank_change.spectral_projector_distance, 1.0)


def test_scaled_active_subspace_stability_rejects_incompatible_coordinates():
    with pytest.raises(ValueError, match="ambient dimensions"):
        compare_scaled_active_subspaces(np.eye(2), np.eye(3))
    with pytest.raises(ValueError, match="linearly independent"):
        compare_scaled_active_subspaces(
            np.asarray(((1.0, 1.0), (0.0, 0.0))), np.eye(2)
        )


def test_builder_profiles_nuisance_and_keeps_compensated_null_family():
    # o0 and n0 are exactly confounded; o1 is the only admitted optical mode.
    jacobian = np.asarray(((10.0, 0.0, 10.0), (0.0, 200.0, 0.0)))
    design = nuisance_profiled_identifiable_subspace(
        jacobian,
        np.zeros(3),
        np.ones(3),
        optical_parameter_count=2,
        optical_minimum_singular_value=100.0,
        parameter_names=("o0", "o1", "n0"),
    )

    assert design.optical_rank == 1
    assert design.nuisance_rank == 1
    assert design.optical_subspace.rank == 1
    assert design.optical_subspace.nullity == 1
    assert design.optical_subspace.parameter_names == ("o0", "o1")
    assert design.subspace.rank == 2
    assert design.subspace.nullity == 1
    null = design.subspace.null_basis_scaled[:, 0]
    assert np.allclose(jacobian @ null, 0.0, atol=1e-12)
    assert np.isclose(abs(null[0]), 1 / np.sqrt(2), atol=1e-12)
    assert np.isclose(null[0], -null[2], atol=1e-12)
    assert np.allclose(
        design.subspace.active_basis_scaled.T @ design.subspace.null_basis_scaled,
        0.0,
        atol=1e-12,
    )
    recovery = assess_matched_model_recovery(
        design.optical_subspace,
        np.asarray((0.0, 0.002)),
        np.zeros(2),
    )
    assert recovery.active_one_percent_gate_passed
    assert recovery.non_identifiable_parameter_names == ("o0",)


def test_builder_retains_weak_but_numerically_resolved_nuisance():
    jacobian = np.asarray(((200.0, 0.0), (0.0, 0.01)))
    design = nuisance_profiled_identifiable_subspace(
        jacobian,
        np.zeros(2),
        np.ones(2),
        optical_parameter_count=1,
        optical_minimum_singular_value=100.0,
    )

    assert design.optical_rank == 1
    assert design.nuisance_rank == 1
    assert design.subspace.rank == 2
    assert design.subspace.nullity == 0


def test_builder_rejects_subthreshold_optical_direction_as_practical_null():
    jacobian = np.asarray(
        ((200.0, 0.0, 0.0), (0.0, 50.0, 0.0), (0.0, 0.0, 2.0))
    )
    design = nuisance_profiled_identifiable_subspace(
        jacobian,
        np.zeros(3),
        np.ones(3),
        optical_parameter_count=2,
        optical_minimum_singular_value=100.0,
    )

    assert design.optical_rank == 1
    assert design.nuisance_rank == 1
    assert design.subspace.rank == 2
    assert design.subspace.nullity == 1
    residual_norm = np.linalg.norm(
        jacobian @ design.subspace.null_basis_scaled[:, 0]
    )
    assert np.isclose(residual_norm, 50.0)
    assert np.isclose(design.maximum_practical_null_residual_norm, 50.0)


def test_fitter_returns_existing_fit_type_and_recovers_selected_gauge():
    jacobian = np.asarray(((10.0, 0.0, 10.0), (0.0, 200.0, 0.0)))
    truth = np.asarray((0.4, 0.3, 0.4))
    observed = jacobian @ truth

    def residual(parameters):
        return jnp.asarray(jacobian) @ parameters - jnp.asarray(observed)

    design = nuisance_profiled_identifiable_subspace(
        jacobian,
        np.zeros(3),
        np.ones(3),
        optical_parameter_count=2,
        optical_minimum_singular_value=100.0,
    )
    fit = fit_nuisance_profiled_subspace(
        residual,
        design,
        bounds=(np.full(3, -2.0), np.full(3, 2.0)),
        multistart=2,
        seed=5,
        max_nfev=100,
    )

    assert isinstance(fit, IdentifiableSubspaceFit)
    assert fit.success
    assert fit.fit_rank == design.subspace.rank
    assert np.allclose(jacobian @ fit.parameters, observed, atol=1e-9)
    assert np.allclose(fit.parameters, truth, atol=1e-9)
    active_lower, active_upper = nuisance_profiled_active_bounds(
        design, (np.full(3, -2.0), np.full(3, 2.0))
    )
    assert np.all(fit.active_coordinates >= active_lower)
    assert np.all(fit.active_coordinates <= active_upper)


def test_full_polytope_fitter_reaches_beyond_conservative_cartesian_box():
    target = np.asarray((0.8, -0.75))

    def residual(parameters):
        return parameters - jnp.asarray(target)

    design = nuisance_profiled_identifiable_subspace(
        np.eye(2),
        np.zeros(2),
        np.ones(2),
        optical_parameter_count=2,
        optical_minimum_singular_value=0.1,
        parameter_names=("o0", "o1"),
    )
    physical_bounds = (np.full(2, -1.0), np.full(2, 1.0))
    conservative_lower, conservative_upper = nuisance_profiled_active_bounds(
        design, physical_bounds
    )
    fit = fit_nuisance_profiled_polytope(
        residual,
        design,
        bounds=physical_bounds,
        multistart=6,
        seed=11,
        hit_and_run_burnin=32,
        hit_and_run_thinning=8,
        max_nfev=100,
    )

    assert isinstance(fit, IdentifiableSubspaceFit)
    assert fit.success
    assert np.allclose(fit.parameters, target, atol=2e-6)
    assert np.any(
        (fit.active_coordinates < conservative_lower - 1e-3)
        | (fit.active_coordinates > conservative_upper + 1e-3)
    )
    assert np.all(fit.active_coordinate_bounds[0] <= conservative_lower)
    assert np.all(fit.active_coordinate_bounds[1] >= conservative_upper)
    all_physical_starts = np.asarray(
        [
            design.subspace.parameters_from_active(jnp.asarray(start))
            for start in fit.active_fit.start_parameters
        ]
    )
    assert np.all(all_physical_starts >= physical_bounds[0] - 1e-10)
    assert np.all(all_physical_starts <= physical_bounds[1] + 1e-10)
    assert fit.active_fit.start_parameters.shape == (6, 2)


def test_full_polytope_fitter_requires_finite_physical_box():
    design = nuisance_profiled_identifiable_subspace(
        np.eye(2),
        np.zeros(2),
        np.ones(2),
        optical_parameter_count=2,
        optical_minimum_singular_value=0.1,
    )

    with pytest.raises(ValueError, match="finite physical bounds"):
        fit_nuisance_profiled_polytope(
            lambda parameters: parameters,
            design,
            bounds=(np.full(2, -np.inf), np.ones(2)),
        )


def test_full_polytope_scales_coupled_constraints_before_slsqp():
    # The first physical coordinate is twelve orders of magnitude smaller
    # than the others.  Its bound is nevertheless a real coupled face, not an
    # absolute-tolerance rounding error for SLSQP.
    parameter_scales = np.asarray((1.0e-12, 1.0, 1.0))
    scaled_jacobian = np.asarray(((1.0, -1.0, 0.0), (1.0, 0.0, -1.0)))
    design = nuisance_profiled_identifiable_subspace(
        scaled_jacobian / parameter_scales[None, :],
        np.zeros(3),
        parameter_scales,
        optical_parameter_count=3,
        optical_minimum_singular_value=0.1,
    )
    physical_basis = np.asarray(design.subspace.active_basis_physical)
    target_active = np.asarray((1.3, 1.3))
    target_physical = physical_basis @ target_active
    inverse_basis = np.linalg.pinv(physical_basis)

    def residual(parameters):
        return jnp.asarray(inverse_basis) @ (
            parameters - jnp.asarray(target_physical)
        )

    fit = fit_nuisance_profiled_polytope(
        residual,
        design,
        bounds=(-parameter_scales, parameter_scales),
        multistart=3,
        seed=4,
        hit_and_run_burnin=8,
        hit_and_run_thinning=2,
        max_nfev=100,
    )

    assert fit.success
    assert np.all(fit.parameters >= -parameter_scales - 1e-14 * parameter_scales)
    assert np.all(fit.parameters <= parameter_scales + 1e-14 * parameter_scales)
    # The unconstrained target violates the narrow first-parameter face.
    assert target_physical[0] < -parameter_scales[0]
    assert np.isclose(fit.parameters[0], -parameter_scales[0], rtol=1e-7)
    assert not np.allclose(fit.active_coordinates, target_active, atol=1e-3)


def test_hit_and_run_uses_interior_chain_when_requested_start_is_corner():
    dimension = 8
    target = np.full(dimension, 0.6)
    design = nuisance_profiled_identifiable_subspace(
        np.diag(np.arange(dimension, 0, -1, dtype=float)),
        np.zeros(dimension),
        np.ones(dimension),
        optical_parameter_count=dimension,
        optical_minimum_singular_value=0.1,
    )

    def run(seed):
        return fit_nuisance_profiled_polytope(
            lambda parameters: parameters - jnp.asarray(target),
            design,
            bounds=(np.zeros(dimension), np.ones(dimension)),
            multistart=4,
            seed=seed,
            hit_and_run_burnin=0,
            hit_and_run_thinning=1,
            max_nfev=100,
        )

    first = run(19)
    repeated = run(19)
    changed_seed = run(20)
    starts = first.active_fit.start_parameters

    assert np.array_equal(starts[0], np.zeros(dimension))
    assert np.all(np.linalg.norm(starts[1:], axis=1) > 0.0)
    assert np.all(starts >= 0.0)
    assert np.all(starts <= 1.0)
    assert np.array_equal(starts, repeated.active_fit.start_parameters)
    assert not np.array_equal(starts[1:], changed_seed.active_fit.start_parameters[1:])


def test_hit_and_run_multichain_plan_is_balanced_deterministic_and_feasible():
    initial = np.asarray((0.37, -0.21, 0.11))
    interior = np.zeros(3)
    constraint_matrix = np.eye(3)
    constraint_lower = np.full(3, -1.0)
    constraint_upper = np.full(3, 1.0)

    def plan(seed):
        return profiled_module._hit_and_run_polytope_start_plan(
            initial,
            interior,
            constraint_matrix,
            constraint_lower,
            constraint_upper,
            count=8,
            seed=seed,
            chains=3,
            burnin=2,
            thinning=2,
            interior_fraction=0.9,
        )

    first = plan(713)
    repeated = plan(713)
    changed_seed = plan(714)

    assert np.array_equal(first.starts, repeated.starts)
    assert np.array_equal(first.chain_seed_words, repeated.chain_seed_words)
    assert not np.array_equal(first.starts[1:], changed_seed.starts[1:])
    assert np.array_equal(first.chain_sample_counts, (3, 2, 2))
    assert np.array_equal(first.chain_indices, (-1, 0, 0, 0, 1, 1, 2, 2))
    assert np.array_equal(first.within_chain_indices, (-1, 0, 1, 2, 0, 1, 0, 1))
    assert np.unique(first.chain_seed_words).size == 3
    assert np.array_equal(first.starts[0], initial)
    assert np.count_nonzero(np.all(first.starts == initial[None, :], axis=1)) == 1
    embedded = first.starts @ constraint_matrix.T
    assert np.all(embedded >= constraint_lower[None, :])
    assert np.all(embedded <= constraint_upper[None, :])
    assert not first.starts.flags.writeable
    assert not first.chain_indices.flags.writeable
    assert not first.chain_seed_words.flags.writeable

    # Each block is generated from an independently spawned SeedSequence and
    # begins at the same verified interior point, rather than continuing one
    # long correlated chain.
    child_sequences = np.random.SeedSequence(713).spawn(3)
    block_start = 1
    for allocation, child_sequence in zip(
        first.chain_sample_counts, child_sequences, strict=True
    ):
        expected = profiled_module._hit_and_run_chain_samples(
            interior,
            constraint_matrix,
            constraint_lower,
            constraint_upper,
            count=int(allocation),
            generator=np.random.default_rng(child_sequence),
            burn=2,
            thin=2,
            interior_fraction=0.9,
            excluded_point=initial,
        )
        assert np.array_equal(
            first.starts[block_start : block_start + allocation], expected
        )
        block_start += int(allocation)


def test_full_polytope_uses_four_chains_by_default_and_keeps_total_count():
    design = nuisance_profiled_identifiable_subspace(
        np.eye(3),
        np.zeros(3),
        np.ones(3),
        optical_parameter_count=3,
        optical_minimum_singular_value=0.1,
    )
    active_x0 = np.asarray((0.17, -0.13, 0.09))
    fit = fit_nuisance_profiled_polytope(
        lambda parameters: parameters,
        design,
        bounds=(np.full(3, -1.0), np.full(3, 1.0)),
        active_x0=active_x0,
        multistart=10,
        seed=41,
        hit_and_run_burnin=2,
        hit_and_run_thinning=1,
        max_nfev=100,
    )

    starts = fit.active_fit.start_parameters
    assert starts.shape == (10, 3)
    assert np.array_equal(starts[0], active_x0)
    assert np.count_nonzero(np.all(starts == active_x0[None, :], axis=1)) == 1
    assert "4 independently seeded hit-and-run chain(s)" in fit.active_fit.message
    assert "[3, 2, 2, 2]" in fit.active_fit.message


def test_full_polytope_multistart_one_does_not_instantiate_sampling_chain(
    monkeypatch,
):
    design = nuisance_profiled_identifiable_subspace(
        np.eye(2),
        np.zeros(2),
        np.ones(2),
        optical_parameter_count=2,
        optical_minimum_singular_value=0.1,
    )
    monkeypatch.setattr(
        profiled_module,
        "_active_polytope_interior_point",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("one-start fit must not initialize a sampling chain")
        ),
    )
    active_x0 = np.asarray((0.2, -0.1))
    fit = fit_nuisance_profiled_polytope(
        lambda parameters: parameters - jnp.asarray(active_x0),
        design,
        bounds=(np.full(2, -1.0), np.full(2, 1.0)),
        active_x0=active_x0,
        multistart=1,
        hit_and_run_chains=17,
        max_nfev=100,
    )

    assert fit.active_fit.start_parameters.shape == (1, 2)
    assert np.array_equal(fit.active_fit.start_parameters[0], active_x0)
    assert "0 independently seeded hit-and-run chain(s)" in fit.active_fit.message
    assert "[]" in fit.active_fit.message


def test_full_polytope_never_selects_infeasible_failed_solution(monkeypatch):
    design = nuisance_profiled_identifiable_subspace(
        np.eye(2),
        np.zeros(2),
        np.ones(2),
        optical_parameter_count=2,
        optical_minimum_singular_value=0.1,
    )
    fake_results = iter(
        (
            SimpleNamespace(
                x=np.asarray((2.0, 0.0)),
                fun=0.0,
                jac=np.zeros(2),
                success=False,
                message="infeasible low cost",
                nfev=1,
                njev=1,
            ),
            SimpleNamespace(
                x=np.asarray((0.2, 0.3)),
                fun=1.0,
                jac=np.zeros(2),
                success=False,
                message="feasible fallback",
                nfev=1,
                njev=1,
            ),
        )
    )
    monkeypatch.setattr(
        profiled_module, "minimize", lambda *args, **kwargs: next(fake_results)
    )

    fit = fit_nuisance_profiled_polytope(
        lambda parameters: parameters,
        design,
        bounds=(np.full(2, -1.0), np.full(2, 1.0)),
        multistart=2,
        seed=3,
        hit_and_run_burnin=0,
        hit_and_run_thinning=1,
    )

    assert not fit.success
    assert np.allclose(fit.active_coordinates, (0.2, 0.3))
    assert np.allclose(fit.parameters, (0.2, 0.3))
    assert not np.any(fit.active_fit.start_success)


def test_full_polytope_selects_known_lower_cost_even_when_optimizer_flags_failure(
    monkeypatch,
):
    design = nuisance_profiled_identifiable_subspace(
        np.eye(2),
        np.zeros(2),
        np.ones(2),
        optical_parameter_count=2,
        optical_minimum_singular_value=0.1,
    )
    fake_results = iter(
        (
            SimpleNamespace(
                x=np.zeros(2),
                fun=99.0,
                jac=np.ones(2),
                success=False,
                message="lower point exhausted maxiter",
                nfev=1,
                njev=1,
            ),
            SimpleNamespace(
                x=np.asarray((0.5, 0.0)),
                fun=0.0,
                jac=np.zeros(2),
                success=True,
                message="higher point converged",
                nfev=1,
                njev=1,
            ),
        )
    )
    monkeypatch.setattr(
        profiled_module, "minimize", lambda *args, **kwargs: next(fake_results)
    )

    fit = fit_nuisance_profiled_polytope(
        lambda parameters: parameters,
        design,
        bounds=(np.full(2, -1.0), np.full(2, 1.0)),
        multistart=2,
        seed=3,
        hit_and_run_burnin=0,
        hit_and_run_thinning=1,
    )

    assert not fit.success
    assert np.allclose(fit.parameters, 0.0)
    assert np.isclose(fit.cost, 0.0)
    # Stored costs and gradients are independently re-evaluated at each
    # returned point rather than copied from potentially stale SciPy fields.
    assert np.allclose(fit.active_fit.start_costs, (0.0, 0.125))
    assert np.allclose(fit.active_fit.start_gradients, ((0.0, 0.0), (0.5, 0.0)))


def test_full_polytope_covariance_uses_absolute_whitened_fisher_scale():
    design = nuisance_profiled_identifiable_subspace(
        np.eye(2),
        np.zeros(2),
        np.ones(2),
        optical_parameter_count=2,
        optical_minimum_singular_value=0.1,
    )
    target = jnp.asarray((0.2, -0.1))

    def residual(parameters):
        parameter_residual = jnp.asarray((2.0, 4.0)) * (parameters - target)
        # Nonzero realized residual must not rescale a likelihood whose
        # measurement covariance was already used for whitening.
        return jnp.concatenate((parameter_residual, jnp.asarray((7.0,))))

    fit = fit_nuisance_profiled_polytope(
        residual,
        design,
        bounds=(np.full(2, -1.0), np.full(2, 1.0)),
        max_nfev=100,
    )

    expected = np.diag((1.0 / 4.0, 1.0 / 16.0))
    assert fit.success
    assert np.allclose(fit.active_fisher_covariance, expected, atol=1e-12)
    assert np.allclose(fit.identifiable_covariance, expected, atol=1e-12)
    assert np.allclose(
        fit.projected_parameter_standard_errors,
        np.asarray((0.5, 0.25)),
        atol=1e-12,
    )
    assert np.isclose(fit.cost, 24.5)


def test_log_soft_norm_conditions_neoarm_scale_without_changing_raw_outputs():
    sensitivities = np.asarray((5.0e9, 1.0e7, 1.0e4, 1.0e2))
    target = 0.08 * np.sin((np.arange(4) + 1) * np.sqrt(2.0))
    jacobian = np.vstack((np.diag(sensitivities), np.zeros((8, 4))))
    design = nuisance_profiled_identifiable_subspace(
        jacobian,
        np.zeros(4),
        np.ones(4),
        optical_parameter_count=4,
        optical_minimum_singular_value=75.0,
    )

    def residual(parameters):
        return jnp.concatenate(
            (
                jnp.asarray(sensitivities) * (parameters - jnp.asarray(target)),
                jnp.ones(8),
            )
        )

    fit = fit_nuisance_profiled_polytope(
        residual,
        design,
        bounds=(-np.ones(4), np.ones(4)),
        max_nfev=900,
        ftol=1.0e-12,
        optimizer_cost_transform="log_soft_norm",
    )

    assert fit.success
    assert np.allclose(fit.parameters, target, atol=1.0e-8)
    # The eight irreducible unit residuals contribute a raw cost of four.  A
    # transformed optimizer value would be O(1), but must never be reported as
    # the physical likelihood or used to rescale the Fisher covariance.
    assert np.isclose(fit.cost, 4.0, atol=1.0e-10)
    assert np.isclose(fit.active_fit.start_costs[0], fit.cost, atol=1.0e-12)
    assert np.allclose(
        np.diag(fit.active_fisher_covariance) * sensitivities**2,
        1.0,
        atol=1.0e-10,
    )
    assert np.allclose(
        fit.active_fit.start_gradients[0],
        fit.active_fit.jacobian.T @ fit.residuals,
        atol=1.0e-7,
    )
    assert "strictly monotone log-soft" in fit.active_fit.message
    assert "residual-norm scale 3.46410161514" in fit.active_fit.message
    assert "reported costs/gradients/Fisher quantities remain raw" in (
        fit.active_fit.message
    )


def test_log_soft_norm_retains_coupled_physical_bound_at_large_sensitivity():
    parameter_scales = np.asarray((1.0e-12, 1.0, 1.0))
    scaled_geometry = np.asarray(((1.0, -1.0, 0.0), (1.0, 0.0, -1.0)))
    sensitivities = np.asarray((5.0e9, 1.0e2))
    design = nuisance_profiled_identifiable_subspace(
        sensitivities[:, None] * scaled_geometry / parameter_scales[None, :],
        np.zeros(3),
        parameter_scales,
        optical_parameter_count=3,
        optical_minimum_singular_value=75.0,
    )
    physical_basis = np.asarray(design.subspace.active_basis_physical)
    target_active = np.asarray((1.3, 1.3))
    target_physical = physical_basis @ target_active
    inverse_basis = np.linalg.pinv(physical_basis)

    def residual(parameters):
        return jnp.asarray(sensitivities) * (
            jnp.asarray(inverse_basis)
            @ (parameters - jnp.asarray(target_physical))
        )

    fit = fit_nuisance_profiled_polytope(
        residual,
        design,
        bounds=(-parameter_scales, parameter_scales),
        max_nfev=900,
        ftol=1.0e-12,
        optimizer_cost_transform="log_soft_norm",
    )

    assert fit.success
    # Allow a few last-bit matrix-embedding ulps at the physical face.  The
    # normalized active constraints themselves are checked by the fitter.
    assert np.all(fit.parameters >= -parameter_scales - 5.0e-14 * parameter_scales)
    assert np.all(fit.parameters <= parameter_scales + 5.0e-14 * parameter_scales)
    # The unconstrained target crosses the narrow first-parameter face.  The
    # conditioned solve must still find that exact coupled physical boundary.
    assert target_physical[0] < -parameter_scales[0]
    assert np.isclose(fit.parameters[0], -parameter_scales[0], rtol=1.0e-7)
    assert fit.cost > 1.0e3


def test_log_soft_norm_conditions_high_dynamic_range_polytope_profile():
    sensitivity = 5.0e9
    jacobian = np.asarray(
        ((sensitivity, sensitivity), (100.0, -100.0))
    )
    target = np.asarray((0.08, -0.03))
    design = nuisance_profiled_identifiable_subspace(
        jacobian,
        np.zeros(2),
        np.ones(2),
        optical_parameter_count=2,
        optical_minimum_singular_value=75.0,
    )

    def residual(parameters):
        return jnp.asarray(jacobian) @ (parameters - jnp.asarray(target))

    bounds = (-np.ones(2), np.ones(2))
    fit = fit_nuisance_profiled_polytope(
        residual,
        design,
        bounds=bounds,
        max_nfev=900,
        ftol=1.0e-12,
        optimizer_cost_transform="log_soft_norm",
    )
    profile = profile_nuisance_profiled_polytope_direction(
        residual,
        fit,
        np.asarray((1.0, 0.0)),
        np.asarray((-0.02, 0.0, 0.02)),
        bounds=bounds,
        multistart=1,
        max_nfev=900,
        ftol=1.0e-12,
        optimizer_cost_transform="log_soft_norm",
    )

    expected_y = target[1] - (profile.conditional_parameters[:, 0] - target[0])
    assert fit.success
    assert np.all(profile.success)
    assert np.allclose(profile.conditional_parameters[:, 1], expected_y, atol=1.0e-7)
    assert np.allclose(profile.costs[[0, 2]], 8.0, rtol=2.0e-5)
    assert profile.costs[1] < 1.0e-9


def test_log_soft_norm_matches_raw_fit_at_moderate_scale():
    jacobian = np.asarray(((3.0, 1.0), (-0.5, 2.0), (1.0, -1.0)))
    target = np.asarray((0.17, -0.24))
    design = nuisance_profiled_identifiable_subspace(
        jacobian,
        np.zeros(2),
        np.ones(2),
        optical_parameter_count=2,
        optical_minimum_singular_value=0.1,
    )

    def residual(parameters):
        return jnp.asarray(jacobian) @ (parameters - jnp.asarray(target))

    bounds = (-np.ones(2), np.ones(2))
    raw = fit_nuisance_profiled_polytope(
        residual,
        design,
        bounds=bounds,
        max_nfev=200,
        ftol=1.0e-12,
    )
    conditioned = fit_nuisance_profiled_polytope(
        residual,
        design,
        bounds=bounds,
        max_nfev=200,
        ftol=1.0e-14,
        optimizer_cost_transform="log_soft_norm",
    )

    assert raw.success and conditioned.success
    assert np.allclose(
        conditioned.parameters, raw.parameters, rtol=0.0, atol=1.0e-8
    )
    assert conditioned.cost < 1.0e-14
    assert raw.cost < 1.0e-14
    assert np.allclose(
        conditioned.active_fisher_covariance,
        raw.active_fisher_covariance,
        atol=1.0e-12,
    )


def test_log_soft_norm_scalar_conditioner_is_stable_at_float64_extremes():
    helper = profiled_module._condition_slsqp_cost_and_gradient
    largest = np.finfo(float).max
    smallest = np.nextafter(0.0, 1.0)

    high_value, high_gradient = helper(
        largest,
        np.asarray((1.0,)),
        transform="log_soft_norm",
        residual_norm_scale_squared=largest,
    )
    low_value, low_gradient = helper(
        np.finfo(float).tiny,
        np.asarray((1.0,)),
        transform="log_soft_norm",
        residual_norm_scale_squared=smallest,
    )
    subnormal_value, subnormal_gradient = helper(
        smallest,
        np.asarray((smallest,)),
        transform="log_soft_norm",
        residual_norm_scale_squared=smallest,
    )

    assert np.isclose(high_value, 0.5 * np.log(3.0), atol=1.0e-14)
    assert high_gradient[0] > 0.0
    assert np.isfinite(high_gradient[0])
    assert np.isfinite(low_value)
    assert np.isfinite(low_gradient[0])
    assert np.isclose(subnormal_value, 0.5 * np.log(3.0), atol=1.0e-14)
    assert np.isclose(subnormal_gradient[0], 1.0 / 3.0)


def test_optimizer_cost_conditioning_arguments_are_fail_closed():
    design = nuisance_profiled_identifiable_subspace(
        np.eye(2),
        np.zeros(2),
        np.ones(2),
        optical_parameter_count=2,
        optical_minimum_singular_value=0.1,
    )
    bounds = (-np.ones(2), np.ones(2))

    with pytest.raises(ValueError, match="optimizer_cost_transform"):
        fit_nuisance_profiled_polytope(
            lambda parameters: parameters,
            design,
            bounds=bounds,
            optimizer_cost_transform="soft_l1",
        )
    with pytest.raises(TypeError, match="must be a string"):
        fit_nuisance_profiled_polytope(
            lambda parameters: parameters,
            design,
            bounds=bounds,
            optimizer_cost_transform=None,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        fit_nuisance_profiled_polytope(
            lambda parameters: parameters,
            design,
            bounds=bounds,
            optimizer_cost_transform="log_soft_norm",
            optimizer_residual_norm_scale=0.0,
        )
    with pytest.raises(ValueError, match="squared must be finite and positive"):
        fit_nuisance_profiled_polytope(
            lambda parameters: parameters,
            design,
            bounds=bounds,
            optimizer_cost_transform="log_soft_norm",
            optimizer_residual_norm_scale=1.0e200,
        )
    with pytest.raises(ValueError, match="only meaningful"):
        fit_nuisance_profiled_polytope(
            lambda parameters: parameters,
            design,
            bounds=bounds,
            optimizer_residual_norm_scale=2.0,
        )


def test_full_polytope_profile_uses_complete_conditional_slice():
    dimension = 4
    epsilon = 0.1
    design = nuisance_profiled_identifiable_subspace(
        np.eye(dimension),
        np.zeros(dimension),
        np.ones(dimension),
        optical_parameter_count=dimension,
        optical_minimum_singular_value=0.1,
    )

    def residual(parameters):
        return jnp.asarray(
            (
                parameters[0] + parameters[1],
                epsilon * parameters[1],
                epsilon * parameters[2],
                epsilon * parameters[3],
            )
        )

    bounds = (np.full(dimension, -1.0), np.full(dimension, 1.0))
    fit = fit_nuisance_profiled_polytope(
        residual,
        design,
        bounds=bounds,
        max_nfev=100,
    )
    profile = profile_nuisance_profiled_polytope_direction(
        residual,
        fit,
        np.asarray((1.0, 0.0, 0.0, 0.0)),
        np.asarray((-0.8, 0.0, 0.8)),
        bounds=bounds,
        multistart=3,
        seed=31,
        hit_and_run_burnin=16,
        hit_and_run_thinning=3,
        max_nfev=100,
    )

    expected_compensation = -0.8 / (1.0 + epsilon**2)
    expected_cost = 0.5 * 0.8**2 * epsilon**2 / (1.0 + epsilon**2)
    assert np.all(profile.success)
    assert np.isclose(
        profile.conditional_parameters[-1, 1], expected_compensation, atol=2e-5
    )
    assert abs(profile.conditional_parameters[-1, 1]) > 1.0 / 3.0
    assert np.isclose(profile.costs[-1], expected_cost, rtol=2e-4)
    assert np.isclose(profile.costs[0], expected_cost, rtol=2e-4)
    assert np.isclose(profile.costs[1], 0.0, atol=1e-10)


def test_full_polytope_profile_follows_compensated_local_null_family():
    design = nuisance_profiled_identifiable_subspace(
        np.asarray(((1.0, 0.0),)),
        np.zeros(2),
        np.ones(2),
        optical_parameter_count=2,
        optical_minimum_singular_value=0.5,
    )

    def residual(parameters):
        return jnp.asarray((parameters[0] - parameters[1] ** 2,))

    bounds = (np.full(2, -1.0), np.full(2, 1.0))
    fit = fit_nuisance_profiled_polytope(
        residual,
        design,
        bounds=bounds,
        max_nfev=100,
    )
    null_direction = design.subspace.null_basis_scaled[:, 0]
    profile = profile_nuisance_profiled_polytope_direction(
        residual,
        fit,
        null_direction,
        np.asarray((-0.5, 0.0, 0.5)),
        bounds=bounds,
        multistart=2,
        seed=7,
        hit_and_run_burnin=8,
        hit_and_run_thinning=2,
        max_nfev=100,
    )

    assert np.all(profile.success)
    assert np.allclose(profile.delta_cost, 0.0, atol=1e-10)
    assert np.allclose(profile.conditional_parameters[[0, 2], 0], 0.25, atol=2e-5)


def test_full_polytope_profile_accepts_off_box_base_with_feasible_compensation():
    # Fixing the (1, 1) direction at +0.5 moves the uncorrected base past the
    # x=1 face.  The exact conditional slice is nevertheless non-empty because
    # the orthogonal (-1, 1) direction can compensate it.
    target = np.asarray((0.9, -0.9))

    def residual(parameters):
        return parameters - jnp.asarray(target)

    design = nuisance_profiled_identifiable_subspace(
        np.eye(2),
        np.zeros(2),
        np.ones(2),
        optical_parameter_count=2,
        optical_minimum_singular_value=0.1,
    )
    bounds = (np.full(2, -1.0), np.full(2, 1.0))
    fit = fit_nuisance_profiled_polytope(
        residual, design, bounds=bounds, max_nfev=100
    )
    direction = np.asarray((1.0, 1.0))
    profile = profile_nuisance_profiled_polytope_direction(
        residual,
        fit,
        direction,
        np.asarray((0.0, 0.5)),
        bounds=bounds,
        multistart=3,
        seed=17,
        hit_and_run_burnin=8,
        hit_and_run_thinning=2,
        max_nfev=100,
    )

    assert np.all(profile.success)
    assert np.all(profile.conditional_parameters >= bounds[0] - 1e-10)
    assert np.all(profile.conditional_parameters <= bounds[1] + 1e-10)
    scaled_displacements = (
        profile.conditional_parameters - fit.parameters
    ) / design.subspace.parameter_scales
    assert np.allclose(
        scaled_displacements @ profile.direction_scaled,
        profile.grid,
        atol=1e-10,
    )
    assert np.isclose(profile.conditional_parameters[-1, 0], 1.0, atol=2e-6)


def test_full_polytope_profile_preserves_mixed_unit_fixed_direction():
    scales = np.asarray((1.0e-12, 1.0))
    target = np.asarray((0.9e-12, -0.9))

    def residual(parameters):
        return (parameters - jnp.asarray(target)) / jnp.asarray(scales)

    design = nuisance_profiled_identifiable_subspace(
        np.diag(1.0 / scales),
        np.zeros(2),
        scales,
        optical_parameter_count=2,
        optical_minimum_singular_value=0.1,
    )
    bounds = (-scales, scales)
    fit = fit_nuisance_profiled_polytope(
        residual, design, bounds=bounds, max_nfev=100
    )
    profile = profile_nuisance_profiled_polytope_direction(
        residual,
        fit,
        np.ones(2),
        np.asarray((0.0, 0.5)),
        bounds=bounds,
        multistart=2,
        seed=23,
        hit_and_run_burnin=8,
        hit_and_run_thinning=2,
        max_nfev=100,
    )

    scaled_displacements = (profile.conditional_parameters - fit.parameters) / scales
    assert np.all(profile.success)
    assert np.allclose(
        scaled_displacements @ profile.direction_scaled,
        profile.grid,
        atol=1e-10,
    )
    assert profile.conditional_parameters[-1, 0] <= scales[0] * (1.0 + 1e-10)
    assert np.isclose(
        profile.conditional_parameters[-1, 0], scales[0], rtol=2e-6
    )


def test_full_polytope_profile_uses_deterministic_balanced_multichain_plans(
    monkeypatch,
):
    design = nuisance_profiled_identifiable_subspace(
        np.eye(4),
        np.zeros(4),
        np.ones(4),
        optical_parameter_count=4,
        optical_minimum_singular_value=0.1,
    )
    bounds = (np.full(4, -1.0), np.full(4, 1.0))
    fit = fit_nuisance_profiled_polytope(
        lambda parameters: parameters,
        design,
        bounds=bounds,
        max_nfev=100,
    )
    original_planner = profiled_module._hit_and_run_polytope_start_plan
    recorded: list[tuple[object, np.ndarray, np.ndarray, np.ndarray]] = []

    def recording_planner(*args, **kwargs):
        plan = original_planner(*args, **kwargs)
        recorded.append(
            (
                plan,
                np.asarray(args[2]).copy(),
                np.asarray(args[3]).copy(),
                np.asarray(args[4]).copy(),
            )
        )
        return plan

    monkeypatch.setattr(
        profiled_module, "_hit_and_run_polytope_start_plan", recording_planner
    )
    grid = np.asarray((-0.2, 0.0, 0.2))

    def run(seed, chains=None):
        first_record = len(recorded)
        result = profile_nuisance_profiled_polytope_direction(
            lambda parameters: parameters,
            fit,
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            grid,
            bounds=bounds,
            multistart=6,
            seed=seed,
            hit_and_run_burnin=2,
            hit_and_run_thinning=1,
            hit_and_run_chains=chains,
            max_nfev=100,
        )
        return result, recorded[first_record:]

    first_result, first_plans = run(53)
    repeated_result, repeated_plans = run(53)
    changed_result, changed_plans = run(54)
    _two_chain_result, two_chain_plans = run(53, chains=2)

    assert np.array_equal(first_result.costs, repeated_result.costs)
    assert np.array_equal(
        first_result.conditional_parameters,
        repeated_result.conditional_parameters,
    )
    assert np.array_equal(first_result.costs, changed_result.costs)
    assert len(first_plans) == len(repeated_plans) == len(changed_plans) == 3
    for grid_index, (first_record, repeated_record, changed_record) in enumerate(
        zip(first_plans, repeated_plans, changed_plans, strict=True)
    ):
        first_plan, matrix, lower, upper = first_record
        repeated_plan = repeated_record[0]
        changed_plan = changed_record[0]
        assert first_plan.starts.shape == (6, 3)
        assert np.array_equal(first_plan.chain_sample_counts, (2, 1, 1, 1))
        assert np.array_equal(first_plan.starts, repeated_plan.starts)
        assert np.array_equal(
            first_plan.chain_seed_words, repeated_plan.chain_seed_words
        )
        assert not np.array_equal(first_plan.starts[1:], changed_plan.starts[1:])
        assert np.count_nonzero(
            np.all(first_plan.starts == first_plan.starts[0], axis=1)
        ) == 1
        embedded = first_plan.starts @ matrix.T
        assert np.all(embedded >= lower[None, :])
        assert np.all(embedded <= upper[None, :])
        expected_root_seed = 53 + 104729 * grid_index
        expected_seed_words = np.asarray(
            [
                child.generate_state(1, dtype=np.uint64)[0]
                for child in np.random.SeedSequence(expected_root_seed).spawn(4)
            ],
            dtype=np.uint64,
        )
        assert np.array_equal(first_plan.chain_seed_words, expected_seed_words)

    # Grid zero must retain the global-fit point as the first conditional
    # start, even when multiple independent sampling chains are requested.
    assert np.array_equal(first_plans[1][0].starts[0], np.zeros(3))
    assert all(
        np.array_equal(record[0].chain_sample_counts, (3, 2))
        for record in two_chain_plans
    )


def test_full_polytope_profile_multistart_one_uses_only_base_start(monkeypatch):
    design = nuisance_profiled_identifiable_subspace(
        np.eye(3),
        np.zeros(3),
        np.ones(3),
        optical_parameter_count=3,
        optical_minimum_singular_value=0.1,
    )
    bounds = (np.full(3, -1.0), np.full(3, 1.0))
    fit = fit_nuisance_profiled_polytope(
        lambda parameters: parameters,
        design,
        bounds=bounds,
        max_nfev=100,
    )
    monkeypatch.setattr(
        profiled_module,
        "_active_polytope_interior_point",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("one-start profile must not initialize a sampling chain")
        ),
    )

    profile = profile_nuisance_profiled_polytope_direction(
        lambda parameters: parameters,
        fit,
        np.asarray((1.0, 0.0, 0.0)),
        np.asarray((0.0, 0.2)),
        bounds=bounds,
        multistart=1,
        hit_and_run_chains=19,
        max_nfev=100,
    )

    assert np.all(profile.success)
    assert np.allclose(profile.conditional_parameters[:, 1:], 0.0, atol=1e-12)


@pytest.mark.parametrize("chain_count", (0, -1, 1.5, True))
def test_full_polytope_profile_rejects_invalid_chain_count(chain_count):
    design = nuisance_profiled_identifiable_subspace(
        np.eye(2),
        np.zeros(2),
        np.ones(2),
        optical_parameter_count=2,
        optical_minimum_singular_value=0.1,
    )
    bounds = (np.full(2, -1.0), np.full(2, 1.0))
    fit = fit_nuisance_profiled_polytope(
        lambda parameters: parameters,
        design,
        bounds=bounds,
        max_nfev=100,
    )

    with pytest.raises(ValueError, match="hit_and_run_chains"):
        profile_nuisance_profiled_polytope_direction(
            lambda parameters: parameters,
            fit,
            np.asarray((1.0, 0.0)),
            np.asarray((0.0, 0.2)),
            bounds=bounds,
            hit_and_run_chains=chain_count,
        )


def test_full_polytope_profile_does_not_compute_redundant_lp_marginals(monkeypatch):
    design = nuisance_profiled_identifiable_subspace(
        np.eye(3),
        np.zeros(3),
        np.ones(3),
        optical_parameter_count=3,
        optical_minimum_singular_value=0.1,
    )
    bounds = (np.full(3, -1.0), np.full(3, 1.0))
    fit = fit_nuisance_profiled_polytope(
        lambda parameters: parameters,
        design,
        bounds=bounds,
        max_nfev=100,
    )

    def forbidden_marginal_lp(*args, **kwargs):
        raise AssertionError("conditional profiles must use only the exact constraints")

    monkeypatch.setattr(
        profiled_module,
        "_active_polytope_marginal_bounds",
        forbidden_marginal_lp,
    )
    profile = profile_nuisance_profiled_polytope_direction(
        lambda parameters: parameters,
        fit,
        np.asarray((1.0, 1.0, 0.0)),
        np.asarray((-0.2, 0.0, 0.2)),
        bounds=bounds,
        multistart=2,
        seed=29,
        hit_and_run_burnin=4,
        hit_and_run_thinning=1,
        max_nfev=100,
    )

    assert np.all(profile.success)


def test_full_polytope_profile_selects_known_lower_cost_failed_candidate(monkeypatch):
    # A feasible point is still a valid upper bound on the conditional minimum
    # when SLSQP exhausts maxiter.  Never discard it in favour of a higher-cost
    # point merely because the latter reports success: doing so could overstate
    # profile curvature.  The selected point remains explicitly unsuccessful.
    design = nuisance_profiled_identifiable_subspace(
        np.eye(2),
        np.zeros(2),
        np.ones(2),
        optical_parameter_count=2,
        optical_minimum_singular_value=0.1,
    )
    bounds = (np.full(2, -1.0), np.full(2, 1.0))
    fit = fit_nuisance_profiled_polytope(
        lambda parameters: parameters,
        design,
        bounds=bounds,
        max_nfev=100,
    )
    fake_results = iter(
        SimpleNamespace(x=np.asarray((x,)), fun=reported, success=success)
        for _coordinate in range(2)
        for x, reported, success in (
            (0.0, 99.0, False),
            (0.5, 0.0, True),
        )
    )
    monkeypatch.setattr(
        profiled_module, "minimize", lambda *args, **kwargs: next(fake_results)
    )

    profile = profile_nuisance_profiled_polytope_direction(
        lambda parameters: parameters,
        fit,
        np.asarray((1.0, 0.0)),
        np.asarray((0.0, 0.2)),
        bounds=bounds,
        multistart=2,
        seed=31,
        hit_and_run_burnin=0,
        hit_and_run_thinning=1,
    )

    assert not np.any(profile.success)
    assert np.allclose(profile.conditional_parameters[:, 1], 0.0)
    assert np.allclose(profile.costs, (0.0, 0.5 * 0.2**2))


@pytest.mark.parametrize(
    "keyword,value",
    (
        ("multistart", 1.5),
        ("seed", -1),
        ("hit_and_run_burnin", 0.5),
        ("hit_and_run_thinning", 0),
        ("hit_and_run_chains", 0),
        ("hit_and_run_chains", 1.5),
        ("max_nfev", True),
    ),
)
def test_full_polytope_rejects_silently_truncated_integer_options(keyword, value):
    design = nuisance_profiled_identifiable_subspace(
        np.eye(2),
        np.zeros(2),
        np.ones(2),
        optical_parameter_count=2,
        optical_minimum_singular_value=0.1,
    )

    with pytest.raises(ValueError, match=keyword):
        fit_nuisance_profiled_polytope(
            lambda parameters: parameters,
            design,
            bounds=(np.full(2, -1.0), np.full(2, 1.0)),
            **{keyword: value},
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"optical_parameter_count": 0, "optical_minimum_singular_value": 1.0},
        {"optical_parameter_count": 1, "optical_minimum_singular_value": 0.0},
        {
            "optical_parameter_count": 1,
            "optical_minimum_singular_value": 1.0,
            "nuisance_rank_rtol": 0.0,
        },
    ),
)
def test_builder_rejects_invalid_threshold_contract(kwargs):
    with pytest.raises(ValueError):
        nuisance_profiled_identifiable_subspace(
            np.eye(2), np.zeros(2), np.ones(2), **kwargs
        )
