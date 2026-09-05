from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.microscope_inversion import (
    CalibratedSourceState,
    DEFAULT_LENS_CHANNELS,
    FrameObservationBlock,
    HeterogeneousInversionDataset,
    IlluminationProjection2D,
    InversionDataset,
    SamplePlaneState,
    calibrated_source_basis,
    compact_30_state_plan,
    compact_illumination_moment_experiment,
    compose_matrices,
    dac_hard_edge_solenoid_matrix,
    d_optimal_logdet,
    diagnose_multistart_basins,
    drift_matrix,
    extract_position_tilt_response,
    equivalent_solution_family,
    equivalent_solutions,
    fit_dataset,
    fit_heterogeneous_dataset,
    fit_identifiable_subspace,
    fit_least_squares,
    generic_round_symplectic_matrix,
    glaser_solenoid_matrix,
    greedy_d_optimal_selection,
    hard_edge_solenoid_matrix,
    illumination_covariance_design_matrix,
    jax_jacobian,
    jacobian_svd,
    larmor_rotation_4x4,
    mathematical_sinc,
    predict_acquisition_plan,
    predict_illumination_experiment,
    profile_likelihood,
    profile_likelihood_direction,
    propagate_sample_state,
    response_observable_vector,
    scaled_identifiable_subspace,
    smooth_field_solenoid_matrix,
    symplectic_defect,
    standard_illumination_projections,
    thin_lens_matrix,
)


jax.config.update("jax_enable_x64", True)


def assert_symplectic(matrix, atol=2.0e-11):
    np.testing.assert_allclose(
        np.asarray(symplectic_defect(matrix)), np.zeros((4, 4)), atol=atol
    )


def test_mathematical_sinc_is_stable_at_zero():
    assert float(mathematical_sinc(0.0)) == pytest.approx(1.0)
    assert float(jax.grad(mathematical_sinc)(0.0)) == pytest.approx(0.0, abs=1e-15)
    near_zero = float(mathematical_sinc(1.0e-8))
    assert near_zero == pytest.approx(1.0 - 1.0e-16 / 6.0, abs=1e-15)


def test_thin_hard_edge_and_larmor_matrices_are_symplectic():
    assert_symplectic(thin_lens_matrix(power=3.2, rotation=0.17))
    assert_symplectic(larmor_rotation_4x4(-0.43))
    assert_symplectic(hard_edge_solenoid_matrix(q=0.71, length=0.083))


def test_hard_edge_zero_field_and_thin_limit():
    length = 0.12
    np.testing.assert_allclose(
        np.asarray(hard_edge_solenoid_matrix(0.0, length)),
        np.asarray(drift_matrix(length)),
        atol=1e-14,
    )

    power = 2.5
    short_length = 1.0e-14
    q = np.sqrt(power * short_length)
    np.testing.assert_allclose(
        np.asarray(hard_edge_solenoid_matrix(q, short_length)),
        np.asarray(thin_lens_matrix(power)),
        atol=2.0e-6,
    )


def test_dac_hard_edge_law_uses_raw_control_displacement():
    expected_q = 0.2 + 0.03 * (17.0 - 12.0) - 0.001 * (17.0 - 12.0) ** 2
    actual = dac_hard_edge_solenoid_matrix(
        dac=17.0,
        length=0.05,
        q_coefficients=(0.2, 0.03, -0.001),
        reference_dac=12.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(hard_edge_solenoid_matrix(expected_q, 0.05)),
        atol=1e-13,
    )


def test_field_integrators_and_generic_section_are_symplectic():
    edges = jnp.linspace(-0.08, 0.09, 97)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    alpha = 4.0 / (1.0 + (midpoints / 0.025) ** 2)
    assert_symplectic(smooth_field_solenoid_matrix(edges, alpha), atol=5e-11)
    assert_symplectic(
        glaser_solenoid_matrix(
            q=0.8,
            center=0.004,
            width=0.021,
            z_min=-0.09,
            z_max=0.11,
            n_slices=160,
        ),
        atol=5e-11,
    )
    assert_symplectic(
        generic_round_symplectic_matrix(
            phase=0.7,
            log_magnification=-0.23,
            shear=0.31,
            rotation=-0.19,
            length_scale=0.4,
        )
    )

    def differentiable_glaser(parameters):
        return glaser_solenoid_matrix(
            q=parameters[0],
            center=0.004,
            width=parameters[1],
            z_min=-0.09,
            z_max=0.11,
            n_slices=64,
        )

    derivative = jax.jacfwd(differentiable_glaser)(jnp.asarray((0.8, 0.021)))
    assert derivative.shape == (4, 4, 2)
    assert np.all(np.isfinite(np.asarray(derivative)))


def test_compact_plan_is_deterministic_and_has_requested_budget():
    targets = (40_000.0, 80_000.0, 150_000.0, 300_000.0, 600_000.0)
    first = compact_30_state_plan(target_magnifications=targets)
    second = compact_30_state_plan()

    assert first.channel_names == DEFAULT_LENS_CHANNELS
    assert DEFAULT_LENS_CHANNELS == (
        "CL1",
        "CL3",
        "C-mini",
        "IL1",
        "IL2",
        "IL3",
        "PL1",
    )
    assert first.n_states == 30
    assert first.role_counts() == {
        "nominal": 5,
        "single_lens": 14,
        "d_optimal": 8,
        "held_out": 3,
    }
    assert first.total_frames == 230
    assert first.frame_counts() == {
        "nominal": 60,
        "single_lens": 98,
        "d_optimal": 48,
        "held_out": 24,
    }
    np.testing.assert_array_equal(first.control_array(), second.control_array())
    assert first.training_mask.sum() == 27
    assert first.held_out_mask.sum() == 3
    assert tuple(
        state.target_magnification
        for state in first.states
        if state.role == "nominal"
    ) == targets


def test_single_lens_states_change_exactly_one_raw_dac():
    references = np.asarray((10, 20, 30, 40, 50, 60, 70), dtype=float)
    perturbations = np.asarray((1, 2, 3, 4, 5, 6, 7), dtype=float)
    plan = compact_30_state_plan(
        reference_dacs=references, perturbations=perturbations
    )
    single_states = [state for state in plan.states if state.role == "single_lens"]

    for state in single_states:
        difference = np.asarray(state.dac_values) - references
        changed = np.flatnonzero(difference)
        assert changed.size == 1
        channel_index = plan.channel_names.index(state.perturbed_channel)
        assert changed[0] == channel_index
        assert abs(difference[channel_index]) == pytest.approx(
            perturbations[channel_index]
        )


def test_position_and_tilt_response_observables_for_a_drift():
    distance = 0.37
    matrix = drift_matrix(distance)
    response = extract_position_tilt_response(matrix)

    np.testing.assert_allclose(response.position_from_position, np.eye(2))
    np.testing.assert_allclose(
        response.position_from_tilt, distance * np.eye(2)
    )
    np.testing.assert_allclose(response.tilt_from_position, np.zeros((2, 2)))
    np.testing.assert_allclose(response.tilt_from_tilt, np.eye(2))
    np.testing.assert_allclose(
        response_observable_vector(matrix),
        np.asarray((1.0, 0.0, 0.0, 1.0, distance, 0.0, 0.0, distance)),
    )

    state = SamplePlaneState(x=2.0e-6, theta_y=3.0e-3)
    propagated = propagate_sample_state(matrix, state)
    np.testing.assert_allclose(
        propagated,
        np.asarray((2.0e-6, 0.0, distance * 3.0e-3, 3.0e-3)),
    )


def test_illumination_covariance_tomography_exposes_missing_round_optics_mode():
    ordinary = standard_illumination_projections(
        defocus_distance_m=0.12,
        diffraction_length_m=0.4,
    )
    ordinary_design = illumination_covariance_design_matrix(ordinary)

    assert ordinary_design.shape == (12, 10)
    assert np.linalg.matrix_rank(ordinary_design) == 9
    # Ordinary image/defocus/diffraction projections see only the sum of the
    # two cross-plane position--angle covariances.  Their antisymmetric
    # difference (canonical angular momentum) is an exact null direction.
    missing_mode = np.zeros(10)
    missing_mode[3] = 1.0   # covariance_x_theta_y
    missing_mode[5] = -1.0  # covariance_theta_x_y
    np.testing.assert_allclose(ordinary_design @ missing_mode, 0.0, atol=1e-14)

    complete = standard_illumination_projections(
        defocus_distance_m=0.12,
        diffraction_length_m=0.4,
        coupled_length_m=0.17,
    )
    complete_design = illumination_covariance_design_matrix(complete)
    assert complete_design.shape == (15, 10)
    assert np.linalg.matrix_rank(complete_design) == 10
    assert np.linalg.norm(complete_design @ missing_mode) > 0.1

    affine = np.asarray(((1.02, 0.03), (-0.01, 0.98)))
    calibrated = IlluminationProjection2D.from_transfer_matrix(
        "calibrated_defocus",
        drift_matrix(0.12),
        detector_affine_xy=affine,
    )
    expected = affine @ np.asarray(drift_matrix(0.12))[(0, 2), :]
    np.testing.assert_allclose(calibrated.position_map, expected)

    with pytest.raises(ValueError, match="commuting"):
        IlluminationProjection2D(
            "unphysical_x_and_theta_x",
            np.asarray(((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))),
        )


def test_compact_illumination_experiment_predicts_physical_image_moments():
    covariance = np.diag((4.0e-12, 9.0e-8, 2.5e-12, 4.0e-8))
    sources = calibrated_source_basis(
        covariance,
        position_step_m=1.0e-6,
        tilt_step_rad=2.0e-4,
        symmetric=False,
    )
    projections = standard_illumination_projections(
        defocus_distance_m=0.08,
        diffraction_length_m=0.3,
        coupled_length_m=0.05,
    )
    experiment = compact_illumination_moment_experiment(sources, projections)

    assert len(sources) == 5
    assert experiment.n_frames == 13
    assert len(experiment.observable_names) == 5 * experiment.n_frames
    prediction = predict_illumination_experiment(drift_matrix(0.2), experiment)
    assert prediction.shape == (13, 5)
    np.testing.assert_allclose(prediction[0, :2], 0.0, atol=1e-15)
    expected_covariance = np.asarray(drift_matrix(0.2)) @ covariance @ np.asarray(
        drift_matrix(0.2)
    ).T
    np.testing.assert_allclose(
        prediction[0, 2:],
        expected_covariance[np.ix_((0, 2), (0, 2))][(0, 0, 1), (0, 1, 1)],
    )

    derivative = jax.jacfwd(
        lambda distance: predict_illumination_experiment(
            drift_matrix(distance), experiment
        )
    )(0.2)
    assert derivative.shape == (13, 5)
    assert np.all(np.isfinite(np.asarray(derivative)))

    with pytest.raises(ValueError, match="positive semidefinite"):
        CalibratedSourceState(np.zeros(4), np.diag((1.0, 1.0, 1.0, -0.1)))


def test_upstream_image_moments_add_condenser_rank_that_sample_inputs_cannot():
    covariance = np.diag((2.0e-12, 3.0e-8, 1.5e-12, 2.0e-8))
    sources = calibrated_source_basis(
        covariance,
        position_step_m=0.8e-6,
        tilt_step_rad=0.15e-3,
        symmetric=False,
    )
    projections = standard_illumination_projections(
        defocus_distance_m=0.06,
        diffraction_length_m=0.25,
        coupled_length_m=0.04,
    )
    experiment = compact_illumination_moment_experiment(sources, projections)
    controls = np.zeros((7, 3), dtype=float)
    for row, (channel, sign) in enumerate(
        ((0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0), (2, -1.0), (2, 1.0)),
        start=1,
    ):
        controls[row, channel] = 0.1 * sign
    controls = jnp.asarray(controls)

    def source_to_sample(parameters, control):
        return compose_matrices(
            drift_matrix(0.025),
            hard_edge_solenoid_matrix(parameters[0] + 0.4 * control[0], 0.008),
            drift_matrix(0.032),
            hard_edge_solenoid_matrix(parameters[1] + 0.5 * control[1], 0.007),
            drift_matrix(0.027),
            hard_edge_solenoid_matrix(parameters[2] + 0.3 * control[2], 0.006),
            drift_matrix(0.018),
        )

    def upstream_image_prediction(parameters):
        return jnp.concatenate(
            tuple(
                predict_illumination_experiment(
                    source_to_sample(parameters, control), experiment
                ).reshape(-1)
                for control in controls
            )
        )

    parameters = jnp.asarray((0.22, -0.17, 0.13))
    upstream_jacobian = jax_jacobian(upstream_image_prediction, parameters)
    assert jacobian_svd(upstream_jacobian, rtol=1e-9).rank == 3

    calibrated_sample_states = jnp.asarray(
        tuple(source.mean_state for source in sources)
    )
    downstream_matrix = drift_matrix(0.4)

    def downstream_images_with_calibrated_sample_inputs(parameters):
        propagated = (downstream_matrix @ calibrated_sample_states.T).T
        detector_centroids = propagated[:, jnp.asarray((0, 2))]
        return detector_centroids.reshape(-1) + 0.0 * jnp.sum(parameters)

    downstream_jacobian = jax_jacobian(
        downstream_images_with_calibrated_sample_inputs, parameters
    )
    assert jacobian_svd(downstream_jacobian, rtol=1e-9).rank == 0


def test_jax_jacobian_and_svd_diagnostics():
    def function(parameters):
        return jnp.asarray(
            (parameters[0] + 2.0 * parameters[1], parameters[0] * parameters[1])
        )

    jacobian = jax_jacobian(function, jnp.asarray((3.0, 4.0)))
    np.testing.assert_allclose(jacobian, np.asarray(((1.0, 2.0), (4.0, 3.0))))
    diagnostics = jacobian_svd(jacobian)
    assert diagnostics.rank == 2
    assert diagnostics.nullity == 0
    assert diagnostics.smallest_to_largest > 0.0
    assert np.isfinite(diagnostics.condition_number)

    underdetermined = jacobian_svd(np.asarray(((1.0, 0.0, 0.0),)))
    assert underdetermined.rank == 1
    assert underdetermined.nullity == 2
    assert underdetermined.right_singular_vectors.shape == (3, 3)
    assert underdetermined.smallest_to_largest == 0.0
    assert np.isinf(underdetermined.condition_number)


def test_greedy_d_optimal_selection_builds_full_rank_information():
    candidates = np.asarray(((1.0, 0.0), (0.0, 3.0), (0.2, 0.2)))
    selection = greedy_d_optimal_selection(candidates, n_select=2)

    assert len(selection.indices) == 2
    assert len(set(selection.indices)) == 2
    assert np.linalg.matrix_rank(selection.information) == 2
    selected = candidates[np.asarray(selection.indices)]
    assert np.isfinite(d_optimal_logdet(selected))

    # A zero ridge still resolves all initially rank-deficient ties deterministically.
    no_ridge = greedy_d_optimal_selection(candidates, n_select=2, ridge=0.0)
    assert len(no_ridge.indices) == 2


def test_d_optimal_logdet_stays_finite_for_extreme_rank_deficiency():
    jacobian = np.asarray(
        ((1.0e12, 1.0e12, 0.0), (1.0e12, 1.0e12, 0.0))
    )
    ridge = 1.0e-12

    score = d_optimal_logdet(jacobian, ridge=ridge)
    expected = np.log(4.0e24 + ridge) + 2.0 * np.log(ridge)

    assert np.isfinite(score)
    assert score == pytest.approx(expected, rel=2.0e-14, abs=2.0e-14)


def test_d_optimal_logdet_retains_a_small_psd_prior_at_large_scale():
    jacobian = np.asarray(((1.0e12, 1.0e12, 0.0),))
    prior = np.diag((1.0e-6, 1.0e-6, 1.0e-3))
    ridge = 1.0e-12

    score = d_optimal_logdet(
        jacobian, prior_precision=prior, ridge=ridge
    )
    expected = (
        np.log(2.0e24 + 1.0e-6 + ridge)
        + np.log(1.0e-6 + ridge)
        + np.log(1.0e-3 + ridge)
    )

    assert np.isfinite(score)
    assert score == pytest.approx(expected, rel=2.0e-14, abs=2.0e-14)


def test_extreme_scale_greedy_selection_prefers_independent_information():
    candidates = np.asarray(
        (
            (1.0e12, 1.0e12),
            (1.0e12, 1.0e12),
            (0.0, 1.0e6),
        )
    )

    first = greedy_d_optimal_selection(candidates, n_select=2, ridge=1.0e-12)
    second = greedy_d_optimal_selection(candidates, n_select=2, ridge=1.0e-12)

    assert first.indices == (0, 2)
    assert second.indices == first.indices
    assert np.all(np.isfinite(first.logdet_history))
    assert first.logdet_history[1] > first.logdet_history[0]


def test_fit_dataset_recovers_hard_edge_length_and_dac_law():
    plan = compact_30_state_plan(perturbations=0.8)
    true_parameters = jnp.asarray((0.064, 0.24, 0.075))

    def prediction(parameters):
        length, q0, q_slope = parameters

        def sample_to_detector(controls):
            q = q0 + q_slope * controls[0]
            return compose_matrices(
                hard_edge_solenoid_matrix(q, length), drift_matrix(0.31)
            )

        return predict_acquisition_plan(plan, sample_to_detector)

    exact_values = np.asarray(prediction(true_parameters))
    dataset = InversionDataset(
        plan=plan,
        values=exact_values,
        standard_deviations=np.asarray(
            (1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 1e-6, 1e-6, 1e-6)
        ),
    )
    result = fit_dataset(
        prediction,
        dataset,
        x0=np.asarray((0.09, 0.18, 0.11)),
        bounds=(np.asarray((0.02, -0.8, -0.4)), np.asarray((0.15, 0.8, 0.4))),
        parameter_names=("length", "q0", "q_slope"),
        loss="linear",
        multistart=2,
        seed=4,
        max_nfev=300,
    )

    assert result.success
    np.testing.assert_allclose(result.parameters, true_parameters, rtol=2e-6, atol=2e-8)
    assert result.parameter_mapping["length"] == pytest.approx(0.064, rel=2e-6)
    assert result.weighted_rmse < 1e-6
    assert result.singular_values.size == 3
    assert result.solution_parameters.shape == (2, 3)
    assert equivalent_solutions(result, delta_cost=1.0).parameters.shape[1] == 3


def test_equivalent_solution_family_reports_scaled_null_directions():
    jacobian = np.asarray(((1.0, 2.0, 0.0), (0.0, 0.0, 3.0)))
    parameters = np.asarray((2.0, 4.0, 0.1))
    family = equivalent_solution_family(
        jacobian,
        parameters,
        parameter_names=("a", "b", "c"),
        parameter_scales=(1.0, 1.0, 1.0),
    )
    assert family.rank == 2
    assert family.nullity == 1
    np.testing.assert_allclose(jacobian @ family.basis, 0.0, atol=1e-13)
    displaced = family.displaced(np.asarray((-0.5, 0.0, 0.5))[:, None])
    assert displaced.shape == (3, 3)
    np.testing.assert_allclose(
        (displaced - parameters) @ jacobian.T, 0.0, atol=1e-13
    )


def test_rank_deficient_fit_reports_infinite_not_zero_null_uncertainty():
    result = fit_least_squares(
        lambda parameters: jnp.asarray((parameters[0] - 2.0,)),
        x0=np.asarray((0.0, 7.0)),
        loss="linear",
        max_nfev=30,
    )
    assert result.jacobian_rank == 1
    assert result.null_space_basis.shape == (2, 1)
    assert np.isfinite(result.parameter_standard_errors[0])
    assert np.isinf(result.parameter_standard_errors[1])
    assert np.isnan(result.covariance[1, 1])
    assert result.identifiable_covariance[1, 1] == pytest.approx(0.0)


def test_heterogeneous_frame_blocks_use_full_covariances_and_fit():
    plan = compact_30_state_plan()
    truth = np.asarray((1.2, -0.35))

    def model(parameters):
        a, b = parameters
        return (jnp.asarray((a + 2.0 * b,)), jnp.asarray((a - b, 2.0 * a + b)))

    exact = model(jnp.asarray(truth))
    dataset = HeterogeneousInversionDataset(
        plan,
        (
            FrameObservationBlock(0, 0, np.asarray(exact[0]), np.asarray(((0.04,),))),
            FrameObservationBlock(
                1,
                1,
                np.asarray(exact[1]),
                np.asarray(((0.09, 0.018), (0.018, 0.16))),
            ),
        ),
    )
    assert dataset.n_observations == 3
    result = fit_heterogeneous_dataset(
        model,
        dataset,
        np.asarray((0.0, 0.0)),
        parameter_names=("a", "b"),
        loss="linear",
        max_nfev=50,
    )
    np.testing.assert_allclose(result.parameters, truth, atol=1e-10)
    assert result.jacobian_rank == 2


def test_profile_likelihood_refits_other_parameters():
    truth = np.asarray((7.0 / 3.0, 4.0 / 3.0))

    def residual(parameters):
        return jnp.asarray(
            (parameters[0] + 2.0 * parameters[1] - 5.0,
             parameters[0] - parameters[1] - 1.0)
        )

    grid = truth[0] + np.asarray((-0.3, 0.0, 0.3))
    profile = profile_likelihood(
        residual,
        truth,
        0,
        grid,
        parameter_names=("a", "b"),
        loss="linear",
        max_nfev=100,
    )
    assert profile.parameter_name == "a"
    assert np.argmin(profile.costs) == 1
    assert profile.costs[1] < 1e-20
    assert np.all(profile.success)


def test_identifiable_subspace_fit_embeds_only_resolved_directions_and_uses_absolute_noise():
    design = jnp.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (1.0, 2.0, 0.0),
        )
    )
    # The added vector lies in the left null space of the two resolved columns,
    # leaving a deliberately non-zero optimum residual.  Absolute-noise Fisher
    # covariance must not be inflated by that observed residual variance.
    truth = jnp.asarray((0.7, -0.4, 9.0))
    observations = design @ truth + jnp.asarray((3.0, 3.0, -3.0))

    def residual(parameters):
        return design @ parameters - observations

    reference = np.asarray((0.0, 0.0, 4.5))
    scales = np.asarray((2.0, 0.5, 3.0))
    fit = fit_identifiable_subspace(
        residual,
        reference,
        scales,
        rank_rtol=1e-8,
        bounds=(np.asarray((-3.0, -2.0, 0.0)), np.asarray((3.0, 2.0, 10.0))),
        parameter_names=("a", "b", "unidentified"),
        loss="linear",
        multistart=3,
        seed=7,
        max_nfev=100,
    )

    assert fit.success
    assert fit.subspace.rank == 2
    assert fit.subspace.nullity == 1
    np.testing.assert_allclose(fit.parameters[:2], np.asarray(truth[:2]), atol=1e-10)
    assert fit.parameters[2] == pytest.approx(reference[2])
    np.testing.assert_allclose(
        np.asarray(design) @ fit.subspace.null_basis_physical,
        0.0,
        atol=1e-13,
    )
    expected_active_fisher = np.linalg.inv(
        fit.active_fit.jacobian.T @ fit.active_fit.jacobian
    )
    np.testing.assert_allclose(
        fit.active_fisher_covariance,
        expected_active_fisher,
        rtol=1e-11,
        atol=1e-13,
    )
    residual_variance = 2.0 * fit.cost / fit.active_fit.degrees_of_freedom
    assert residual_variance > 10.0
    assert not np.allclose(
        fit.active_fisher_covariance,
        expected_active_fisher * residual_variance,
    )
    assert np.all(fit.parameters >= np.asarray((-3.0, -2.0, 0.0)))
    assert np.all(fit.parameters <= np.asarray((3.0, 2.0, 10.0)))


def test_scaled_identifiable_subspace_requires_explicit_practical_rank():
    jacobian = np.asarray(((1.0, 0.0), (0.0, 1e-5)))
    practical = scaled_identifiable_subspace(
        jacobian,
        np.zeros(2),
        np.ones(2),
        rank_rtol=1e-3,
    )
    numerical = scaled_identifiable_subspace(
        jacobian,
        np.zeros(2),
        np.ones(2),
        rank_rtol=1e-7,
    )
    assert practical.rank == 1
    assert practical.nullity == 1
    assert numerical.rank == 2
    with pytest.raises(ValueError, match="rank_rtol"):
        scaled_identifiable_subspace(
            jacobian, np.zeros(2), np.ones(2), rank_rtol=0.0
        )


def test_direction_profile_refits_orthogonal_active_modes_and_handles_null_family():
    design = jnp.asarray(((1.0, 1.0, 0.0), (1.0, -1.0, 0.0)))
    truth = jnp.asarray((0.4, -0.2, 1.5))

    def residual(parameters):
        return design @ (parameters - truth)

    bounds = (np.asarray((-1.0, -1.0, 0.0)), np.asarray((1.0, 1.0, 3.0)))
    scales = np.asarray((0.5, 0.25, 2.0))
    fit = fit_identifiable_subspace(
        residual,
        np.asarray((0.0, 0.0, 1.5)),
        scales,
        rank_rtol=1e-8,
        bounds=bounds,
        loss="linear",
        max_nfev=100,
    )
    active_direction = np.asarray((1.0, 1.0, 0.0))
    profile = profile_likelihood_direction(
        residual,
        fit,
        active_direction,
        np.asarray((-0.2, 0.0, 0.2)),
        bounds=bounds,
        direction_name="sum",
        max_nfev=100,
    )
    assert profile.orthogonal_active_basis_scaled.shape == (3, 1)
    assert np.all(profile.success)
    assert np.argmin(profile.costs) == 1
    np.testing.assert_allclose(
        ((profile.conditional_parameters - fit.parameters) / scales)
        @ profile.direction_scaled,
        profile.grid,
        atol=1e-10,
    )
    assert np.all(profile.conditional_parameters >= bounds[0])
    assert np.all(profile.conditional_parameters <= bounds[1])

    null_profile = profile_likelihood_direction(
        residual,
        fit,
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((-0.4, 0.0, 0.4)),
        bounds=bounds,
        direction_name="null",
        max_nfev=100,
    )
    assert null_profile.orthogonal_active_basis_scaled.shape == (3, 2)
    np.testing.assert_allclose(null_profile.delta_cost, 0.0, atol=1e-20)
    with pytest.raises(ValueError, match="outside the physical bounds"):
        profile_likelihood_direction(
            residual,
            fit,
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((0.0, 2.0)),
            bounds=bounds,
        )


def test_equivalent_solutions_excludes_failed_multistart_results():
    result = fit_least_squares(
        lambda parameters: jnp.asarray((parameters[0] - 1.0,)),
        np.asarray((0.0,)),
        loss="linear",
        multistart=2,
        max_nfev=20,
    )
    modified = replace(
        result,
        solution_parameters=np.asarray(((1.0,), (1.5,))),
        start_costs=np.asarray((0.0, 0.0)),
        start_success=np.asarray((True, False)),
    )
    solutions = equivalent_solutions(
        modified, delta_cost=1.0, scaled_distance_min=1e-6
    )
    np.testing.assert_allclose(solutions.parameters, np.asarray(((1.0,),)))


def test_multistart_basin_diagnostics_cost_gate_and_named_clusters():
    base = fit_least_squares(
        lambda parameters: parameters,
        np.asarray((0.2, -0.1)),
        parameter_names=("q", "length"),
        loss="linear",
        max_nfev=20,
    )
    solutions = np.asarray(
        (
            (0.0, 0.0),
            (0.04, 0.0),
            (2.0, 0.0),
            (-2.0, 0.0),
            (0.0, 0.0),
            (np.nan, 0.0),
        )
    )
    modified = replace(
        base,
        parameters=solutions[0],
        cost=5.0,
        solution_parameters=solutions,
        start_costs=np.asarray((5.0, 5.1, 5.2, 12.0, 5.05, 5.05)),
        start_success=np.asarray((True, True, True, True, False, True)),
        start_gradients=np.asarray(
            (
                (2.0, 3.0),
                (0.1, 0.2),
                (0.3, 0.4),
                (0.5, 0.6),
                (0.7, 0.8),
                (0.9, 1.0),
            )
        ),
        start_active_mask=np.zeros((6, 2), dtype=int),
    )
    diagnostics = diagnose_multistart_basins(
        modified,
        delta_cost=0.25,
        scaled_distance_threshold=0.1,
        parameter_scales=np.asarray((1.0, 2.0)),
        bounds=(np.asarray((-2.0, -4.0)), np.asarray((2.0, 4.0))),
    )

    assert diagnostics.best_start_index == 0
    assert diagnostics.best_cost == pytest.approx(5.0)
    assert diagnostics.classifications == (
        "near_best",
        "near_best",
        "near_best",
        "high_cost",
        "failed",
        "nonfinite",
    )
    np.testing.assert_allclose(
        diagnostics.delta_costs[:4], np.asarray((0.0, 0.1, 0.2, 7.0))
    )
    assert diagnostics.n_near_best_basins == 2
    assert not diagnostics.single_sampled_near_best_basin
    assert diagnostics.basins[0].member_start_indices == (0, 1)
    assert diagnostics.basins[1].member_start_indices == (2,)
    assert diagnostics.basins[1].representative_bound_hits == ("q:upper",)
    assert diagnostics.basins[1].representative_parameter_mapping == {
        "q": 2.0,
        "length": 0.0,
    }
    assert diagnostics.scaled_gradient_norms[0] == pytest.approx(6.0)
    assert diagnostics.bound_status[3, 0] == -1

    # Tightening the explicit cost gate leaves one sampled competitive basin;
    # the two successful higher-cost minima do not become a uniqueness failure.
    tighter = diagnose_multistart_basins(
        modified,
        delta_cost=0.11,
        scaled_distance_threshold=0.1,
        parameter_scales=np.asarray((1.0, 2.0)),
    )
    assert tighter.single_sampled_near_best_basin
    np.testing.assert_array_equal(tighter.high_cost_start_indices, (2, 3))


def test_fit_retains_per_start_gradient_and_bound_diagnostics():
    result = fit_least_squares(
        lambda parameters: jnp.asarray((parameters[0] - 2.0,)),
        np.asarray((0.0,)),
        bounds=(np.asarray((-1.0,)), np.asarray((1.0,))),
        parameter_names=("bounded_q",),
        loss="linear",
        multistart=2,
        seed=3,
        max_nfev=30,
    )
    assert result.start_gradients.shape == (2, 1)
    assert result.start_active_mask.shape == (2, 1)

    diagnostics = diagnose_multistart_basins(
        result,
        delta_cost=1.0e-8,
        scaled_distance_threshold=1.0e-6,
        parameter_scales=np.asarray((1.0,)),
        bounds=(np.asarray((-1.0,)), np.asarray((1.0,))),
    )
    assert diagnostics.single_sampled_near_best_basin
    assert diagnostics.basins[0].representative_bound_hits == (
        "bounded_q:upper",
    )
    assert np.isfinite(diagnostics.scaled_gradient_norms).all()
