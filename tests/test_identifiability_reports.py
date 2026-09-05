import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.identifiability_reporting import (
    assess_matched_model_recovery,
    verify_local_null_families,
)
from temgym_core.microscope_inversion import (
    equivalent_solution_family,
    scaled_identifiable_subspace,
)


def test_matched_recovery_gates_active_combinations_and_hides_null_mixed_parameters():
    subspace = scaled_identifiable_subspace(
        np.asarray(((1.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        np.zeros(3),
        np.ones(3),
        parameter_names=("lens_a", "lens_b", "detector_z"),
        rank_rtol=1.0e-8,
    )
    active_coordinates = np.asarray((0.005, -0.004))
    null_coordinates = np.asarray((0.4,))
    scaled_difference = (
        subspace.active_basis_scaled @ active_coordinates
        + subspace.null_basis_scaled @ null_coordinates
    )

    report = assess_matched_model_recovery(
        subspace,
        fitted_parameters=scaled_difference,
        truth_parameters=np.zeros(3),
    )

    np.testing.assert_allclose(
        report.active_combination_error_scaled, active_coordinates, atol=1.0e-14
    )
    np.testing.assert_allclose(
        report.null_combination_error_scaled, null_coordinates, atol=1.0e-14
    )
    assert report.maximum_active_combination_error_scaled == pytest.approx(0.005)
    assert report.active_one_percent_gate_passed
    assert report.one_percent_gate_passed
    assert report.identifiable_parameter_names == ("detector_z",)
    assert report.non_identifiable_parameter_names == ("lens_a", "lens_b")
    assert report.parameters[0].scaled_error is None
    assert report.parameters[0].one_percent_gate_passed is None
    assert report.parameters[1].scaled_error is None
    assert report.parameters[2].one_percent_gate_passed is True
    with pytest.raises(ValueError):
        report.fitted_parameters[0] = 3.0


def test_matched_recovery_one_percent_gate_is_strict_and_uses_declared_scales():
    subspace = scaled_identifiable_subspace(
        np.diag((2.0, 3.0)),
        np.zeros(2),
        np.asarray((10.0, 0.5)),
        parameter_names=("q", "z"),
        rank_rtol=1.0e-8,
    )
    report = assess_matched_model_recovery(
        subspace,
        fitted_parameters=np.asarray((0.1, 0.0)),
        truth_parameters=np.zeros(2),
    )

    assert report.maximum_active_combination_error_scaled == pytest.approx(0.01)
    assert not report.active_one_percent_gate_passed
    assert not report.one_percent_gate_passed
    assert report.parameters[0].scaled_error == pytest.approx(0.01)
    assert report.parameters[0].one_percent_gate_passed is False
    assert report.parameters[1].identifiable_individually


def test_physical_optical_policy_uses_named_metrics_and_excludes_nuisances():
    names = (
        "CL1.log_length_m",
        "IL1.q",
        "C-mini.slope",
        "calibration.gain",
    )
    truth = np.asarray((np.log(0.05), 2.0, 0.0, 100.0))
    fitted = truth + np.asarray((0.005, 0.018, 0.005, 50.0))
    subspace = scaled_identifiable_subspace(
        np.eye(4),
        truth,
        np.asarray((1.0e-3, 1.0e-3, 1.0, 1.0)),
        parameter_names=names,
        rank_rtol=1.0e-8,
    )

    report = assess_matched_model_recovery(
        subspace,
        fitted,
        truth,
        named_metric_policy="physical_optical",
        optical_parameter_names=names[:3],
    )

    assert not report.active_one_percent_gate_passed
    assert report.named_one_percent_gate_passed
    assert report.one_percent_gate_passed
    assert report.gated_parameter_names == names[:3]
    assert report.requested_parameter_names == names[:3]
    assert report.parameters[0].named_metric_kind == "absolute_log_length_error"
    assert report.parameters[0].named_metric_error == pytest.approx(0.005)
    assert report.parameters[1].named_metric_kind == "absolute_fractional_truth_error"
    assert report.parameters[1].named_metric_error == pytest.approx(0.009)
    assert (
        report.parameters[2].named_metric_kind
        == "absolute_error_over_declared_zero_reference_scale"
    )
    assert report.parameters[2].named_metric_error == pytest.approx(0.005)
    assert not report.parameters[3].included_in_named_gate
    assert report.parameters[3].scaled_error is None
    assert report.parameters[3].named_metric_error is None
    assert report.parameters[3].one_percent_gate_passed is None


def test_physical_optical_policy_requires_explicit_known_optical_names():
    subspace = scaled_identifiable_subspace(
        np.eye(2),
        np.ones(2),
        np.ones(2),
        parameter_names=("q", "nuisance"),
        rank_rtol=1.0e-8,
    )
    with pytest.raises(ValueError, match="explicit optical_parameter_names"):
        assess_matched_model_recovery(
            subspace,
            np.ones(2),
            np.ones(2),
            named_metric_policy="physical_optical",
        )
    with pytest.raises(ValueError, match="unknown names"):
        assess_matched_model_recovery(
            subspace,
            np.ones(2),
            np.ones(2),
            named_metric_policy="physical_optical",
            optical_parameter_names=("missing",),
        )


def test_physical_optical_axial_metric_is_invariant_to_z_origin():
    names = ("CL1.z_m", "source_z_m", "detector_z_m")
    scales = np.asarray((5.0e-4, 1.0e-3, 1.0e-3))
    truth = np.asarray((0.2, 0.0, 1.04))
    displacement = 0.005 * scales
    subspace = scaled_identifiable_subspace(
        np.eye(3),
        truth,
        scales,
        parameter_names=names,
        rank_rtol=1.0e-8,
    )

    first = assess_matched_model_recovery(
        subspace,
        truth + displacement,
        truth,
        named_metric_policy="physical_optical",
        optical_parameter_names=names,
    )
    shifted_truth = truth + 50.0
    shifted_subspace = scaled_identifiable_subspace(
        np.eye(3),
        shifted_truth,
        scales,
        parameter_names=names,
        rank_rtol=1.0e-8,
    )
    shifted = assess_matched_model_recovery(
        shifted_subspace,
        shifted_truth + displacement,
        shifted_truth,
        named_metric_policy="physical_optical",
        optical_parameter_names=names,
    )

    np.testing.assert_allclose(
        [item.named_metric_error for item in first.parameters],
        [item.named_metric_error for item in shifted.parameters],
    )
    assert all(
        item.named_metric_kind == "absolute_axial_error_over_declared_scale"
        for item in first.parameters
    )


@pytest.mark.parametrize(
    ("fitted", "truth", "null_weight"),
    [
        (np.zeros(3), np.zeros(2), 1.0e-8),
        (np.zeros(2), np.asarray((0.0, np.nan)), 1.0e-8),
        (np.zeros(2), np.zeros(2), -1.0),
        (np.zeros(2), np.zeros(2), 1.0),
    ],
)
def test_matched_recovery_validates_inputs(fitted, truth, null_weight):
    subspace = scaled_identifiable_subspace(
        np.eye(2), np.zeros(2), np.ones(2), rank_rtol=1.0e-8
    )
    with pytest.raises(ValueError):
        assess_matched_model_recovery(
            subspace,
            fitted_parameters=fitted,
            truth_parameters=truth,
            negligible_null_weight=null_weight,
        )


def test_null_family_audit_enumerates_all_exact_bounded_equivalences():
    parameters = np.zeros(3)
    family = equivalent_solution_family(
        np.asarray(((1.0, 0.0, 0.0),)),
        parameters,
        parameter_names=("measured", "gauge_y", "gauge_z"),
        parameter_scales=np.asarray((1.0, 2.0, 4.0)),
        rtol=1.0e-8,
    )

    report = verify_local_null_families(
        family,
        lambda value: jnp.asarray((value[0],)),
        bounds=(np.full(3, -1.0), np.full(3, 1.0)),
        parameter_scales=np.asarray((1.0, 2.0, 4.0)),
        delta_cost_tolerance=1.0e-12,
        dominant_loading_count=2,
    )

    assert len(report.families) == family.nullity == 2
    assert report.all_verified_equivalent
    assert {item.dominant_loadings[0].parameter_name for item in report.families} == {
        "gauge_y",
        "gauge_z",
    }
    for item in report.families:
        assert item.status == "verified_straight_equivalent"
        assert item.verified_straight_equivalent
        assert item.negative_endpoint.delta_cost == pytest.approx(0.0)
        assert item.positive_endpoint.delta_cost == pytest.approx(0.0)
        assert item.negative_endpoint.signed_scaled_coordinate < 0.0
        assert item.positive_endpoint.signed_scaled_coordinate > 0.0
        assert item.negative_endpoint.active_bounds
        assert item.positive_endpoint.active_bounds
        assert item.negative_physical_bound_reached
        assert item.positive_physical_bound_reached
        assert item.requested_negative_scaled_coordinate == pytest.approx(
            item.negative_endpoint.signed_scaled_coordinate
        )
        assert item.requested_positive_scaled_coordinate == pytest.approx(
            item.positive_endpoint.signed_scaled_coordinate
        )
        np.testing.assert_allclose(np.linalg.norm(item.direction_scaled), 1.0)
        assert item.sampled_scaled_coordinates.shape == (9,)
        assert 0.0 in item.sampled_scaled_coordinates
        np.testing.assert_allclose(item.sampled_delta_costs, 0.0)
        assert item.dominant_loadings[0].scaled_loading > 0.0
        with pytest.raises(ValueError):
            item.direction_physical[0] = 2.0


def test_null_family_audit_keeps_nonlinear_null_tangent_as_candidate():
    family = equivalent_solution_family(
        np.asarray(((1.0, 0.0), (0.0, 0.0))),
        np.zeros(2),
        parameter_names=("measured", "quadratic_gauge"),
        parameter_scales=np.ones(2),
        rtol=1.0e-8,
    )

    report = verify_local_null_families(
        family,
        lambda value: jnp.asarray((value[0], value[1] ** 2)),
        bounds=(np.full(2, -1.0), np.full(2, 1.0)),
        parameter_scales=np.ones(2),
        delta_cost_tolerance=0.1,
    )

    assert len(report.families) == 1
    item = report.families[0]
    assert item.status == "local_null_candidate"
    assert not item.verified_equivalent
    assert item.negative_endpoint.delta_cost == pytest.approx(0.5)
    assert item.positive_endpoint.delta_cost == pytest.approx(0.5)
    assert item.maximum_absolute_endpoint_delta_cost == pytest.approx(0.5)
    assert not report.all_verified_equivalent


def test_null_family_audit_rejects_endpoint_flat_but_curved_interior_segment():
    family = equivalent_solution_family(
        np.asarray(((1.0, 0.0),)),
        np.zeros(2),
        parameter_names=("measured", "curved_tangent"),
        parameter_scales=np.ones(2),
        rtol=1.0e-8,
    )
    report = verify_local_null_families(
        family,
        lambda value: jnp.asarray(
            (value[0], value[1] ** 2 * (1.0 - value[1] ** 2))
        ),
        bounds=(np.full(2, -1.0), np.full(2, 1.0)),
        parameter_scales=np.ones(2),
        delta_cost_tolerance=1.0e-3,
        straight_sample_count=9,
    )

    item = report.families[0]
    assert item.negative_endpoint.delta_cost == pytest.approx(0.0)
    assert item.positive_endpoint.delta_cost == pytest.approx(0.0)
    assert item.maximum_absolute_endpoint_delta_cost == pytest.approx(0.0)
    assert item.maximum_absolute_sampled_delta_cost > 1.0e-3
    assert item.status == "local_null_candidate"
    assert not item.verified_straight_equivalent


def test_null_family_audit_requires_finite_boxes_and_stable_residual_shape():
    family = equivalent_solution_family(
        np.asarray(((1.0, 0.0),)),
        np.zeros(2),
        parameter_scales=np.ones(2),
        rtol=1.0e-8,
    )
    with pytest.raises(ValueError, match="finite bounds"):
        verify_local_null_families(
            family,
            lambda value: jnp.asarray((value[0],)),
            bounds=(-np.inf, np.inf),
            parameter_scales=np.ones(2),
            delta_cost_tolerance=1.0,
        )

    def changing_shape(value):
        return np.zeros(1 if float(value[1]) == 0.0 else 2)

    with pytest.raises(ValueError, match="changed residual-vector size"):
        verify_local_null_families(
            family,
            changing_shape,
            bounds=(np.full(2, -1.0), np.full(2, 1.0)),
            parameter_scales=np.ones(2),
            delta_cost_tolerance=1.0,
        )


def test_null_family_audit_truncates_nonfinite_model_domain_without_aborting():
    family = equivalent_solution_family(
        np.asarray(((1.0, 0.0, 0.0),)),
        np.zeros(3),
        parameter_names=("measured", "singular_y", "free_z"),
        parameter_scales=np.ones(3),
        rtol=1.0e-8,
    )

    def residual(value):
        singular = jnp.where(jnp.abs(value[1]) < 0.4, value[0], jnp.nan)
        return jnp.asarray((singular,))

    report = verify_local_null_families(
        family,
        residual,
        bounds=(np.full(3, -1.0), np.full(3, 1.0)),
        parameter_scales=np.ones(3),
        delta_cost_tolerance=1.0e-12,
        straight_sample_count=9,
    )

    assert len(report.families) == 2
    truncated = next(
        item
        for item in report.families
        if item.dominant_loadings[0].parameter_name == "singular_y"
    )
    unaffected = next(
        item
        for item in report.families
        if item.dominant_loadings[0].parameter_name == "free_z"
    )
    assert truncated.status == "finite_domain_truncated"
    assert not truncated.verified_equivalent
    assert not truncated.negative_physical_bound_reached
    assert not truncated.positive_physical_bound_reached
    assert abs(truncated.negative_endpoint.signed_scaled_coordinate) < 0.4
    assert abs(truncated.positive_endpoint.signed_scaled_coordinate) < 0.4
    assert np.all(np.isfinite(truncated.sampled_costs))
    assert unaffected.status == "verified_straight_equivalent"
    assert unaffected.negative_physical_bound_reached
    assert unaffected.positive_physical_bound_reached

@pytest.mark.parametrize(
    "kwargs",
    [
        {"parameter_scales": np.asarray((1.0,))},
        {"parameter_scales": np.asarray((1.0, 0.0))},
        {"delta_cost_tolerance": -1.0},
        {"dominant_loading_count": 0},
        {"straight_sample_count": 4},
    ],
)
def test_null_family_audit_validates_controls(kwargs):
    family = equivalent_solution_family(
        np.asarray(((1.0, 0.0),)),
        np.zeros(2),
        parameter_scales=np.ones(2),
        rtol=1.0e-8,
    )
    options = {
        "bounds": (np.full(2, -1.0), np.full(2, 1.0)),
        "parameter_scales": np.ones(2),
        "delta_cost_tolerance": 1.0,
    }
    options.update(kwargs)
    with pytest.raises(ValueError):
        verify_local_null_families(
            family,
            lambda value: jnp.asarray((value[0],)),
            **options,
        )
