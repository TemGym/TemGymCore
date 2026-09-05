import json

import numpy as np
import pytest

from temgym_core.gaussian import make_gaussian
from temgym_core.neoarm_calibration import (
    DetectorGeometry,
    apply_transfer_matrix_to_gaussians,
    render_gaussian_detector_plane,
)
from temgym_core.topology_validation import (
    AnalyticGaussianDetectorReference,
    FisherBiasBlockInfluence,
    FisherParameterStabilityReport,
    GaussianPerforationCertificate,
    PhysicalRenderSignature,
    PhysicalValidationEvidence,
    PhysicalValidationLedger,
    PhysicalValidationRequirements,
    PhysicalValidationThresholds,
    analytic_single_gaussian_detector_reference,
    assess_fisher_parameter_stability,
    assess_nuisance_profiled_optical_stability,
    certify_single_gaussian_perforation,
    evaluate_physical_validation,
    stable_validation_signature,
)


def _drift_matrix(length_m):
    return np.asarray(
        (
            (1.0, length_m, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, length_m),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def _signature(**updates):
    values = {
        "acquisition_kind": "imaging",
        "frozen_topology_id": "silicon:L0:reexpand-v1:0123456789abcdef",
        "interaction_route": "sample_plane_reexpansion",
        "source_signature": "sha256:source",
        "downstream_signature": "sha256:downstream",
        "detector_signature": "sha256:detector",
        "render_signature": "gaussian-gpu-v5",
        "renderer_family": "adaptive_gaussian_beamlets",
        "optical_scope": "neoarm_physical_column",
    }
    values.update(updates)
    return PhysicalRenderSignature(**values)


def _evidence(signature, **updates):
    values = {
        "signature": signature,
        "reference_method": "scaled CZT grid-wave propagation",
        "reference_signature": "grid-czt-v2:sha256:reference",
        "reference_renderer_family": "scaled_czt_grid_wave",
        "reference_optical_scope": signature.optical_scope,
        "sample_field_rms_in_noise_sigma": 0.03,
        "detector_rms_in_noise_sigma": 0.04,
        "observable_residuals_in_sigma": {
            "scale": 0.05,
            "rotation": 0.05,
            "centroid": 0.05,
            "fourier_vector": 0.05,
            "intensity": 0.05,
        },
        "parameter_change_in_sigma": 0.02,
        "provenance": {"candidate_checkpoint": "abc", "reference_grid": [512, 512]},
    }
    values.update(updates)
    return PhysicalValidationEvidence(**values)


def test_render_identity_is_not_topology_id_alone():
    baseline = _signature()
    variants = (
        _signature(source_signature="sha256:shifted-source"),
        _signature(downstream_signature="sha256:wobbled-downstream"),
        _signature(detector_signature="sha256:different-mtf"),
        _signature(render_signature="gaussian-gpu-v6"),
        _signature(renderer_family="direct_grid_wave"),
        _signature(optical_scope="declared_ideal_fourier_relay"),
    )
    assert len({baseline, *variants}) == len(variants) + 1
    assert len({item.digest for item in (baseline, *variants)}) == len(variants) + 1


def test_numeric_signature_is_order_and_backend_dtype_stable():
    arrays_a = {
        "matrix": np.asarray(((1.0, 2.0), (3.0, 4.0)), dtype=np.float32),
        "indices": np.asarray((1, 2), dtype=np.int16),
    }
    arrays_b = {
        "indices": np.asarray((1, 2), dtype=np.int64),
        "matrix": np.asarray(((1.0, 2.0), (3.0, 4.0)), dtype=np.float64),
    }
    first = stable_validation_signature(
        "downstream-v1", arrays=arrays_a, metadata={"z_m": 1.0, "mode": "image"}
    )
    second = stable_validation_signature(
        "downstream-v1", arrays=arrays_b, metadata={"mode": "image", "z_m": 1.0}
    )
    assert first == second
    changed = stable_validation_signature(
        "downstream-v1",
        arrays={**arrays_b, "matrix": arrays_b["matrix"] + np.eye(2)},
        metadata={"mode": "image", "z_m": 1.0},
    )
    assert changed != first
    with pytest.raises(ValueError, match="non-finite"):
        stable_validation_signature("invalid", arrays={"bad": np.asarray([np.nan])})


def test_complete_evidence_passes_at_inclusive_plan_thresholds():
    signature = _signature()
    evidence = _evidence(
        signature,
        sample_field_rms_in_noise_sigma=0.2,
        detector_rms_in_noise_sigma=0.2,
        observable_residuals_in_sigma={
            "scale": 0.2,
            "rotation": 0.2,
            "centroid": 0.2,
            "fourier_vector": 0.2,
            "intensity": 0.2,
        },
        parameter_change_in_sigma=0.1,
    )
    decision = evaluate_physical_validation(evidence)
    assert decision.passed
    assert decision.failure_reasons == ()


def test_missing_or_dependent_reference_evidence_fails_closed():
    signature = _signature()
    evidence = _evidence(
        signature,
        reference_signature=signature.render_signature,
        reference_renderer_family=signature.renderer_family,
        sample_field_rms_in_noise_sigma=None,
        observable_residuals_in_sigma={"scale": 0.01},
        parameter_change_in_sigma=None,
    )
    decision = evaluate_physical_validation(evidence)
    assert not decision.passed
    assert "reference_independence" in decision.failure_reasons
    assert "missing:sample_field" in decision.failure_reasons
    assert "missing:parameter_stability" in decision.failure_reasons
    assert "missing:observable:rotation" in decision.failure_reasons


def test_ideal_relay_cannot_satisfy_native_column_scope_requirement():
    signature = _signature(
        acquisition_kind="diffraction",
        optical_scope="declared_ideal_fourier_relay",
    )
    requirements = PhysicalValidationRequirements(
        required_observables=("diffraction_vector", "intensity"),
        allowed_optical_scopes=("neoarm_physical_column",),
    )
    evidence = _evidence(
        signature,
        reference_optical_scope="declared_ideal_fourier_relay",
        observable_residuals_in_sigma={
            "diffraction_vector": 0.05,
            "intensity": 0.05,
        },
    )
    decision = evaluate_physical_validation(evidence, requirements)
    assert not decision.passed
    assert decision.failure_reasons == ("candidate_optical_scope",)


def test_reference_scope_must_match_candidate_physics():
    signature = _signature()
    decision = evaluate_physical_validation(
        _evidence(signature, reference_optical_scope="thin_lens_surrogate")
    )
    assert not decision.passed
    assert "reference_optical_scope" in decision.failure_reasons


def test_ledger_requires_every_exact_signature_and_retains_failed_audit(tmp_path):
    first = _signature()
    second = _signature(source_signature="sha256:tilted-source")
    requirements = PhysicalValidationRequirements(
        allowed_optical_scopes=("neoarm_physical_column",)
    )
    ledger = PhysicalValidationLedger((first, second), default_requirements=requirements)

    failed = ledger.record(_evidence(first, detector_rms_in_noise_sigma=0.21))
    assert not failed.passed
    assert ledger.coverage.failed_count == 1
    assert ledger.coverage.missing_count == 1
    assert not ledger.complete

    passed = ledger.record(_evidence(first))
    assert passed.passed
    assert len(ledger.decisions_for(first)) == 2
    assert ledger.coverage.accepted_count == 1
    assert ledger.coverage.failed_count == 0
    assert ledger.coverage.missing_count == 1

    ledger.record(_evidence(second))
    assert ledger.complete
    assert ledger.coverage.required_count == 2
    assert ledger.coverage.accepted_count == 2

    destination = ledger.write_json(tmp_path / "validation" / "ledger.json")
    payload = json.loads(destination.read_text())
    assert payload["schema"] == "temgym_core.physical_topology_validation_ledger"
    assert payload["coverage"]["complete"] is True
    assert len(payload["entries"][0]["decisions"]) in {1, 2}
    assert len(payload["sha256"]) == 64


def test_ledger_rejects_unplanned_signature():
    required = _signature()
    ledger = PhysicalValidationLedger((required,))
    unplanned = _signature(downstream_signature="sha256:unplanned")
    with pytest.raises(KeyError, match="predeclared"):
        ledger.record(_evidence(unplanned))


def test_custom_thresholds_and_diffraction_observable_contract():
    signature = _signature(acquisition_kind="diffraction")
    requirements = PhysicalValidationRequirements(
        required_observables=("diffraction_vector", "camera_length", "intensity"),
        thresholds=PhysicalValidationThresholds(
            sample_field_rms_in_noise_sigma=0.1,
            detector_rms_in_noise_sigma=0.1,
            observable_residual_in_sigma=0.1,
            parameter_change_in_sigma=0.05,
        ),
    )
    evidence = _evidence(
        signature,
        observable_residuals_in_sigma={
            "diffraction_vector": 0.08,
            "camera_length": 0.11,
            "intensity": 0.02,
        },
    )
    decision = evaluate_physical_validation(evidence, requirements)
    assert not decision.passed
    assert decision.failure_reasons == ("observable:camera_length",)


def test_analytic_single_gaussian_reference_matches_production_field_evaluation():
    beam = make_gaussian(
        x=1.3e-6,
        y=-0.7e-6,
        dx=2.0e-5,
        dy=-1.0e-5,
        z=0.0,
        voltage=200e3,
        amp=np.sqrt(2.0 / np.pi) / 8.0e-6,
        waist_x=8.0e-6,
        waist_y=11.0e-6,
        rcurv_x=0.2,
        rcurv_y=-0.3,
    )
    distance = 0.37
    matrix = _drift_matrix(distance)
    detector = DetectorGeometry(
        shape_yx=(67, 65),
        pixel_size_yx_m=(1.1e-6, 1.3e-6),
        z_m=distance,
        mtf_sigma_px=0.65,
        affine_xy=np.asarray(((1.01, 0.03), (-0.02, 0.98))),
    )
    propagated = apply_transfer_matrix_to_gaussians(
        beam, matrix, z_out_m=distance
    )
    production = render_gaussian_detector_plane(
        propagated,
        detector,
        tile_rows=detector.shape_yx[0],
        beam_batch_size=1,
        method="loop",
    )
    reference = analytic_single_gaussian_detector_reference(
        beam, matrix, detector
    )

    assert isinstance(reference, AnalyticGaussianDetectorReference)
    np.testing.assert_allclose(
        reference.intensity_yx,
        production.intensity_yx,
        rtol=3e-13,
        atol=2e-5,
    )
    expected_centre = (
        matrix[np.ix_((0, 2), (0, 2))] @ np.asarray((1.3e-6, -0.7e-6))
        + matrix[np.ix_((0, 2), (1, 3))] @ np.asarray((2.0e-5, -1.0e-5))
    )
    np.testing.assert_allclose(
        reference.detector_centre_column_xy_m, expected_centre, rtol=0, atol=1e-18
    )
    assert 0.99 < reference.sampled_power_fraction < 1.0
    assert reference.detector_intensity_covariance_pixel2.shape == (2, 2)
    assert not reference.intensity_yx.flags.writeable


def test_analytic_reference_retains_calibrated_position_and_tilt_separately():
    detector = DetectorGeometry(
        shape_yx=(41, 43),
        pixel_size_yx_m=(2e-6, 3e-6),
        z_m=0.2,
        affine_xy=np.asarray(((1.0, 0.1), (-0.05, 0.95))),
    )
    matrix = _drift_matrix(0.2)
    base = make_gaussian(
        x=2e-6,
        y=-1e-6,
        dx=0.0,
        dy=0.0,
        z=0.0,
        voltage=200e3,
        waist_x=7e-6,
        waist_y=7e-6,
    )
    tilted = base.derive(dx=3e-5, dy=-4e-5)
    base_reference = analytic_single_gaussian_detector_reference(
        base, matrix, detector
    )
    tilted_reference = analytic_single_gaussian_detector_reference(
        tilted, matrix, detector
    )
    expected_shift = matrix[np.ix_((0, 2), (1, 3))] @ np.asarray((3e-5, -4e-5))
    np.testing.assert_allclose(
        tilted_reference.detector_centre_column_xy_m
        - base_reference.detector_centre_column_xy_m,
        expected_shift,
        rtol=0,
        atol=1e-18,
    )
    # A pure calibrated tilt translates the intensity but does not alter its
    # covariance under this first-order drift.
    np.testing.assert_allclose(
        tilted_reference.detector_intensity_covariance_column_m2,
        base_reference.detector_intensity_covariance_column_m2,
        rtol=2e-14,
        atol=1e-24,
    )


def test_finite_perforation_certificate_matches_isotropic_gaussian_tail():
    waist = 4e-6
    radius = 12e-6
    beam = make_gaussian(
        x=0.0,
        y=0.0,
        z=0.0,
        voltage=200e3,
        waist_x=waist,
        waist_y=waist,
    )
    supplied_centre = np.zeros(2)
    certificate = certify_single_gaussian_perforation(
        beam,
        perforation_centre_xy_m=supplied_centre,
        perforation_radius_m=radius,
        maximum_lost_power_fraction=1e-7,
    )
    exact_isotropic_tail = np.exp(-2.0 * radius**2 / waist**2)

    assert isinstance(certificate, GaussianPerforationCertificate)
    assert certificate.passed
    assert certificate.radial_clearance_sigma == pytest.approx(2.0 * radius / waist)
    assert certificate.lost_power_fraction_upper_bound == pytest.approx(
        exact_isotropic_tail
    )
    assert supplied_centre.flags.writeable
    assert not certificate.beam_centre_xy_m.flags.writeable


def test_perforation_certificate_fails_closed_for_edge_crossing_or_uncertainty():
    beam = make_gaussian(
        x=9e-6,
        y=0.0,
        z=0.0,
        voltage=200e3,
        waist_x=4e-6,
        waist_y=3e-6,
    )
    failed = certify_single_gaussian_perforation(
        beam,
        perforation_centre_xy_m=(0.0, 0.0),
        perforation_radius_m=10e-6,
        uncertain_edge_width_m=2e-6,
        maximum_lost_power_fraction=0.1,
    )
    assert not failed.passed
    assert failed.lost_power_fraction_upper_bound == 1.0
    with pytest.raises(ValueError, match="smaller than the radius"):
        certify_single_gaussian_perforation(
            beam,
            perforation_centre_xy_m=(0.0, 0.0),
            perforation_radius_m=10e-6,
            uncertain_edge_width_m=10e-6,
        )


def test_analytic_reference_rejects_a_multi_gaussian_bundle():
    beam = make_gaussian(
        x=np.asarray((0.0, 1e-6)),
        y=np.zeros(2),
        z=np.zeros(2),
        voltage=np.full(2, 200e3),
        waist_x=np.full(2, 5e-6),
        waist_y=np.full(2, 5e-6),
    )
    detector = DetectorGeometry(shape_yx=(9, 9), pixel_size_yx_m=(2e-6, 2e-6))
    with pytest.raises(ValueError, match="exactly one Gaussian"):
        analytic_single_gaussian_detector_reference(
            beam, np.eye(4), detector
        )


def test_aggregate_fisher_parameter_stability_is_strict_and_fail_closed():
    jacobian = np.diag((2.0, 4.0))
    # This discrepancy is exactly cancelled by an active shift (-0.02, -0.01).
    discrepancy = jacobian @ np.asarray((0.02, 0.01))
    report = assess_fisher_parameter_stability(
        jacobian,
        discrepancy,
        np.ones(2),
        np.eye(2),
        successive_refinement_changes_in_sigma=(0.08, 0.06),
    )
    assert isinstance(report, FisherParameterStabilityReport)
    assert report.active_design_full_rank
    assert np.allclose(report.active_parameter_shift, (-0.02, -0.01))
    assert np.allclose(report.active_standard_errors, (0.5, 0.25))
    assert np.allclose(report.active_coordinate_change_in_sigma, (0.04, 0.04))
    assert np.isclose(
        report.fisher_mahalanobis_change_in_sigma,
        np.sqrt(0.04**2 + 0.04**2),
    )
    assert report.fisher_mahalanobis_change_in_sigma \
        >= report.maximum_active_coordinate_change_in_sigma
    assert report.aggregate_projection_passed
    assert report.successive_refinements_passed
    assert report.passed
    assert not report.active_parameter_shift.flags.writeable
    assert report.to_payload()["passed"] is True

    at_boundary = assess_fisher_parameter_stability(
        np.eye(2),
        np.asarray((0.1, 0.0)),
        np.ones(2),
        np.eye(2),
        successive_refinement_changes_in_sigma=(0.01, 0.02),
    )
    assert not at_boundary.aggregate_projection_passed
    assert not at_boundary.passed


def test_aggregate_fisher_parameter_stability_requires_two_real_transitions():
    missing_history = assess_fisher_parameter_stability(
        np.eye(2),
        np.zeros(2),
        np.ones(2),
        np.eye(2),
        successive_refinement_changes_in_sigma=(0.01,),
    )
    assert missing_history.aggregate_projection_passed
    assert not missing_history.successive_refinements_passed
    assert not missing_history.passed

    strict_history = assess_fisher_parameter_stability(
        np.eye(2),
        np.zeros(2),
        np.ones(2),
        np.eye(2),
        successive_refinement_changes_in_sigma=(0.01, 0.1),
    )
    assert not strict_history.successive_refinements_passed
    with pytest.raises(ValueError, match="non-negative"):
        assess_fisher_parameter_stability(
            np.eye(2),
            np.zeros(2),
            np.ones(2),
            np.eye(2),
            successive_refinement_changes_in_sigma=(np.nan, 0.01),
        )


def test_aggregate_fisher_parameter_stability_uses_basis_invariant_gate():
    jacobian = np.asarray(((3.0, 0.4), (0.2, 2.0), (1.0, -0.5)))
    discrepancy = np.asarray((0.04, -0.02, 0.01))
    angle = 0.73
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    first = assess_fisher_parameter_stability(
        jacobian,
        discrepancy,
        np.ones(2),
        np.eye(2),
        successive_refinement_changes_in_sigma=(0.01, 0.01),
    )
    rotated = assess_fisher_parameter_stability(
        jacobian,
        discrepancy,
        np.ones(2),
        rotation,
        successive_refinement_changes_in_sigma=(0.01, 0.01),
    )
    assert np.isclose(
        first.fisher_mahalanobis_change_in_sigma,
        rotated.fisher_mahalanobis_change_in_sigma,
        rtol=1e-12,
        atol=1e-14,
    )
    assert first.aggregate_projection_passed == rotated.aggregate_projection_passed


def test_aggregate_fisher_parameter_stability_rejects_rank_deficient_active_design():
    report = assess_fisher_parameter_stability(
        np.asarray(((1.0, 1.0), (2.0, 2.0))),
        np.asarray((0.0, 0.0)),
        np.ones(2),
        np.eye(2),
        successive_refinement_changes_in_sigma=(0.01, 0.01),
    )
    assert report.active_design_rank == 1
    assert not report.active_design_full_rank
    assert not report.aggregate_projection_passed
    assert not report.passed


def test_nuisance_profiled_optical_stability_excludes_nuisance_motion_and_decomposes_blocks():
    jacobian = np.asarray(
        (
            (1.0, 0.0, 1.0),
            (0.0, 2.0, 0.0),
            (1.0, 0.0, -1.0),
        )
    )
    optical_block = np.asarray((0.02, 0.03, 0.02))
    nuisance_block = np.asarray((0.2, 0.0, -0.2))
    joint = assess_fisher_parameter_stability(
        jacobian,
        optical_block + nuisance_block,
        np.ones(3),
        np.eye(3),
        successive_refinement_changes_in_sigma=(0.02, 0.03),
    )
    report = assess_nuisance_profiled_optical_stability(
        jacobian,
        optical_block + nuisance_block,
        np.ones(3),
        optical_parameter_count=2,
        optical_active_basis_scaled=np.eye(2),
        nuisance_rank=1,
        optical_parameter_names=("q_1", "q_2"),
        bias_blocks=(
            ("optical", optical_block),
            ("nuisance", nuisance_block),
        ),
        successive_refinement_changes_in_sigma=(0.02, 0.03),
    )
    assert not joint.aggregate_projection_passed
    assert report.passed
    assert report.stability.active_rank == 2
    assert report.nuisance_rank == 1
    assert np.isclose(
        report.stability.fisher_mahalanobis_change_in_sigma,
        np.hypot(np.sqrt(2.0) * 0.02, 0.03),
    )
    assert np.isclose(
        np.linalg.norm(report.signed_optical_fisher_mode_changes_in_sigma),
        report.stability.fisher_mahalanobis_change_in_sigma,
    )
    assert np.allclose(
        report.optical_fisher_mode_directions_scaled.T
        @ report.optical_fisher_mode_directions_scaled,
        np.eye(2),
    )
    by_name = {value.name: value for value in report.block_influences}
    assert isinstance(by_name["optical"], FisherBiasBlockInfluence)
    assert np.isclose(
        by_name["optical"].leave_block_fisher_change_in_sigma, 0.0
    )
    assert np.isclose(
        by_name["nuisance"].projected_fisher_change_in_sigma, 0.0, atol=1e-14
    )
    assert np.isclose(
        sum(value.signed_fisher_squared_contribution for value in by_name.values()),
        report.stability.fisher_mahalanobis_change_in_sigma**2,
    )
    assert report.to_payload()["scope"] \
        == "nuisance_profiled_identifiable_optical_quotient"


def test_nuisance_profiled_optical_stability_requires_complete_bias_blocks():
    with pytest.raises(ValueError, match="sum to the complete"):
        assess_nuisance_profiled_optical_stability(
            np.eye(3),
            np.asarray((0.01, 0.02, 0.03)),
            np.ones(3),
            optical_parameter_count=2,
            optical_active_basis_scaled=np.eye(2),
            nuisance_rank=1,
            bias_blocks=(("incomplete", np.asarray((0.01, 0.02, 0.0))),),
            successive_refinement_changes_in_sigma=(0.01, 0.01),
        )
