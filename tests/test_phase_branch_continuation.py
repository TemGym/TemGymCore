from dataclasses import replace

import numpy as np
import pytest

from temgym_core.microscope_inversion import hard_edge_solenoid_matrix
from temgym_core.phase_branch_continuation import (
    CONTINUATION_MANIFEST_SCHEMA,
    HARD_EDGE_SIGN_EXPLANATION,
    MAX_CONTINUATION_PHASE_STEP_RAD,
    PhaseContinuationError,
    audit_continuation_manifest,
    build_experimental_continuation_manifest,
    continuation_manifest_from_payload,
    continuation_manifest_to_payload,
    design_raw_dac_continuation_grid,
    enumerate_hard_edge_phase_aliases,
    fit_relative_q_slope,
    transform_q_reference_covariance,
    unwrap_oriented_ellipse_angles,
    unwrap_signed_response_vector_rotations,
)


def _diagonal_angle_covariance(size: int, sigma_rad: float = 0.005) -> np.ndarray:
    return np.eye(size, dtype=float) * sigma_rad**2


def test_raw_dac_grid_is_connected_and_preserves_exact_anchors():
    grid = design_raw_dac_continuation_grid(
        -3.0,
        7.0,
        max_abs_dq_per_dac=0.4,
        anchor_raw_dac_values=(-1.25, 2.0, 6.125),
        target_max_abs_delta_q_rad=0.31,
    )

    assert grid.start_raw_dac == -3.0
    assert grid.stop_raw_dac == 7.0
    assert all(
        grid.raw_dac_values[index] == anchor
        for index, anchor in zip(
            grid.anchor_indices, grid.anchor_raw_dac_values, strict=True
        )
    )
    assert np.all(np.diff(grid.raw_dac_values) > 0.0)
    assert grid.maximum_predicted_abs_delta_q_rad <= 0.31 + 1e-15
    with pytest.raises(ValueError, match="read-only"):
        grid.raw_dac_values[0] = 100.0


def test_raw_dac_grid_supports_descending_scan_and_rejects_bad_contracts():
    grid = design_raw_dac_continuation_grid(
        4.0,
        -2.0,
        max_abs_dq_per_dac=0.0,
        anchor_raw_dac_values=(3.0, 0.0),
    )
    np.testing.assert_array_equal(grid.raw_dac_values, (4.0, 3.0, 0.0, -2.0))
    assert grid.maximum_predicted_abs_delta_q_rad == 0.0

    with pytest.raises(ValueError, match="pi/4"):
        design_raw_dac_continuation_grid(
            0.0,
            1.0,
            max_abs_dq_per_dac=1.0,
            target_max_abs_delta_q_rad=np.pi / 3.0,
        )
    with pytest.raises(ValueError, match="between the endpoints"):
        design_raw_dac_continuation_grid(
            0.0,
            1.0,
            max_abs_dq_per_dac=1.0,
            anchor_raw_dac_values=(2.0,),
        )


def test_oriented_ellipse_uses_doubled_angle_and_preserves_intercept():
    grid = design_raw_dac_continuation_grid(
        0.0,
        4.0,
        max_abs_dq_per_dac=0.2,
        target_max_abs_delta_q_rad=0.2,
    )
    true_unwrapped = 1.42 + 0.2 * grid.raw_dac_values
    # This principal ellipse-axis representation jumps by pi at pi/2.
    measured = 0.5 * np.angle(np.exp(2.0j * true_unwrapped))
    result = unwrap_oriented_ellipse_angles(
        grid,
        measured,
        _diagonal_angle_covariance(measured.size),
    )

    assert result.passed
    assert result.periodicity_rad == np.pi
    np.testing.assert_allclose(
        np.diff(result.unwrapped_angles_rad),
        np.diff(true_unwrapped),
        rtol=0.0,
        atol=2e-15,
    )
    # The common tag orientation remains present; it was not subtracted.
    assert result.unwrapped_angles_rad[0] == pytest.approx(true_unwrapped[0])


def test_oriented_ellipse_uncertainty_gate_fails_closed_with_diagnostics():
    grid = design_raw_dac_continuation_grid(
        0.0,
        2.0,
        max_abs_dq_per_dac=0.1,
        target_max_abs_delta_q_rad=0.1,
    )
    angles = np.asarray((0.0, 0.70, 0.72))
    covariance = _diagonal_angle_covariance(3, sigma_rad=0.02)

    with pytest.raises(PhaseContinuationError) as error:
        unwrap_oriented_ellipse_angles(grid, angles, covariance)
    report = error.value.diagnostics
    assert not report.passed
    assert report.failed_interval_indices == (0,)
    assert report.uncertainty_gate_values_rad[0] > np.pi / 4.0
    assert "No branch-unwrapped values are admitted" in str(error.value)

    diagnostic_only = unwrap_oriented_ellipse_angles(
        grid, angles, covariance, fail_closed=False
    )
    assert not diagnostic_only.passed


def test_signed_response_vectors_unwrap_2pi_and_propagate_cartesian_covariance():
    grid = design_raw_dac_continuation_grid(
        -1.0,
        3.0,
        max_abs_dq_per_dac=0.2,
        target_max_abs_delta_q_rad=0.2,
    )
    unwrapped = 2.9 + 0.2 * (grid.raw_dac_values + 1.0)
    radius = 2.0
    vectors = radius * np.column_stack((np.cos(unwrapped), np.sin(unwrapped)))
    sigma_angle = 0.004
    vector_covariance = np.repeat(
        ((radius * sigma_angle) ** 2 * np.eye(2))[None, :, :],
        vectors.shape[0],
        axis=0,
    )
    result = unwrap_signed_response_vector_rotations(
        grid,
        vectors,
        response_vector_covariance_xy=vector_covariance,
    )

    assert result.passed
    assert result.periodicity_rad == 2.0 * np.pi
    np.testing.assert_allclose(np.diag(result.angle_covariance_rad2), sigma_angle**2)
    np.testing.assert_allclose(
        np.diff(result.unwrapped_angles_rad), np.diff(unwrapped), atol=2e-15
    )


def test_signed_response_vector_rejects_zero_and_ambiguous_covariance_input():
    grid = design_raw_dac_continuation_grid(
        0.0, 1.0, max_abs_dq_per_dac=0.1
    )
    vectors = np.asarray(((1.0, 0.0), (0.0, 0.0)))
    with pytest.raises(ValueError, match="undefined"):
        unwrap_signed_response_vector_rotations(
            grid,
            vectors,
            angle_covariance_rad2=np.eye(2) * 1e-4,
        )
    vectors[1] = (1.0, 0.1)
    with pytest.raises(ValueError, match="exactly one"):
        unwrap_signed_response_vector_rotations(grid, vectors)
    with pytest.raises(ValueError, match="exactly one"):
        unwrap_signed_response_vector_rotations(
            grid,
            vectors,
            response_vector_covariance_xy=np.eye(4),
            angle_covariance_rad2=np.eye(2),
        )


def test_signed_response_vector_rejects_cartesian_confidence_region_at_zero():
    grid = design_raw_dac_continuation_grid(
        0.0, 1.0, max_abs_dq_per_dac=0.1
    )
    vectors = np.asarray(((1e-9, 0.0), (1e-9, 0.0)))
    covariance = np.zeros((4, 4), dtype=float)
    covariance[0, 0] = covariance[2, 2] = 1e-12
    covariance[1, 1] = covariance[3, 3] = 1e-30

    with pytest.raises(ValueError, match="confidence region"):
        unwrap_signed_response_vector_rotations(
            grid,
            vectors,
            response_vector_covariance_xy=covariance,
        )


def test_signed_response_vector_preserves_cross_frame_cartesian_covariance():
    grid = design_raw_dac_continuation_grid(
        0.0, 2.0, max_abs_dq_per_dac=0.1
    )
    size = grid.raw_dac_values.size
    radius = 10.0
    vectors = np.column_stack((np.full(size, radius), np.zeros(size)))
    independent_variance = 0.01**2
    common_variance = 0.03**2
    covariance = np.eye(2 * size) * independent_variance
    y_indices = 2 * np.arange(size) + 1
    covariance[np.ix_(y_indices, y_indices)] += common_variance

    result = unwrap_signed_response_vector_rotations(
        grid,
        vectors,
        response_vector_covariance_xy=covariance,
    )

    expected_angle_covariance = np.eye(size) * independent_variance / radius**2
    expected_angle_covariance += (
        np.ones((size, size)) * common_variance / radius**2
    )
    np.testing.assert_allclose(
        result.angle_covariance_rad2, expected_angle_covariance
    )
    np.testing.assert_allclose(
        result.interval_standard_uncertainties_rad,
        np.sqrt(2.0 * independent_variance) / radius,
    )


def test_relative_q_slope_gls_keeps_unknown_orientation_as_intercept():
    grid = design_raw_dac_continuation_grid(
        -10.0,
        10.0,
        max_abs_dq_per_dac=0.04,
        anchor_raw_dac_values=(0.0,),
        target_max_abs_delta_q_rad=0.2,
    )
    q_slope = 0.04
    unknown_orientation = 0.63
    measured = unknown_orientation - q_slope * grid.raw_dac_values
    # A common-mode orientation calibration uncertainty belongs primarily to
    # the retained intercept; interval differences cancel it through the full
    # covariance rather than double-counting it as independent noise.
    covariance = (
        np.eye(measured.size) * 0.003**2
        + np.ones((measured.size, measured.size)) * 0.010**2
    )
    continuation = unwrap_oriented_ellipse_angles(
        grid,
        measured,
        covariance,
    )
    fit = fit_relative_q_slope(continuation, reference_raw_dac=0.0)

    assert fit.q_slope_per_dac == pytest.approx(q_slope, abs=2e-15)
    assert fit.unknown_orientation_intercept_rad == pytest.approx(
        unknown_orientation, abs=2e-15
    )
    assert not fit.absolute_q_reference_identified
    assert fit.chi_square == pytest.approx(0.0, abs=1e-24)
    assert fit.q_slope_standard_uncertainty > 0.0


def test_relative_q_slope_refuses_failed_continuation():
    grid = design_raw_dac_continuation_grid(
        0.0, 1.0, max_abs_dq_per_dac=0.1
    )
    failed = unwrap_oriented_ellipse_angles(
        grid,
        (0.0, 0.75),
        _diagonal_angle_covariance(2, sigma_rad=0.02),
        fail_closed=False,
    )
    with pytest.raises(PhaseContinuationError):
        fit_relative_q_slope(failed)


def test_q_reference_transform_moves_mean_and_full_covariance():
    covariance = np.asarray(((0.04, 0.006), (0.006, 0.0025)))
    result = transform_q_reference_covariance(
        1.2,
        -0.03,
        covariance,
        source_reference_raw_dac=10.0,
        target_reference_raw_dac=14.0,
    )
    expected_jacobian = np.asarray(((1.0, 4.0), (0.0, 1.0)))
    assert result.q_bias == pytest.approx(1.08)
    assert result.q_slope_per_dac == -0.03
    np.testing.assert_array_equal(result.jacobian, expected_jacobian)
    np.testing.assert_allclose(
        result.covariance_q_bias_q_slope,
        expected_jacobian @ covariance @ expected_jacobian.T,
    )


def test_hard_edge_alias_enumeration_is_numerically_exact_and_sign_separate():
    base_q = 0.4
    base_length = 0.02
    enumeration = enumerate_hard_edge_phase_aliases(
        base_q,
        base_length,
        q_bounds=(0.1, 7.0),
        length_bounds_m=(0.001, 0.5),
    )

    assert enumeration.branch_indices == (0, 1, 2)
    reference = np.asarray(hard_edge_solenoid_matrix(base_q, base_length))
    for alias in enumeration.aliases:
        candidate = np.asarray(hard_edge_solenoid_matrix(alias.q, alias.length_m))
        np.testing.assert_allclose(candidate, reference, rtol=2e-6, atol=2e-6)
        assert np.sign(alias.q) == np.sign(base_q)
    assert enumeration.q_sign_is_separate
    assert enumeration.sign_explanation == HARD_EDGE_SIGN_EXPLANATION
    assert "q -> -q" in enumeration.sign_explanation


def test_hard_edge_alias_error_paths():
    with pytest.raises(ValueError, match="non-zero"):
        enumerate_hard_edge_phase_aliases(
            0.0,
            0.02,
            q_bounds=(-1.0, 1.0),
            length_bounds_m=(0.01, 0.1),
        )
    with pytest.raises(ValueError, match="positive increasing"):
        enumerate_hard_edge_phase_aliases(
            0.4,
            0.02,
            q_bounds=(-1.0, 1.0),
            length_bounds_m=(-0.01, 0.1),
        )

    negative = enumerate_hard_edge_phase_aliases(
        -0.4,
        0.02,
        q_bounds=(-7.0, -0.1),
        length_bounds_m=(0.001, 0.5),
    )
    assert negative.branch_indices == (-2, -1, 0)
    assert all(alias.q < 0.0 for alias in negative.aliases)


def test_hard_edge_alias_enumeration_keeps_closed_floating_endpoint():
    base_q = 0.4
    base_length = 0.02
    branch_index = 13
    endpoint_q = base_q + branch_index * np.pi
    endpoint_length = base_length * endpoint_q / base_q

    enumeration = enumerate_hard_edge_phase_aliases(
        base_q,
        base_length,
        q_bounds=(endpoint_q, np.nextafter(endpoint_q, np.inf)),
        length_bounds_m=(
            endpoint_length,
            np.nextafter(endpoint_length, np.inf),
        ),
    )

    assert enumeration.branch_indices == (branch_index,)
    assert enumeration.aliases[0].q == endpoint_q
    assert enumeration.aliases[0].length_m == endpoint_length


def test_upstream_manifest_requires_real_connected_tag_images():
    grid = design_raw_dac_continuation_grid(
        0.0,
        2.0,
        max_abs_dq_per_dac=0.2,
        anchor_raw_dac_values=(1.0,),
        target_max_abs_delta_q_rad=0.2,
    )
    planned = build_experimental_continuation_manifest(
        grid,
        acquisition_id="cl1-forward",
        lens_name="CL1",
        raw_dac_channel_name="CL1",
        route="upstream_astigmatic_tag",
        fixed_raw_dac_channels={"CL3": 1200.0, "C-mini": 800.0},
        tag_generation_z_m=-0.2,
        varied_lens_entrance_z_m=-0.1,
    )
    planned_audit = audit_continuation_manifest(planned)
    assert planned_audit.structural_passed
    assert not planned_audit.measured_continuity_passed
    assert planned_audit.recorded_frame_count == 0
    assert "intermediate_images_not_completely_recorded" in planned_audit.failure_reasons

    measured = build_experimental_continuation_manifest(
        grid,
        acquisition_id="cl1-forward",
        lens_name="CL1",
        raw_dac_channel_name="CL1",
        route="upstream_astigmatic_tag",
        tag_generation_z_m=-0.2,
        varied_lens_entrance_z_m=-0.1,
        recorded_image_ids=tuple(
            f"sha256:image-{index}" for index in range(grid.raw_dac_values.size)
        ),
    )
    audit = audit_continuation_manifest(measured)
    assert audit.measured_continuity_passed
    assert audit.failure_reasons == ()
    assert audit.endpoint_anchor_coverage_passed


def test_upstream_tag_plane_must_really_precede_varied_lens():
    grid = design_raw_dac_continuation_grid(
        0.0, 1.0, max_abs_dq_per_dac=0.1
    )
    with pytest.raises(ValueError, match="upstream"):
        build_experimental_continuation_manifest(
            grid,
            acquisition_id="bad-tag",
            lens_name="CL1",
            raw_dac_channel_name="CL1",
            route="upstream_astigmatic_tag",
            tag_generation_z_m=0.0,
            varied_lens_entrance_z_m=-0.1,
        )


def test_downstream_manifest_has_on_axis_position_and_tilt_at_every_dac():
    grid = design_raw_dac_continuation_grid(
        -2.0,
        2.0,
        max_abs_dq_per_dac=0.15,
        target_max_abs_delta_q_rad=0.2,
    )
    frame_count = 3 * grid.raw_dac_values.size
    manifest = build_experimental_continuation_manifest(
        grid,
        acquisition_id="il1-forward",
        lens_name="IL1",
        raw_dac_channel_name="IL1",
        route="downstream_calibrated_response",
        position_plus_state_x_theta_x_y_theta_y=(2e-9, 0.0, 0.0, 0.0),
        tilt_plus_state_x_theta_x_y_theta_y=(0.0, 1e-3, 0.0, 0.0),
        sample_state_covariance=np.diag((1e-20, 1e-8, 1e-20, 1e-8)),
        recorded_image_ids=tuple(f"image-{index}" for index in range(frame_count)),
    )
    audit = audit_continuation_manifest(manifest)

    assert len(manifest.frames) == frame_count
    assert audit.measured_continuity_passed
    for grid_index in range(grid.raw_dac_values.size):
        roles = {
            frame.role for frame in manifest.frames if frame.grid_index == grid_index
        }
        assert roles == {"on_axis", "position_plus", "tilt_plus"}


def test_manifest_audit_rejects_missing_point_and_legacy_endpoint_metadata():
    grid = design_raw_dac_continuation_grid(
        0.0, 2.0, max_abs_dq_per_dac=0.2, target_max_abs_delta_q_rad=0.2
    )
    count = grid.raw_dac_values.size
    complete = build_experimental_continuation_manifest(
        grid,
        acquisition_id="cl3-forward",
        lens_name="CL3",
        raw_dac_channel_name="CL3",
        route="upstream_astigmatic_tag",
        tag_generation_z_m=-0.3,
        varied_lens_entrance_z_m=-0.2,
        recorded_image_ids=tuple(f"image-{index}" for index in range(count)),
    )
    missing = replace(complete, frames=complete.frames[:-1])
    missing_audit = audit_continuation_manifest(missing)
    assert not missing_audit.structural_passed
    assert not missing_audit.measured_continuity_passed

    legacy = replace(
        complete, evidence_origin="legacy_endpoint_ramp_metadata"
    )
    legacy_audit = audit_continuation_manifest(legacy)
    assert legacy_audit.structural_passed
    assert not legacy_audit.measured_continuity_passed
    assert (
        "legacy_endpoint_ramp_is_not_measured_continuation"
        in legacy_audit.failure_reasons
    )


def test_manifest_audit_rejects_reused_images_and_changing_basis_states():
    grid = design_raw_dac_continuation_grid(
        0.0, 1.0, max_abs_dq_per_dac=0.1
    )
    frame_count = 3 * grid.raw_dac_values.size
    complete = build_experimental_continuation_manifest(
        grid,
        acquisition_id="il2-forward",
        lens_name="IL2",
        raw_dac_channel_name="IL2",
        route="downstream_calibrated_response",
        position_plus_state_x_theta_x_y_theta_y=(2e-9, 0.0, 0.0, 0.0),
        tilt_plus_state_x_theta_x_y_theta_y=(0.0, 1e-3, 0.0, 0.0),
        sample_state_covariance=np.diag((1e-20, 1e-8, 1e-20, 1e-8)),
        recorded_image_ids=tuple(f"image-{index}" for index in range(frame_count)),
    )

    reused_frames = list(complete.frames)
    reused_frames[1] = replace(
        reused_frames[1], image_record_id=reused_frames[0].image_record_id
    )
    reused_audit = audit_continuation_manifest(
        replace(complete, frames=tuple(reused_frames))
    )
    assert not reused_audit.structural_passed
    assert not reused_audit.measured_continuity_passed
    assert "image_record_id_reused" in reused_audit.failure_reasons

    changed_frames = list(complete.frames)
    position_index = next(
        index
        for index, frame in enumerate(changed_frames)
        if frame.grid_index == 1 and frame.role == "position_plus"
    )
    changed_frames[position_index] = replace(
        changed_frames[position_index],
        sample_state_x_theta_x_y_theta_y=(3e-9, 0.0, 0.0, 0.0),
    )
    changed_audit = audit_continuation_manifest(
        replace(complete, frames=tuple(changed_frames))
    )
    assert not changed_audit.structural_passed
    assert not changed_audit.measured_continuity_passed
    assert (
        "sample_input_state_changed_within_scan"
        in changed_audit.failure_reasons
    )


def test_manifest_json_payload_round_trip_retains_provenance_and_audit():
    grid = design_raw_dac_continuation_grid(
        -1.0,
        1.0,
        max_abs_dq_per_dac=0.1,
        anchor_raw_dac_values=(0.0,),
    )
    manifest = build_experimental_continuation_manifest(
        grid,
        acquisition_id="cmini-forward",
        lens_name="C-mini",
        raw_dac_channel_name="C-mini",
        route="upstream_astigmatic_tag",
        tag_generation_z_m=-0.15,
        varied_lens_entrance_z_m=-0.1,
        recorded_image_ids=tuple(
            f"image-{index}" for index in range(grid.raw_dac_values.size)
        ),
    )
    payload = continuation_manifest_to_payload(manifest)
    restored = continuation_manifest_from_payload(payload)

    assert payload["schema"] == CONTINUATION_MANIFEST_SCHEMA
    assert continuation_manifest_to_payload(restored) == payload
    assert audit_continuation_manifest(restored).measured_continuity_passed

    payload_with_unknown_key = dict(payload)
    payload_with_unknown_key["unsupported_future_claim"] = True
    with pytest.raises(ValueError, match="keys differ"):
        continuation_manifest_from_payload(payload_with_unknown_key)


def test_public_package_reexports_continuation_api():
    import temgym_core

    assert temgym_core.DACContinuationGrid.__module__.endswith(
        "phase_branch_continuation"
    )
    assert temgym_core.audit_continuation_manifest is audit_continuation_manifest


def test_constants_encode_conservative_quarter_pi_gate():
    assert MAX_CONTINUATION_PHASE_STEP_RAD == np.pi / 4.0
    assert CONTINUATION_MANIFEST_SCHEMA.endswith("/v1")
