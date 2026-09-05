import importlib
import json
from pathlib import Path
import tomllib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate

calibration_module = importlib.import_module("temgym_core.neoarm_calibration")

from temgym_core.gaussian import make_gaussian
from temgym_core.neoarm_calibration import (
    CheckpointObservableDataset,
    CheckpointCompatibilityError,
    CheckpointedAcquisition,
    DetectorGeometry,
    DetectorNoiseModel,
    FramePrediction,
    FrameRequest,
    ForwardValidationThresholds,
    GaussianRefinementSnapshot,
    GaussianRenderResult,
    GaussianSliceSafetyReport,
    IncompleteCheckpointError,
    KnownFourierFeatureSet,
    MagnificationDACInfeasibleError,
    NoisyDetectorFrame,
    RefinementRegion,
    SignedObjectDefocusInfeasibleError,
    apply_transfer_matrix_to_gaussians,
    aggregate_checkpoint_observables_for_fit,
    assess_gaussian_slice_safety,
    build_acquisition_manifest,
    build_heterogeneous_dataset_from_checkpoint,
    build_multiresolution_gaussian_topology,
    calibrated_state_centroid_covariance_px2,
    collins_propagate_2d,
    conjugate_gaussian_grid_reference,
    design_nominal_magnification_dacs,
    design_signed_object_defocus_dacs,
    estimate_gaussian_workload,
    evaluate_held_out_metrics,
    extract_image_observables,
    fit_fourier_affine,
    frozen_topology_fingerprint,
    gaussian_bundle_from_sample,
    gaussian_cardinal_coefficients,
    hard_edge_column_from_config,
    interact_gaussian_with_sample,
    known_fourier_features_from_sample_view,
    propagate_gaussians_through_column,
    predict_frame,
    refine_gaussian_topology,
    render_frame_sequence_checkpointed,
    render_gaussian_detector_plane,
    simulate_detector_counts,
    simpson_weights,
    validate_forward_models,
    validate_gaussian_refinement_sequence,
)
from temgym_core.microscope_inversion import (
    AcquisitionPlan,
    AcquisitionState,
    FrameObservationBlock,
    HeterogeneousInversionDataset,
    SamplePlaneState,
    compact_30_state_plan,
    symplectic_defect,
)


jax.config.update("jax_enable_x64", True)


class _AnalyticView:
    z = 0.325

    @staticmethod
    def evaluate_actions(xy):
        xy = jnp.asarray(xy)
        path = 1e-12 + 0.2e-3 * xy[..., 0]
        log_amplitude = -0.1 - 1e6 * xy[..., 1] ** 2
        return jnp.stack((path, log_amplitude), axis=-1)

    @staticmethod
    def __call__(beam):
        return beam


def test_detector_uses_pixel_centres_and_yx_storage():
    detector = DetectorGeometry(
        shape_yx=(3, 5),
        pixel_size_yx_m=(2.0, 3.0),
        centre_xy_m=(10.0, 20.0),
    )
    np.testing.assert_allclose(detector.x_coords_m, [4.0, 7.0, 10.0, 13.0, 16.0])
    np.testing.assert_allclose(detector.y_coords_m, [18.0, 20.0, 22.0])
    assert detector.coords_xy_m.shape == (3, 5, 2)
    np.testing.assert_allclose(detector.coords_xy_m[1, 2], [10.0, 20.0])


def test_detector_affine_maps_render_coordinates_and_state_covariance():
    affine = np.asarray(((2.0, 0.0), (0.0, 1.0)))
    detector = DetectorGeometry(
        shape_yx=(5, 9),
        pixel_size_yx_m=(1e-6, 1e-6),
        affine_xy=affine,
    )
    np.testing.assert_allclose(
        detector.optical_coords_xy_m[2, :, 0],
        detector.x_coords_m / 2.0,
    )
    np.testing.assert_allclose(
        detector.column_to_detector_xy(detector.optical_coords_xy_m),
        detector.coords_xy_m,
    )

    beam = make_gaussian(
        z=detector.z_m,
        voltage=200e3,
        waist_x=2e-6,
        waist_y=2e-6,
    )
    identity_render = render_gaussian_detector_plane(
        beam,
        DetectorGeometry(
            shape_yx=detector.shape_yx,
            pixel_size_yx_m=detector.pixel_size_yx_m,
            z_m=detector.z_m,
        ),
        tile_rows=2,
        method="loop",
    )
    affine_render = render_gaussian_detector_plane(
        beam,
        detector,
        tile_rows=2,
        method="loop",
    )
    # In calibrated detector coordinates the x profile is stretched by two.
    np.testing.assert_allclose(
        affine_render.field_yx[2, ::2],
        identity_render.field_yx[2, 2:7],
        rtol=2e-12,
        atol=2e-12,
    )

    transfer = np.eye(4)
    state_covariance = np.diag((4e-12, 0.0, 9e-12, 0.0))
    covariance_px2 = calibrated_state_centroid_covariance_px2(
        transfer,
        state_covariance,
        detector,
    )
    np.testing.assert_allclose(covariance_px2, np.diag((16.0, 9.0)))
    with pytest.raises(ValueError, match="invertible"):
        DetectorGeometry(affine_xy=np.zeros((2, 2)))


def test_conjugate_grid_reference_directly_samples_sample_without_beamlets():
    detector = DetectorGeometry(
        shape_yx=(5, 7),
        pixel_size_yx_m=(0.4e-6, 0.6e-6),
        affine_xy=np.asarray(((1.1, 0.08), (-0.03, 0.9))),
    )
    matrix = np.eye(4)
    matrix[0, 0] = 3.0
    matrix[2, 2] = 2.0
    centre = np.asarray((0.15e-6, -0.10e-6))
    waist = np.asarray((1.3e-6, 0.9e-6))
    result = conjugate_gaussian_grid_reference(
        _AnalyticView(),
        matrix,
        detector,
        incident_centre_xy_m=centre,
        incident_waist_xy_m=waist,
        wavelength_m=2.5e-12,
        sample_bandlimit_spacing_m=0.2e-9,
    )

    A = matrix[np.ix_((0, 2), (0, 2))]
    sample_xy = np.einsum(
        "ij,...j->...i", np.linalg.inv(A), detector.optical_coords_xy_m
    )
    actions = np.asarray(
        _AnalyticView.evaluate_actions(jnp.asarray(sample_xy.reshape((-1, 2))))
    ).reshape((*detector.shape_yx, 2))
    offset = sample_xy - centre
    expected = (
        2.0
        / (np.pi * waist[0] * waist[1])
        * np.exp(
            -2.0
            * (
                (offset[..., 0] / waist[0]) ** 2
                + (offset[..., 1] / waist[1]) ** 2
            )
        )
        * np.exp(2.0 * actions[..., 1])
        / abs(np.linalg.det(A))
    )
    np.testing.assert_allclose(result.intensity_yx, expected, rtol=2e-13, atol=0.0)
    np.testing.assert_allclose(result.magnification_block_xy, A)
    assert result.maximum_b_displacement_px == 0.0


def test_conjugate_grid_reference_rejects_resolved_nonzero_b_block():
    matrix = np.eye(4)
    matrix[0, 1] = 0.02
    detector = DetectorGeometry(
        shape_yx=(5, 5), pixel_size_yx_m=(10e-6, 10e-6)
    )
    with pytest.raises(RuntimeError, match="Use Collins/CZT"):
        conjugate_gaussian_grid_reference(
            _AnalyticView(),
            matrix,
            detector,
            incident_centre_xy_m=(0.0, 0.0),
            incident_waist_xy_m=(1e-6, 1e-6),
            wavelength_m=2.5e-12,
            sample_bandlimit_spacing_m=0.2e-9,
            maximum_b_displacement_px=1e-3,
        )


def test_multiresolution_topology_is_frozen_and_has_no_implicit_cap():
    region = RefinementRegion((-0.25, 0.25, -0.25, 0.25), 0.05, "lattice")
    topology = build_multiresolution_gaussian_topology(
        (-1.0, 1.0, -1.0, 1.0),
        0.2,
        centres_per_period=8,
        refinement_regions=(region,),
    )
    assert topology.beam_count > 80 * 80
    assert "lattice" in topology.labels
    assert not topology.centres_xy_m.flags.writeable
    with pytest.raises(ValueError):
        topology.centres_xy_m[0, 0] = 1.0
    detector = DetectorGeometry(shape_yx=(8, 8))
    estimate = estimate_gaussian_workload(topology, detector, tile_shape_yx=(4, 4))
    assert estimate.interaction_count == topology.beam_count * 64
    assert "beamlets" in estimate.human_summary()


def test_refinement_unions_errors_across_states_and_preserves_area():
    topology = build_multiresolution_gaussian_topology(
        (-1.0, 1.0, -1.0, 1.0), 1.0, centres_per_period=8
    )
    errors = np.zeros((2, topology.beam_count))
    errors[0, 2] = 1.2
    errors[1, 7] = 2.0
    powers = np.ones_like(errors)
    refined = refine_gaussian_topology(
        topology,
        errors,
        relative_power_by_state=powers,
        factor=2,
    )
    assert refined.refined_parent_mask.sum() == 2
    assert refined.new_beam_count == topology.beam_count + 2 * 3
    np.testing.assert_allclose(
        refined.topology.cell_area_m2.sum(), topology.cell_area_m2.sum()
    )
    assert not refined.topology.centres_xy_m.flags.writeable


def test_gaussian_bundle_samples_pathlength_and_log_field_amplitude():
    topology = build_multiresolution_gaussian_topology(
        (-2e-9, 2e-9, -1e-9, 1e-9), 1e-9, centres_per_period=8
    )
    beam = gaussian_bundle_from_sample(_AnalyticView(), topology, voltage=200e3)
    assert beam.r_xy.shape == topology.centres_xy_m.shape
    expected_path = 1e-12 + 0.2e-3 * topology.centres_xy_m[:, 0]
    np.testing.assert_allclose(np.asarray(beam.pathlength), expected_path)
    assert np.all(np.abs(np.asarray(beam.amplitude)) > 0.0)


def test_rectangular_cardinal_expansion_reconstructs_grid_knots():
    y = np.arange(7)[:, None]
    x = np.arange(11)[None, :]
    target = np.exp(0.1j * x) * (1.0 + 0.02 * y)
    widths = (1.4, 1.7)
    coeff = gaussian_cardinal_coefficients(
        target, width_over_step_yx=widths, kernel_tolerance=1e-14, ridge=1e-12
    )
    Ky = np.exp(-((np.arange(7)[:, None] - np.arange(7)[None, :]) / widths[0]) ** 2)
    Kx = np.exp(-((np.arange(11)[:, None] - np.arange(11)[None, :]) / widths[1]) ** 2)
    reconstructed = Ky @ coeff @ Kx.T
    np.testing.assert_allclose(reconstructed, target, rtol=2e-8, atol=2e-8)


def test_jax_cardinal_coefficients_match_scipy_without_host_rhs_solve(monkeypatch):
    y = np.arange(9)[:, None]
    x = np.arange(13)[None, :]
    target = (1.0 + 0.03 * y) * np.exp(0.13j * x + 0.02j * y)
    kwargs = dict(
        width_over_step_yx=(1.35, 1.75),
        kernel_tolerance=1e-14,
        ridge=1e-11,
    )
    reference = gaussian_cardinal_coefficients(target, method="scipy", **kwargs)

    def forbidden_host_solve(*_args, **_kwargs):
        raise AssertionError("SciPy RHS solve was called by the JAX path")

    monkeypatch.setattr(calibration_module.linalg, "solve_banded", forbidden_host_solve)
    device_result = gaussian_cardinal_coefficients(
        jnp.asarray(target),
        method="jax",
        **kwargs,
    )

    assert isinstance(device_result, jax.Array)
    np.testing.assert_allclose(
        np.asarray(device_result),
        reference,
        rtol=2e-10,
        atol=2e-10,
    )


def test_required_gpu_cardinal_coefficients_fail_closed(monkeypatch):
    monkeypatch.setattr(calibration_module, "gpu_kernel_available", lambda: False)
    with pytest.raises(RuntimeError, match="host/SciPy fallback is disabled"):
        gaussian_cardinal_coefficients(
            jnp.ones((3, 4), dtype=jnp.complex128),
            method="gpu_kernel_required",
        )


def test_sample_interaction_has_explicit_local_and_reexpansion_routes():
    beam = make_gaussian(
        voltage=200e3,
        waist_x=1e-9,
        waist_y=1e-9,
        z=_AnalyticView.z,
    )
    local = interact_gaussian_with_sample(_AnalyticView(), beam, mode="local")
    assert local.route == "local_action"
    assert local.incident_field_yx is None

    axis = np.linspace(-3e-9, 3e-9, 9)
    exact = interact_gaussian_with_sample(
        _AnalyticView(),
        beam,
        mode="reexpand",
        x_coords_m=axis,
        y_coords_m=axis,
        evaluation_method="loop",
    )
    assert exact.route == "sample_plane_reexpansion"
    assert exact.incident_field_yx.shape == (9, 9)
    assert exact.transmitted_field_yx.shape == (9, 9)
    assert exact.beam.r_xy.shape == (81, 2)


def test_transfer_matrix_propagates_gaussian_centres_and_endpoint_action():
    beam = make_gaussian(
        x=np.array([2e-6]),
        y=np.array([-1e-6]),
        dx=np.array([3e-3]),
        dy=np.array([2e-3]),
        z=np.array([0.0]),
        voltage=np.array([200e3]),
        waist_x=np.array([1e-7]),
        waist_y=np.array([1e-7]),
    )
    distance = 0.2
    matrix = np.array(
        [
            [1.0, distance, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, distance],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    out = apply_transfer_matrix_to_gaussians(beam, matrix, z_out_m=distance)
    np.testing.assert_allclose(np.asarray(out.x), [2e-6 + distance * 3e-3])
    np.testing.assert_allclose(np.asarray(out.y), [-1e-6 + distance * 2e-3])
    expected_path = distance + 0.5 * distance * ((3e-3) ** 2 + (2e-3) ** 2)
    np.testing.assert_allclose(np.asarray(out.pathlength), [expected_path])


def test_simpson_weights_integrate_cubic_on_even_grid():
    x = np.linspace(-1.0, 1.0, 16)
    weights = simpson_weights(x)
    assert weights.shape == x.shape
    values = x**3 + 2 * x**2
    np.testing.assert_allclose(
        np.sum(weights * values), integrate.simpson(values, x=x)
    )


def test_collins_reference_preserves_separable_rank():
    x = np.linspace(-3e-6, 3e-6, 17)
    y = np.linspace(-2e-6, 2e-6, 15)
    u_x = np.exp(-(x / 1e-6) ** 2)
    u_y = np.exp(-(y / 0.8e-6) ** 2)
    field = u_y[:, None] * u_x[None, :]
    out = collins_propagate_2d(
        field,
        x,
        y,
        x,
        y,
        abcd_x=(1.0, 0.1, 0.0, 1.0),
        abcd_y=(1.0, 0.1, 0.0, 1.0),
        wavelength_m=2.51e-12,
    )
    singular_values = np.linalg.svd(out, compute_uv=False)
    assert out.shape == field.shape
    assert np.all(np.isfinite(out))
    assert singular_values[1] / singular_values[0] < 1e-12


def test_image_observables_and_noise_simulator_are_deterministic():
    y, x = np.indices((41, 51), dtype=float)
    image = np.exp(-0.5 * (((x - 27.0) / 4.0) ** 2 + ((y - 18.0) / 2.0) ** 2))
    observables = extract_image_observables(image)
    np.testing.assert_allclose(observables.centroid_xy_px, (27.0, 18.0), atol=1e-5)
    np.testing.assert_allclose(observables.principal_sigma_px, (4.0, 2.0), atol=2e-4)
    a = simulate_detector_counts(image, dose_electrons=1e5, read_noise_e=0.5, seed=5)
    b = simulate_detector_counts(image, dose_electrons=1e5, read_noise_e=0.5, seed=5)
    np.testing.assert_array_equal(a, b)


def test_even_detector_fourier_origin_and_absolute_intensity_are_preserved():
    y, x = np.indices((64, 64), dtype=float)
    grating = 1.0 + 0.2 * np.cos(2.0 * np.pi * (7.0 * x + 5.0 * y) / 64.0)
    vector = extract_image_observables(grating, n_fourier_vectors=1).fourier_vectors_yx_px[0]
    np.testing.assert_allclose(np.abs(vector), (5.0, 7.0), atol=1e-3)

    full = simulate_detector_counts(
        np.ones((8, 8)), dose_electrons=200_000, incident_intensity_sum=64, seed=4
    )
    half = simulate_detector_counts(
        0.5 * np.ones((8, 8)),
        dose_electrons=200_000,
        incident_intensity_sum=64,
        seed=5,
    )
    assert half.sum() / full.sum() == pytest.approx(0.5, abs=0.01)


def test_subpixel_fourier_vectors_support_subpercent_scale_measurement():
    size = 256
    y, x = np.indices((size, size), dtype=float)
    expected = np.asarray((5.35, 17.2))
    image = 1.0 + 0.25 * np.cos(
        2.0 * np.pi * (expected[1] * x + expected[0] * y) / size + 0.37
    )
    measured = extract_image_observables(
        image, n_fourier_vectors=1
    ).fourier_vectors_yx_px[0]
    measured *= np.sign(measured[np.argmax(np.abs(measured))])
    np.testing.assert_allclose(measured, expected, atol=0.06)
    assert np.linalg.norm(measured - expected) / np.linalg.norm(expected) < 0.005


def test_forward_validation_uses_noise_relative_acceptance_gates():
    reference = np.ones((8, 8))
    sample_reference = np.ones((8, 8), dtype=complex)
    accepted = validate_forward_models(
        reference + 0.1,
        reference,
        noise_sigma_yx=1.0,
        candidate_sample_field_yx=sample_reference + 0.19j,
        reference_sample_field_yx=sample_reference,
        sample_field_noise_sigma=1.0,
        observable_residuals_in_sigma={
            "scale": 0.1,
            "rotation": -0.19,
            "centroid": 0.05,
            "fourier_vector": 0.08,
            "intensity": -0.12,
        },
        parameter_change_in_sigma=0.09,
    )
    assert accepted.passed
    assert accepted.sample_field_rms_in_noise_sigma == pytest.approx(0.19)
    rejected = validate_forward_models(
        reference + 0.1,
        reference,
        noise_sigma_yx=1.0,
        candidate_sample_field_yx=sample_reference + 0.21j,
        reference_sample_field_yx=sample_reference,
        sample_field_noise_sigma=1.0,
        observable_residuals_in_sigma={
            "scale": 0.0,
            "rotation": 0.0,
            "centroid": 0.0,
            "fourier_vector": 0.0,
            "intensity": 0.0,
        },
        parameter_change_in_sigma=0.0,
        thresholds=ForwardValidationThresholds(),
    )
    assert not rejected.passed

    incomplete = validate_forward_models(
        reference,
        reference,
        noise_sigma_yx=1.0,
    )
    assert not incomplete.passed
    assert "sample_field" in incomplete.missing_gates


def _refinement_snapshot(
    index,
    *,
    detector_level=None,
    fit_value=None,
    fit_standard_errors=(0.5, np.inf),
    identifiable_mask=(True, False),
    topology_id=None,
    beam_count=None,
    omit_observable=None,
):
    step = 0.05 * index
    observables = {
        "scale": 1.0 + step,
        "rotation": 12.0 + step,
        "centroid": np.asarray((step, -step)),
        "fourier_vector": np.asarray(((2.0 + step, 5.0 - step),)),
        "intensity": 10.0 + step,
    }
    if omit_observable is not None:
        observables.pop(omit_observable)
    return GaussianRefinementSnapshot(
        topology_id=f"topology-{index}" if topology_id is None else topology_id,
        beam_count=100 * (index + 1) if beam_count is None else beam_count,
        detector_intensity_yx=np.full(
            (5, 7),
            1.0 + step if detector_level is None else detector_level,
        ),
        sample_field_yx=np.ones((4, 6), dtype=complex) + 1j * step,
        observables=observables,
        fit_parameters=np.asarray(
            (0.04 * index if fit_value is None else fit_value, 7.0)
        ),
        fit_standard_errors=np.asarray(fit_standard_errors),
        identifiable_parameter_mask=np.asarray(identifiable_mask),
    )


def _validate_refinement_snapshots(snapshots):
    return validate_gaussian_refinement_sequence(
        snapshots,
        detector_noise_sigma_yx=1.0,
        sample_field_noise_sigma=1.0,
        observable_standard_errors={
            "scale": 1.0,
            "rotation": 1.0,
            "centroid": 1.0,
            "fourier_vector": 1.0,
            "intensity": 1.0,
        },
    )


def test_refinement_admission_requires_two_successive_complete_transitions():
    snapshots = [_refinement_snapshot(index) for index in range(3)]
    report = _validate_refinement_snapshots(snapshots)

    assert report.passed
    assert report.trailing_successful_refinements == 2
    assert report.failure_reasons == ()
    assert len(report.steps) == 2
    assert all(step.topology_progressed for step in report.steps)
    assert all(step.forward_validation.passed for step in report.steps)
    assert all(step.parameter_stability_passed for step in report.steps)
    last_change = report.steps[-1].forward_validation.parameter_change_in_sigma
    assert last_change == pytest.approx(0.08)

    one_transition = _validate_refinement_snapshots(snapshots[:2])
    assert not one_transition.passed
    assert one_transition.trailing_successful_refinements == 1
    assert "insufficient_refinement_history" in one_transition.failure_reasons


def test_refinement_admission_uses_strict_parameter_limit_and_conservative_sigma():
    strict_boundary = _validate_refinement_snapshots(
        [
            _refinement_snapshot(0, fit_standard_errors=(1.0, np.inf)),
            _refinement_snapshot(1, fit_value=0.1, fit_standard_errors=(1.0, np.inf)),
            _refinement_snapshot(2, fit_value=0.14, fit_standard_errors=(1.0, np.inf)),
        ]
    )
    assert strict_boundary.steps[0].forward_validation.parameter_change_in_sigma \
        == pytest.approx(0.1)
    assert strict_boundary.steps[0].forward_validation.passed
    assert not strict_boundary.steps[0].parameter_stability_passed
    assert not strict_boundary.passed

    conservative_sigma = _validate_refinement_snapshots(
        [
            _refinement_snapshot(0, fit_standard_errors=(1.0, np.inf)),
            _refinement_snapshot(1, fit_value=0.04, fit_standard_errors=(1.0, np.inf)),
            _refinement_snapshot(2, fit_value=0.10, fit_standard_errors=(0.5, np.inf)),
        ]
    )
    assert conservative_sigma.steps[-1].forward_validation.parameter_change_in_sigma \
        == pytest.approx(0.12)
    assert not conservative_sigma.passed
    assert "parameter_stability" in conservative_sigma.failure_reasons


def test_refinement_admission_requires_forward_gates_and_topology_progression():
    missing_observable = _validate_refinement_snapshots(
        [
            _refinement_snapshot(0),
            _refinement_snapshot(1),
            _refinement_snapshot(2, omit_observable="intensity"),
        ]
    )
    assert not missing_observable.passed
    assert "missing:observable:intensity" in missing_observable.failure_reasons

    repeated_topology = _validate_refinement_snapshots(
        [
            _refinement_snapshot(0),
            _refinement_snapshot(1, topology_id="topology-0"),
            _refinement_snapshot(2),
        ]
    )
    assert not repeated_topology.passed
    assert not repeated_topology.steps[0].topology_progressed
    assert "topology_progression" in repeated_topology.failure_reasons

    changed_identifiability = _validate_refinement_snapshots(
        [
            _refinement_snapshot(0),
            _refinement_snapshot(1),
            _refinement_snapshot(
                2,
                fit_standard_errors=(0.5, 0.5),
                identifiable_mask=(True, True),
            ),
        ]
    )
    assert not changed_identifiability.passed
    assert not changed_identifiability.steps[-1].identifiable_parameter_set_stable
    assert (
        "identifiable_parameter_set_changed"
        in changed_identifiability.failure_reasons
    )


def test_refinement_admission_uses_trailing_successful_refinements():
    # An early coarse comparison may fail.  It no longer blocks admission once
    # two later, genuinely refined topologies pass every gate consecutively.
    report = _validate_refinement_snapshots(
        [
            _refinement_snapshot(0, detector_level=1.0),
            _refinement_snapshot(1, detector_level=1.5),
            _refinement_snapshot(2, detector_level=1.55),
            _refinement_snapshot(3, detector_level=1.60),
        ]
    )
    assert not report.steps[0].passed
    assert report.steps[1].passed
    assert report.steps[2].passed
    assert report.trailing_successful_refinements == 2
    assert report.passed


def test_full_column_config_exposes_every_geometry_and_raw_dac_parameter():
    root = Path(__file__).resolve().parents[1]
    with (root / "tests/fixtures/neoarm_synthetic_column.toml").open("rb") as stream:
        config = tomllib.load(stream)
    column = hard_edge_column_from_config(config)
    # source + detector, 3 physical values for all 9 lenses, and one raw-DAC
    # slope for each of the seven controllable lenses.
    assert column.n_parameters == 2 + 3 * 9 + 7
    assert "CL3.q_slope_per_dac" in column.parameter_names
    assert not any(name.startswith("Obj-prefield.q_slope") for name in column.parameter_names)
    matrix = column.matrix(
        column.nominal_parameters,
        np.zeros(len(column.channel_names)),
        segment="full",
    )
    np.testing.assert_allclose(np.asarray(symplectic_defect(matrix)), 0.0, atol=2e-10)
    upstream = column.matrix(
        column.nominal_parameters,
        np.zeros(len(column.channel_names)),
        segment="source_to_sample",
    )
    downstream = column.matrix(
        column.nominal_parameters,
        np.zeros(len(column.channel_names)),
        segment="sample_to_detector",
    )
    np.testing.assert_allclose(np.asarray(downstream @ upstream), np.asarray(matrix), atol=2e-10)
    assert np.min(np.asarray(column.geometry_clearances(column.nominal_parameters))) >= 0.0


def test_default_column_bounds_keep_every_lens_body_in_its_axial_cell():
    root = Path(__file__).resolve().parents[1]
    with (root / "tests/fixtures/neoarm_synthetic_column.toml").open("rb") as stream:
        column = hard_edge_column_from_config(tomllib.load(stream))
    lower, upper = column.default_bounds(
        position_half_width_m=3e-3,
        length_ratio=(0.4, 2.5),
    )
    rng = np.random.default_rng(91)
    for _ in range(200):
        parameters = rng.uniform(lower, upper)
        assert np.min(np.asarray(column.geometry_clearances(parameters))) >= -1e-12


def test_nominal_dac_design_solves_focused_low_magnification_states():
    root = Path(__file__).resolve().parents[1]
    with (root / "tests/fixtures/neoarm_synthetic_column.toml").open("rb") as stream:
        column = hard_edge_column_from_config(tomllib.load(stream))
    targets = np.asarray((1_000.0, 2_000.0, 5_000.0))
    design = design_nominal_magnification_dacs(
        column,
        column.nominal_parameters,
        targets,
        search_points=4097,
    )
    assert design.dac_values.shape == (3, 7)
    assert design.all_targets_feasible
    assert design.maximum_relative_error < 2e-8
    np.testing.assert_allclose(design.anisotropy_ratios, 1.0, atol=2e-6)
    np.testing.assert_allclose(design.specimen_conjugacy_b_norms, 0.0, atol=1e-10)
    # Upstream condenser controls do not define the imaging magnification path.
    np.testing.assert_array_equal(design.dac_values[:, :3], 0.0)


def test_nominal_dac_design_reports_high_magnification_infeasibility():
    root = Path(__file__).resolve().parents[1]
    with (root / "tests/fixtures/neoarm_synthetic_column.toml").open("rb") as stream:
        column = hard_edge_column_from_config(tomllib.load(stream))
    targets = np.asarray((40_000.0, 80_000.0, 150_000.0, 300_000.0, 600_000.0))
    with pytest.raises(MagnificationDACInfeasibleError, match="not focus-feasible") as caught:
        design_nominal_magnification_dacs(
            column,
            column.nominal_parameters,
            targets,
            search_points=4097,
        )
    design = caught.value.design
    assert design.dac_values.shape == (5, 7)
    assert not design.all_targets_feasible
    np.testing.assert_array_equal(design.feasible_mask, np.zeros(5, dtype=bool))
    np.testing.assert_array_equal(design.dac_values[:, :3], 0.0)
    assert np.all(design.achieved_magnifications < targets)
    assert np.all(design.specimen_conjugacy_b_norms > 0.0)
    assert np.max(design.specimen_conjugacy_b_norms) < 0.02
    assert np.min(design.achieved_magnifications) > 15_000.0
    assert np.max(design.achieved_magnifications) < 24_000.0


def test_declared_unscaled_dac_search_finds_focused_40k_to_600k_modes():
    root = Path(__file__).resolve().parents[1]
    with (root / "tests/fixtures/neoarm_synthetic_column.toml").open("rb") as stream:
        config = tomllib.load(stream)
    column = hard_edge_column_from_config(config)
    experiment = config["experiment"]
    targets = np.asarray(experiment["magnifications"], dtype=float)
    design = design_nominal_magnification_dacs(
        column,
        column.nominal_parameters,
        targets,
        path_bounds=tuple(experiment["nominal_dac_search_bounds"]),
        search_points=int(experiment["nominal_dac_search_points"]),
        multistart_points=int(experiment["nominal_dac_multistart_points"]),
        specimen_conjugacy_b_norm_atol=float(
            experiment["nominal_conjugacy_b_atol_m"]
        ),
    )
    assert design.all_targets_feasible
    assert design.maximum_relative_error < 2e-8
    assert (
        design.maximum_specimen_conjugacy_b_norm
        < experiment["nominal_conjugacy_b_atol_m"]
    )
    np.testing.assert_array_equal(design.dac_values[:, :3], 0.0)


def _signed_defocus_600k_fixture():
    root = Path(__file__).resolve().parents[1]
    with (root / "tests/fixtures/neoarm_synthetic_column.toml").open(
        "rb"
    ) as stream:
        config = tomllib.load(stream)
    column = hard_edge_column_from_config(config)
    # Full precision from the deterministic focused branch. Rounding these
    # large unscaled synthetic controls visibly defocuses the 600k state.
    nominal = np.asarray(
        (
            0.0,
            0.0,
            0.0,
            -88.4846806919322,
            -88.46497467005767,
            -88.660575526563,
            -87.3721850312866,
        )
    )
    kwargs = {
        "feature_spacing_m": config["sample"]["certificate"]["si_111_spacing_m"],
        "electron_voltage_v": config["beam"]["voltage_v"],
    }
    return column, nominal, kwargs


def test_signed_object_defocus_design_matches_audited_600k_il2_il3_pair():
    column, nominal, kwargs = _signed_defocus_600k_fixture()
    design = design_signed_object_defocus_dacs(
        column,
        column.nominal_parameters,
        nominal,
        selected_channels=("IL2", "IL3"),
        **kwargs,
    )

    assert design.all_states_feasible
    assert design.nominal_magnification == pytest.approx(600_000.0, rel=1e-10)
    np.testing.assert_allclose(
        design.target_object_defocus_m * 1e9,
        (-19.60176683, 19.60176683),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        design.achieved_object_defocus_m,
        design.target_object_defocus_m,
        rtol=0.0,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        design.achieved_magnifications,
        design.nominal_magnification,
        rtol=1e-10,
    )
    for dacs, target in zip(
        design.dac_values,
        design.target_object_defocus_m,
        strict=True,
    ):
        matrix = np.asarray(
            column.matrix(
                column.nominal_parameters,
                dacs,
                segment="sample_to_detector",
            )
        )
        a_block = matrix[np.ix_((0, 2), (0, 2))]
        b_block = matrix[np.ix_((0, 2), (1, 3))]
        assert np.sqrt(np.linalg.det(a_block)) == pytest.approx(
            design.nominal_magnification,
            rel=1e-10,
        )
        assert 0.5 * np.trace(np.linalg.solve(a_block, b_block)) == pytest.approx(
            target,
            abs=2e-13,
        )
    il2 = column.channel_names.index("IL2")
    il3 = column.channel_names.index("IL3")
    np.testing.assert_allclose(
        design.dac_perturbations[0, (il2, il3)],
        (1.44861e-5, 1.11685e-5),
        rtol=4e-5,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        design.dac_perturbations[1, (il2, il3)],
        -design.dac_perturbations[0, (il2, il3)],
        rtol=0.0,
        atol=3e-10,
    )
    np.testing.assert_allclose(
        design.cumulative_rotation_changes_rad,
        (-10.986e-6, 10.986e-6),
        rtol=1e-4,
        atol=2e-9,
    )
    unchanged = np.ones(len(column.channel_names), dtype=bool)
    unchanged[[il2, il3]] = False
    np.testing.assert_array_equal(design.dac_perturbations[:, unchanged], 0.0)
    assert not design.dac_values.flags.writeable
    assert not design.constraint_jacobian_singular_values.flags.writeable


def test_signed_object_defocus_design_can_preserve_unwrapped_larmor_rotation():
    column, nominal, kwargs = _signed_defocus_600k_fixture()
    design = design_signed_object_defocus_dacs(
        column,
        column.nominal_parameters,
        nominal,
        selected_channels=("IL1", "IL2", "IL3"),
        preserve_rotation=True,
        **kwargs,
    )

    assert design.all_states_feasible
    assert design.constraint_jacobian_singular_values.shape == (3,)
    np.testing.assert_allclose(
        design.achieved_object_defocus_m,
        design.target_object_defocus_m,
        rtol=0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        design.cumulative_rotation_changes_rad,
        0.0,
        rtol=0.0,
        atol=1e-10,
    )
    assert np.max(design.magnification_relative_errors) < 1e-9


def test_signed_object_defocus_design_fails_closed_with_diagnostics():
    column, nominal, kwargs = _signed_defocus_600k_fixture()
    with pytest.raises(ValueError, match="At least 3 selected controls"):
        design_signed_object_defocus_dacs(
            column,
            column.nominal_parameters,
            nominal,
            selected_channels=("IL2", "IL3"),
            preserve_rotation=True,
            **kwargs,
        )
    with pytest.raises(ValueError, match="do not act on the sample-to-detector"):
        design_signed_object_defocus_dacs(
            column,
            column.nominal_parameters,
            nominal,
            selected_channels=("CL1", "CL3"),
            **kwargs,
        )
    with pytest.raises(
        SignedObjectDefocusInfeasibleError,
        match="not feasible within the selected raw-DAC bounds",
    ) as caught:
        design_signed_object_defocus_dacs(
            column,
            column.nominal_parameters,
            nominal,
            selected_channels=("IL2", "IL3"),
            selected_dac_half_width=1e-8,
            **kwargs,
        )
    assert not caught.value.design.all_states_feasible
    assert "negative: target=" in str(caught.value)
    assert "positive: target=" in str(caught.value)


def test_gaussian_slice_safety_decodes_every_lens_and_state():
    root = Path(__file__).resolve().parents[1]
    with (root / "tests/fixtures/neoarm_synthetic_column.toml").open("rb") as stream:
        column = hard_edge_column_from_config(tomllib.load(stream))
    controls = np.zeros((3, len(column.channel_names)), dtype=float)
    controls[1, column.channel_names.index("IL1")] = -4.0
    controls[2, column.channel_names.index("CL3")] = 3.0
    report = assess_gaussian_slice_safety(
        column,
        column.nominal_parameters,
        controls,
        slices_per_lens=4,
        maximum_phase_advance_per_slice_rad=0.5,
    )

    assert isinstance(report, GaussianSliceSafetyReport)
    assert report.integrated_strengths_by_state_lens.shape == (
        3,
        len(column.lenses),
    )
    il1 = report.lens_names.index("IL1")
    cl3 = report.lens_names.index("CL3")
    np.testing.assert_allclose(
        report.integrated_strengths_by_state_lens[:, il1],
        [0.25, -1.75, 0.25],
    )
    np.testing.assert_allclose(
        report.integrated_strengths_by_state_lens[:, cl3],
        [0.15, 0.15, 2.10],
    )
    assert report.maximum_abs_q == pytest.approx(2.10)
    assert report.worst_state_index == 2
    assert report.worst_lens_name == "CL3"
    assert report.recommended_slices_per_lens == 5
    assert report.requested_maximum_phase_advance_rad == pytest.approx(0.525)
    assert report.requested_is_safe is False
    assert not report.integrated_strengths_by_state_lens.flags.writeable


def test_current_focused_600k_state_requires_more_than_8_or_24_slices():
    root = Path(__file__).resolve().parents[1]
    with (root / "tests/fixtures/neoarm_synthetic_column.toml").open("rb") as stream:
        column = hard_edge_column_from_config(tomllib.load(stream))
    # Deterministic focused 600k branch from the declared wide raw-DAC search.
    focused_600k_dacs = np.asarray(
        (0.0, 0.0, 0.0, -88.48468069, -88.46497467, -88.66057553, -87.37218503)
    )
    report_24 = assess_gaussian_slice_safety(
        column,
        column.nominal_parameters,
        focused_600k_dacs,
        slices_per_lens=24,
    )
    report_8 = assess_gaussian_slice_safety(
        column,
        column.nominal_parameters,
        focused_600k_dacs,
        slices_per_lens=8,
    )
    report_recommended = assess_gaussian_slice_safety(
        column,
        column.nominal_parameters,
        focused_600k_dacs,
        slices_per_lens=report_24.recommended_slices_per_lens,
    )

    assert report_24.maximum_abs_q == pytest.approx(43.992340345)
    assert report_24.worst_lens_name == "IL1"
    assert report_24.recommended_slices_per_lens == 57
    assert report_8.requested_is_safe is False
    assert report_24.requested_is_safe is False
    assert report_recommended.requested_is_safe is True
    assert "recommended uniform slices=57" in report_24.human_summary()


def test_coarse_8_and_24_slice_counts_can_select_wrong_metaplectic_sheet():
    root = Path(__file__).resolve().parents[1]
    with (root / "tests/fixtures/neoarm_synthetic_column.toml").open("rb") as stream:
        column = hard_edge_column_from_config(tomllib.load(stream))
    beam = make_gaussian(
        z=column.sample_z_m,
        voltage=200e3,
        waist_x=45e-9,
        waist_y=45e-9,
    )
    cases = (
        (
            8,
            np.asarray(
                (0.0, 0.0, 0.0, -31.972226444, -31.841226183, -30.046920348, -30.353479531)
            ),
        ),
        (
            24,
            np.asarray(
                (0.0, 0.0, 0.0, -50.800861215, -51.876378112, -51.268221784, -47.943988295)
            ),
        ),
    )
    for coarse_slices, dacs in cases:
        coarse = propagate_gaussians_through_column(
            beam,
            column,
            column.nominal_parameters,
            dacs,
            segment="sample_to_detector",
            slices_per_lens=coarse_slices,
        )
        refined = propagate_gaussians_through_column(
            beam,
            column,
            column.nominal_parameters,
            dacs,
            segment="sample_to_detector",
            slices_per_lens=64,
        )
        amplitude_ratio = complex(np.asarray(coarse.amplitude).item()) / complex(
            np.asarray(refined.amplitude).item()
        )
        # Ray/Q maps agree, but the principal determinant root has selected the
        # opposite metaplectic sheet: an otherwise invisible global pi phase.
        assert amplitude_ratio.real < -0.999999
        assert abs(amplitude_ratio.imag) < 1e-8
        np.testing.assert_allclose(
            np.asarray(coarse.Q_inv),
            np.asarray(refined.Q_inv),
            rtol=2e-9,
            atol=1e-9,
        )


def test_gaussian_slice_safety_rejects_nonprotective_phase_limit():
    root = Path(__file__).resolve().parents[1]
    with (root / "tests/fixtures/neoarm_synthetic_column.toml").open("rb") as stream:
        column = hard_edge_column_from_config(tomllib.load(stream))
    with pytest.raises(ValueError, match="strictly between zero and pi/2"):
        assess_gaussian_slice_safety(
            column,
            column.nominal_parameters,
            np.zeros(len(column.channel_names)),
            maximum_phase_advance_per_slice_rad=np.pi / 2,
        )


def test_full_column_and_sample_jacobian_are_jax_differentiable():
    root = Path(__file__).resolve().parents[1]
    with (root / "tests/fixtures/neoarm_synthetic_column.toml").open("rb") as stream:
        column = hard_edge_column_from_config(tomllib.load(stream))
    parameters = jnp.asarray(column.nominal_parameters)
    dacs = jnp.linspace(-0.1, 0.1, len(column.channel_names))
    jacobian = jax.jacfwd(
        lambda values: column.sample_position_jacobian(values, dacs).reshape(-1)
    )(parameters)
    assert jacobian.shape == (4, column.n_parameters)
    assert np.all(np.isfinite(np.asarray(jacobian)))
    # Upstream condenser strength cannot change a sample-to-detector map.
    cl1_q_index = column.parameter_names.index("CL1.q_bias")
    np.testing.assert_allclose(np.asarray(jacobian[:, cl1_q_index]), 0.0, atol=1e-13)


def test_lifted_gaussian_column_matches_ray_map_and_composes_at_sample():
    root = Path(__file__).resolve().parents[1]
    with (root / "tests/fixtures/neoarm_synthetic_column.toml").open("rb") as stream:
        column = hard_edge_column_from_config(tomllib.load(stream))
    parameters = column.nominal_parameters
    dacs = np.linspace(-0.12, 0.15, len(column.channel_names))
    beam = make_gaussian(
        x=0.7e-6,
        y=-0.3e-6,
        dx=0.12e-3,
        dy=-0.08e-3,
        z=column.source_z_m,
        voltage=200e3,
        waist_x=0.4e-6,
        waist_y=0.35e-6,
    )
    direct = propagate_gaussians_through_column(
        beam, column, parameters, dacs, segment="full", slices_per_lens=8
    )
    at_sample = propagate_gaussians_through_column(
        beam, column, parameters, dacs, segment="source_to_sample", slices_per_lens=8
    )
    composed = propagate_gaussians_through_column(
        at_sample,
        column,
        parameters,
        dacs,
        segment="sample_to_detector",
        slices_per_lens=8,
    )
    state = np.asarray((beam.x, beam.dx, beam.y, beam.dy), dtype=float)
    expected = np.asarray(column.matrix(parameters, dacs, segment="full")) @ state
    np.testing.assert_allclose(
        np.asarray((direct.x, direct.dx, direct.y, direct.dy), dtype=float),
        expected,
        rtol=2e-10,
        atol=2e-12,
    )
    np.testing.assert_allclose(np.asarray(composed.r_xy), np.asarray(direct.r_xy), atol=2e-12)
    np.testing.assert_allclose(np.asarray(composed.d_xy), np.asarray(direct.d_xy), atol=2e-12)
    assert np.isfinite(np.asarray(direct.amplitude)).all()


def test_manifest_expands_required_measurements_to_exactly_230_frames():
    plan = compact_30_state_plan()
    manifest = build_acquisition_manifest(plan)
    assert len(manifest) == plan.total_frames == 230
    assert sum(frame.measurement == "indexed_diffraction" for frame in manifest) == 6
    assert sum(frame.measurement == "basis_orthogonal" for frame in manifest) == 6
    assert sum(frame.measurement == "basis_negative" for frame in manifest) == 6
    for state_index, state in enumerate(plan.states):
        frames = [frame for frame in manifest if frame.optical_state_index == state_index]
        assert len(frames) == state.frame_count
        assert all(frame.raw_dac_values == state.dac_values for frame in frames)
        assert {"perforation", "position_basis", "angle_basis", "sample"}.issubset(
            {frame.measurement for frame in frames}
        )
        if state.role == "nominal":
            assert sum(frame.dac_approach == "reverse" for frame in frames) == 4
        elif state.role == "held_out":
            assert sum(frame.dac_approach == "reverse" for frame in frames) == 3
            assert sum(frame.measurement == "indexed_diffraction" for frame in frames) == 1


def test_manifest_records_executable_dac_approach_and_magnification_context():
    plan = compact_30_state_plan(target_magnifications=(40e3, 80e3, 150e3, 300e3, 600e3))
    manifest = build_acquisition_manifest(
        plan,
        approach_precondition_raw_dac_step=0.05,
        dac_ramp_steps=8,
        settling_time_s=2.0,
    )
    for frame in manifest:
        endpoint = np.asarray(frame.raw_dac_values)
        start = np.asarray(frame.dac_approach_start_values)
        expected = endpoint + (-0.05 if frame.dac_approach == "forward" else 0.05)
        np.testing.assert_allclose(start, expected)
        assert frame.dac_ramp_steps == 8
        assert frame.settling_time_s == 2.0
        assert frame.reference_mode_magnification is not None


def test_frozen_topology_fingerprint_covers_crop_pose_and_optional_beam_arrays():
    kwargs = dict(
        specimen_identity="sample-1",
        atlas_version="certificate-1",
        region_name="multilayer_region_4",
        level_index=1,
        crop_bounds_xyxy_m=(-2e-9, 2e-9, -3e-9, 3e-9),
        sample_translation_xy_m=(1e-9, -1e-9),
        projection_world_from_local=np.eye(2),
        route="sample_plane_reexpansion",
        selected_spacing_m=0.05e-9,
        routed_feature_scale_m=10e-9,
        centres_per_period=10.0,
        overlap_factor=2.0,
        interpolation_halo_knots=4,
        centres_xy_m=np.asarray(((-1e-9, 0.0), (1e-9, 0.0))),
        waist_m=np.asarray((0.3e-9, 0.3e-9)),
    )
    first = frozen_topology_fingerprint(**kwargs)
    assert first == frozen_topology_fingerprint(**kwargs)
    shifted = dict(kwargs, crop_bounds_xyxy_m=(-2e-9, 2.1e-9, -3e-9, 3e-9))
    assert frozen_topology_fingerprint(**shifted) != first
    changed_beam = dict(kwargs, waist_m=np.asarray((0.3e-9, 0.31e-9)))
    assert frozen_topology_fingerprint(**changed_beam) != first


def test_manifest_records_frozen_interaction_route_and_topology_identity():
    plan = compact_30_state_plan()
    label = plan.states[0].label
    manifest = build_acquisition_manifest(
        plan,
        interaction_route_by_state={label: "local_action"},
        frozen_topology_id_by_state={label: "atlas-v1:nominal-40k:lod-2"},
    )
    frames = [frame for frame in manifest if frame.optical_state_label == label]
    material = [
        frame
        for frame in frames
        if frame.measurement in {"sample", "sample_repeat", "indexed_diffraction"}
    ]
    vacuum = [frame for frame in frames if frame not in material]
    assert material
    assert all(frame.interaction_route == "local_action" for frame in material)
    assert all(
        frame.frozen_topology_id == "atlas-v1:nominal-40k:lod-2"
        for frame in material
    )
    assert all(frame.interaction_route == "vacuum" for frame in vacuum)
    assert all(frame.frozen_topology_id is None for frame in vacuum)

    with pytest.raises(ValueError, match="interaction_route"):
        FrameRequest(
            frame_id=0,
            optical_state_index=0,
            optical_state_label="state",
            measurement="sample",
            sample_state=SamplePlaneState(),
            raw_dac_values=(0.0,),
            interaction_route="switch_during_fit",
        )


def test_frame_predictor_runs_physical_upstream_and_downstream_vacuum_path():
    root = Path(__file__).resolve().parents[1]
    with (root / "tests/fixtures/neoarm_synthetic_column.toml").open("rb") as stream:
        column = hard_edge_column_from_config(tomllib.load(stream))
    plan = compact_30_state_plan(perturbations=0.05)
    request = build_acquisition_manifest(plan)[0]
    detector = DetectorGeometry(shape_yx=(9, 11), pixel_size_yx_m=(40e-6, 40e-6))
    source = make_gaussian(
        z=column.source_z_m,
        voltage=200e3,
        waist_x=5e-6,
        waist_y=5e-6,
    )
    prediction = predict_frame(
        column,
        column.nominal_parameters,
        request,
        detector,
        source,
        sample_view=None,
        slices_per_lens=2,
        tile_rows=3,
    )
    assert prediction.render.intensity_yx.shape == detector.shape_yx
    assert prediction.interaction_route == "vacuum"
    assert prediction.observables.total_intensity > 0.0
    covariance = calibrated_state_centroid_covariance_px2(
        column.matrix(
            column.nominal_parameters,
            np.asarray(request.raw_dac_values),
            segment="sample_to_detector",
        ),
        request.calibration_covariance + np.eye(4) * 1e-18,
        detector,
    )
    assert covariance.shape == (2, 2)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-15


def test_frame_predictor_threads_required_gpu_to_both_field_evaluations(
    monkeypatch,
):
    methods_and_shapes = []

    def fake_evaluate(_beam, grid, *, method, **_kwargs):
        methods_and_shapes.append((method, grid.shape))
        return jnp.ones(grid.shape, dtype=jnp.complex128)

    monkeypatch.setattr(calibration_module, "evaluate_gaussians", fake_evaluate)
    monkeypatch.setattr(calibration_module, "gpu_kernel_available", lambda: True)

    def forbidden_host_solve(*_args, **_kwargs):
        raise AssertionError("SciPy RHS solve was called by the required-GPU path")

    monkeypatch.setattr(calibration_module.linalg, "solve_banded", forbidden_host_solve)
    monkeypatch.setattr(
        calibration_module,
        "propagate_gaussians_through_column",
        lambda beam, *_args, **_kwargs: beam,
    )
    request = FrameRequest(
        frame_id=0,
        optical_state_index=0,
        optical_state_label="gpu-required",
        measurement="sample",
        sample_state=SamplePlaneState(label="sample"),
        raw_dac_values=(0.0,),
        interaction_route="sample_plane_reexpansion",
        frozen_topology_id="atlas:L0:gpu-required",
    )
    detector = DetectorGeometry(
        shape_yx=(5, 7),
        pixel_size_yx_m=(40e-6, 40e-6),
    )
    source = make_gaussian(
        z=_AnalyticView.z,
        voltage=200e3,
        waist_x=2e-9,
        waist_y=2e-9,
    )
    sample_axis = np.linspace(-3e-9, 3e-9, 5)

    prediction = predict_frame(
        object(),  # propagation is replaced above to isolate dispatch semantics
        np.empty(0),
        request,
        detector,
        source,
        sample_view=_AnalyticView(),
        sample_x_coords_m=sample_axis,
        sample_y_coords_m=sample_axis,
        tile_rows=2,
        evaluation_method="gpu_kernel_required",
    )

    assert prediction.render.intensity_yx.shape == detector.shape_yx
    assert isinstance(prediction.incident_sample_field_yx, jax.Array)
    assert isinstance(prediction.transmitted_sample_field_yx, jax.Array)
    assert isinstance(prediction.render.field_yx, np.ndarray)
    assert methods_and_shapes[0] == ("gpu_kernel_required", (5, 5))
    assert methods_and_shapes[1:] == [
        ("gpu_kernel_required", (2, 7)),
        ("gpu_kernel_required", (2, 7)),
        ("gpu_kernel_required", (1, 7)),
    ]


def test_frame_predictor_rejects_sample_view_selection_substitution(monkeypatch):
    monkeypatch.setattr(
        calibration_module,
        "propagate_gaussians_through_column",
        lambda beam, *_args, **_kwargs: beam,
    )
    request = FrameRequest(
        frame_id=0,
        optical_state_index=0,
        optical_state_label="frozen",
        measurement="sample",
        sample_state=SamplePlaneState(label="sample"),
        raw_dac_values=(0.0,),
        feature_family="individual_layer_widths",
        selected_region_name="multilayer_region_1",
        sample_level_index=2,
        sample_translation_xy_m=(1e-9, -2e-9),
        sample_crop_bounds_xyxy_m=(-4e-9, 4e-9, -3e-9, 3e-9),
        selected_spacing_m=0.5e-9,
        interaction_route="sample_plane_reexpansion",
        frozen_topology_id="atlas:L2:fixed:0123456789abcdef",
    )
    view = _AnalyticView()
    view.selection = SimpleNamespace(
        feature_family="individual_layer_widths",
        selected_region_name="multilayer_region_2",  # forbidden substitution
        level_index=2,
        sample_translation_xy_m=(1e-9, -2e-9),
        local_bounds_xyxy_m=(-4e-9, 4e-9, -3e-9, 3e-9),
        selected_spacing_m=0.5e-9,
    )
    source = make_gaussian(z=view.z, voltage=200e3, waist_x=2e-9, waist_y=2e-9)
    with pytest.raises(ValueError, match="frozen selection mismatch"):
        predict_frame(
            object(),
            np.empty(0),
            request,
            DetectorGeometry(shape_yx=(5, 5)),
            source,
            sample_view=view,
            sample_x_coords_m=np.linspace(-3e-9, 3e-9, 5),
            sample_y_coords_m=np.linspace(-3e-9, 3e-9, 5),
        )


def _checkpoint_requests():
    return (
        FrameRequest(
            frame_id=7,
            optical_state_index=0,
            optical_state_label="nominal",
            measurement="perforation",
            sample_state=SamplePlaneState(label="vacuum"),
            raw_dac_values=(0.0, 0.1),
            interaction_route="vacuum",
        ),
        FrameRequest(
            frame_id=19,
            optical_state_index=0,
            optical_state_label="nominal",
            measurement="sample",
            sample_state=SamplePlaneState(x=1e-9, label="sample"),
            raw_dac_values=(0.0, 0.1),
            interaction_route="local_action",
            frozen_topology_id="atlas:L2:v1:0123456789abcdef",
        ),
        FrameRequest(
            frame_id=31,
            optical_state_index=1,
            optical_state_label="wobble",
            measurement="sample_repeat",
            sample_state=SamplePlaneState(theta_x=2e-6, label="sample_repeat"),
            raw_dac_values=(0.0, 0.12),
            interaction_route="sample_plane_reexpansion",
            frozen_topology_id="atlas:L1:v1:fedcba9876543210",
            repeat_index=1,
        ),
    )


def _checkpoint_renderer(detector, calls, *, fail_on_frame_id=None):
    def render(request):
        calls.append(request.frame_id)
        if request.frame_id == fail_on_frame_id:
            raise RuntimeError("simulated renderer interruption")
        yy, xx = np.indices(detector.shape_yx, dtype=float)
        field = (request.frame_id + 1.0) * (1.0 + 0.01j * (xx + yy))
        intensity = np.abs(field) ** 2
        sample_field = (
            None
            if request.interaction_route == "vacuum"
            else np.full((2, 3), request.frame_id + 1j, dtype=complex)
        )
        result = GaussianRenderResult(
            field_yx=field,
            intensity_yx=intensity,
            elapsed_s=0.0,
        )
        return FramePrediction(
            request=request,
            render=result,
            observables=extract_image_observables(intensity),
            interaction_route=request.interaction_route,
            incident_sample_field_yx=sample_field,
            transmitted_sample_field_yx=sample_field,
        )

    return render


def test_checkpointed_render_chunks_and_resumes_without_object_pickles(tmp_path):
    detector = DetectorGeometry(
        shape_yx=(5, 7),
        pixel_size_yx_m=(40e-6, 42e-6),
        z_m=1.2,
        centre_xy_m=(1e-4, -2e-4),
        mtf_sigma_px=0.4,
        affine_xy=np.asarray(((1.02, 0.01), (-0.02, 0.98))),
    )
    requests = _checkpoint_requests()
    calls = []
    first = render_frame_sequence_checkpointed(
        requests,
        detector,
        _checkpoint_renderer(detector, calls),
        tmp_path / "physical",
        render_signature="matched-hard-edge:v1",
        render_metadata={"parameters_sha256": "abc", "slices_per_lens": 8},
        store_complex_field=True,
        store_sample_fields=True,
        storage_dtype="float32",
        max_new_frames=2,
    )
    assert not first.complete
    assert first.rendered_frame_ids == (7, 19)
    assert first.resumed_frame_ids == ()
    assert first.pending_frame_ids == (31,)
    assert calls == [7, 19]
    assert first.record(7).load_intensity().dtype == np.dtype("float32")
    assert first.record(19).load_field().dtype == np.dtype("complex64")
    assert first.record(19).incident_sample_field_path is not None

    with first.manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    payload = manifest["payload"]
    assert payload["detector"]["shape_yx"] == [5, 7]
    assert payload["detector"]["pixel_size_yx_m"] == [40e-6, 42e-6]
    assert payload["detector"]["affine_xy"] == detector.affine_xy.tolist()
    assert payload["frame_order"] == [7, 19, 31]
    assert payload["frame_requests"][1]["frozen_topology_id"].startswith("atlas:L2")
    assert payload["storage"]["format"] == "npy_numeric_allow_pickle_false"
    assert not list((tmp_path / "physical").rglob("*.npz"))

    calls.clear()
    resumed = render_frame_sequence_checkpointed(
        requests,
        detector,
        _checkpoint_renderer(detector, calls),
        tmp_path / "physical",
        render_signature="matched-hard-edge:v1",
        render_metadata={"parameters_sha256": "abc", "slices_per_lens": 8},
        store_complex_field=True,
        store_sample_fields=True,
        storage_dtype="float32",
    )
    assert resumed.complete
    assert resumed.completed_frame_ids == (7, 19, 31)
    assert resumed.resumed_frame_ids == (7, 19)
    assert resumed.rendered_frame_ids == (31,)
    assert calls == [31]
    assert resumed.record(31).actual_interaction_route == "sample_plane_reexpansion"
    assert resumed.record(31).frozen_topology_id.endswith("fedcba9876543210")


@pytest.mark.parametrize("failure", ["missing", "unequal"])
def test_checkpoint_reexpansion_requires_paired_equal_sample_fields(
    tmp_path, failure
):
    detector = DetectorGeometry(shape_yx=(4, 6))
    request = _checkpoint_requests()[2]

    def invalid_renderer(frame_request):
        prediction = _checkpoint_renderer(detector, [])(frame_request)
        transmitted = (
            None
            if failure == "missing"
            else np.ones((3, 2), dtype=np.complex128)
        )
        return FramePrediction(
            request=prediction.request,
            render=prediction.render,
            observables=prediction.observables,
            interaction_route=prediction.interaction_route,
            incident_sample_field_yx=prediction.incident_sample_field_yx,
            transmitted_sample_field_yx=transmitted,
        )

    with pytest.raises(ValueError, match="sample fields|both incident"):
        render_frame_sequence_checkpointed(
            (request,),
            detector,
            invalid_renderer,
            tmp_path / failure,
            render_signature=f"invalid-{failure}",
            store_sample_fields=True,
        )


def test_checkpointed_render_recovers_interruption_and_corrupt_frame(tmp_path):
    detector = DetectorGeometry(shape_yx=(4, 6))
    requests = _checkpoint_requests()
    directory = tmp_path / "recover"
    calls = []
    with pytest.raises(RuntimeError, match="interruption"):
        render_frame_sequence_checkpointed(
            requests,
            detector,
            _checkpoint_renderer(detector, calls, fail_on_frame_id=19),
            directory,
            render_signature="physical:v2",
        )
    assert calls == [7, 19]

    calls.clear()
    completed = render_frame_sequence_checkpointed(
        requests,
        detector,
        _checkpoint_renderer(detector, calls),
        directory,
        render_signature="physical:v2",
    )
    assert completed.complete
    assert completed.resumed_frame_ids == (7,)
    assert completed.rendered_frame_ids == (19, 31)
    assert calls == [19, 31]

    # A checksum mismatch is treated as an incomplete frame and only that
    # frame is atomically recomputed on the next invocation.
    completed.record(19).intensity_path.write_bytes(b"truncated")
    calls.clear()
    repaired = render_frame_sequence_checkpointed(
        requests,
        detector,
        _checkpoint_renderer(detector, calls),
        directory,
        render_signature="physical:v2",
    )
    assert repaired.complete
    assert repaired.rendered_frame_ids == (19,)
    assert repaired.resumed_frame_ids == (7, 31)
    assert calls == [19]
    assert np.isfinite(repaired.record(19).load_intensity()).all()


def test_checkpoint_resume_fails_closed_for_incompatible_identity(tmp_path):
    detector = DetectorGeometry(shape_yx=(4, 4))
    requests = _checkpoint_requests()
    directory = tmp_path / "incompatible"
    render_frame_sequence_checkpointed(
        requests,
        detector,
        _checkpoint_renderer(detector, []),
        directory,
        render_signature="configuration-a",
        max_new_frames=1,
    )
    with pytest.raises(CheckpointCompatibilityError, match="differ"):
        render_frame_sequence_checkpointed(
            requests,
            detector,
            _checkpoint_renderer(detector, []),
            directory,
            render_signature="configuration-b",
        )
    with pytest.raises(CheckpointCompatibilityError, match="differ"):
        render_frame_sequence_checkpointed(
            requests,
            DetectorGeometry(shape_yx=(4, 5)),
            _checkpoint_renderer(DetectorGeometry(shape_yx=(4, 5)), []),
            directory,
            render_signature="configuration-a",
        )


def test_checkpoint_rejects_renderer_route_or_request_mismatch(tmp_path):
    detector = DetectorGeometry(shape_yx=(4, 4))
    request = _checkpoint_requests()[1]

    def wrong_route(_request):
        prediction = _checkpoint_renderer(detector, [])(_request)
        return FramePrediction(
            request=prediction.request,
            render=prediction.render,
            observables=prediction.observables,
            interaction_route="sample_plane_reexpansion",
        )

    with pytest.raises(ValueError, match="declared route"):
        render_frame_sequence_checkpointed(
            (request,),
            detector,
            wrong_route,
            tmp_path / "route",
            render_signature="route-mismatch",
        )

    def wrong_request(_request):
        prediction = _checkpoint_renderer(detector, [])(_request)
        return FramePrediction(
            request=_checkpoint_requests()[0],
            render=prediction.render,
            observables=prediction.observables,
            interaction_route=prediction.interaction_route,
        )

    with pytest.raises(ValueError, match="wrong request"):
        render_frame_sequence_checkpointed(
            (request,),
            detector,
            wrong_request,
            tmp_path / "request",
            render_signature="request-mismatch",
        )


def _observable_checkpoint_fixture(tmp_path, *, max_new_frames=None):
    detector = DetectorGeometry(shape_yx=(16, 18))
    plan = AcquisitionPlan(
        channel_names=("IL1", "PL1"),
        states=(
            AcquisitionState(
                label="nominal",
                dac_values=(0.2, -0.1),
                role="nominal",
                frame_count=2,
            ),
        ),
        reference_dacs=(0.2, -0.1),
    )
    requests = (
        FrameRequest(
            frame_id=2,
            optical_state_index=0,
            optical_state_label="nominal",
            measurement="sample",
            sample_state=SamplePlaneState(label="sample"),
            raw_dac_values=(0.2, -0.1),
            interaction_route="local_action",
            frozen_topology_id="sample:L0:test:0123456789abcdef",
        ),
        FrameRequest(
            frame_id=9,
            optical_state_index=0,
            optical_state_label="nominal",
            measurement="sample_repeat",
            sample_state=SamplePlaneState(label="sample_repeat"),
            raw_dac_values=(0.2, -0.1),
            interaction_route="local_action",
            frozen_topology_id="sample:L0:test:0123456789abcdef",
            repeat_index=1,
        ),
    )

    def render(request):
        yy, xx = np.indices(detector.shape_yx, dtype=float)
        # Identical deterministic means deliberately exercise independent
        # shot-noise draws for a repeated exposure.
        intensity = 2.0 + 0.7 * np.cos(2.0 * np.pi * xx / 6.0)
        field = np.sqrt(intensity).astype(complex)
        return FramePrediction(
            request=request,
            render=GaussianRenderResult(field, intensity, 0.0),
            observables=extract_image_observables(intensity),
            interaction_route="local_action",
        )

    directory = tmp_path / "observable-checkpoint"
    render_frame_sequence_checkpointed(
        requests,
        detector,
        render,
        directory,
        render_signature="observable-test:v1",
        render_metadata={"truth": "fixed"},
        storage_dtype="float32",
        max_new_frames=max_new_frames,
    )
    return detector, plan, requests, directory


def test_checkpoint_observable_builder_makes_independent_noisy_repeats(tmp_path):
    detector, plan, requests, directory = _observable_checkpoint_fixture(tmp_path)
    known_feature = KnownFourierFeatureSet(
        labels=("layer_normal",),
        local_reciprocal_vectors_xy_per_m=np.asarray(((1.0 / 6e-9, 0.0),)),
        world_reciprocal_vectors_xy_per_m=np.asarray(((1.0 / 6e-9, 0.0),)),
        covariance_world_per_m2=np.eye(2) * 1e8,
        certificate_id="synthetic-layer-certificate",
    )
    product = build_heterogeneous_dataset_from_checkpoint(
        plan,
        requests,
        detector,
        directory,
        render_signature="observable-test:v1",
        render_metadata={"truth": "fixed"},
        storage_dtype="float32",
        default_noise_model=DetectorNoiseModel(
            dose_electrons=2.0e4,
            incident_intensity_sum=576.0,
            read_noise_e=0.25,
            dark_offset_e=0.5,
        ),
        seed=41,
        bootstrap_replicates=12,
        fourier_features_by_family={"auto": known_feature},
        moment_observables_by_measurement={
            "sample": calibration_module._MOMENT_OBSERVABLE_NAMES,
            "sample_repeat": calibration_module._MOMENT_OBSERVABLE_NAMES,
        },
    )
    assert isinstance(product, CheckpointObservableDataset)
    assert isinstance(product.dataset, HeterogeneousInversionDataset)
    assert product.acquisition.complete
    assert product.dataset.n_observations == 16
    assert [block.observable_names[-2:] for block in product.dataset.blocks] == [
        ("fourier_layer_normal_y_px", "fourier_layer_normal_x_px"),
        ("fourier_layer_normal_y_px", "fourier_layer_normal_x_px"),
    ]
    for block in product.dataset.blocks:
        assert block.values.shape == (8,)
        assert np.linalg.eigvalsh(block.covariance).min() > 0.0
    first = product.frame(2)
    repeat = product.frame(9)
    assert first.noise_seed != repeat.noise_seed
    assert not np.array_equal(first.counts_yx, repeat.counts_yx)
    assert not first.counts_yx.flags.writeable
    frame_metadata = product.dataset.metadata["frame_metadata"]
    assert frame_metadata["2"]["measurement"] == "sample"
    assert frame_metadata["9"]["measurement"] == "sample_repeat"

    reproduced = build_heterogeneous_dataset_from_checkpoint(
        plan,
        requests,
        detector,
        directory,
        render_signature="observable-test:v1",
        render_metadata={"truth": "fixed"},
        storage_dtype="float32",
        default_noise_model=DetectorNoiseModel(2.0e4, 576.0, 0.25, 0.5),
        seed=41,
        bootstrap_replicates=12,
        fourier_features_by_family={"auto": known_feature},
        moment_observables_by_measurement={
            "sample": calibration_module._MOMENT_OBSERVABLE_NAMES,
            "sample_repeat": calibration_module._MOMENT_OBSERVABLE_NAMES,
        },
    )
    np.testing.assert_array_equal(
        reproduced.frame(2).counts_yx,
        product.frame(2).counts_yx,
    )
    np.testing.assert_allclose(
        reproduced.dataset.blocks[0].covariance,
        product.dataset.blocks[0].covariance,
    )


def test_checkpoint_observable_builder_rejects_incomplete_or_incompatible(tmp_path):
    detector, plan, requests, directory = _observable_checkpoint_fixture(
        tmp_path,
        max_new_frames=1,
    )
    kwargs = dict(
        render_metadata={"truth": "fixed"},
        storage_dtype="float32",
        default_noise_model=DetectorNoiseModel(1.0e4, 576.0),
        bootstrap_replicates=4,
    )
    with pytest.raises(IncompleteCheckpointError, match="frame record"):
        build_heterogeneous_dataset_from_checkpoint(
            plan,
            requests,
            detector,
            directory,
            render_signature="observable-test:v1",
            **kwargs,
        )
    with pytest.raises(CheckpointCompatibilityError, match="identity differ"):
        build_heterogeneous_dataset_from_checkpoint(
            plan,
            requests,
            detector,
            directory,
            render_signature="wrong-renderer:v9",
            **kwargs,
        )


def test_checkpoint_observable_builder_supports_measurement_extractors(tmp_path):
    detector, plan, requests, directory = _observable_checkpoint_fixture(tmp_path)

    def certified_boundary_extractor(counts, request, noise_model):
        corrected = np.asarray(counts) - noise_model.dark_offset_e
        return (
            np.asarray((np.sum(corrected), np.mean(corrected[:, :9]))),
            ("transmitted_electrons", f"{request.measurement}_left_region_mean_e"),
        )

    product = build_heterogeneous_dataset_from_checkpoint(
        plan,
        requests,
        detector,
        directory,
        render_signature="observable-test:v1",
        render_metadata={"truth": "fixed"},
        storage_dtype="float32",
        default_noise_model=DetectorNoiseModel(2.0e4, 576.0),
        bootstrap_replicates=4,
        observable_extractor_by_measurement={
            "sample": certified_boundary_extractor,
            "sample_repeat": certified_boundary_extractor,
        },
    )
    assert product.dataset.n_observations == 4
    assert product.dataset.blocks[0].observable_names == (
        "transmitted_electrons",
        "sample_left_region_mean_e",
    )
    extractor_name = product.dataset.metadata["frame_metadata"]["2"][
        "observable_extractor"
    ]
    assert extractor_name.endswith(".<locals>.certified_boundary_extractor")


def _joint_aggregation_product(tmp_path):
    plan = AcquisitionPlan(
        channel_names=("IL1",),
        states=(
            AcquisitionState("nominal", (0.2,), "nominal", 7),
            AcquisitionState("held", (0.3,), "held_out", 2),
        ),
        reference_dacs=(0.2,),
    )

    def request(frame_id, state_index, measurement, approach="forward"):
        state = plan.states[state_index]
        return FrameRequest(
            frame_id=frame_id,
            optical_state_index=state_index,
            optical_state_label=state.label,
            measurement=measurement,
            sample_state=SamplePlaneState(label=measurement),
            raw_dac_values=state.dac_values,
            dac_approach=approach,
            interaction_route=(
                "local_action"
                if measurement in {"sample", "sample_repeat"}
                else "vacuum"
            ),
            frozen_topology_id=(
                "atlas:L0:test:0123456789abcdef"
                if measurement in {"sample", "sample_repeat"}
                else None
            ),
        )

    requests = (
        request(0, 0, "perforation"),
        request(1, 0, "position_basis"),
        request(2, 0, "angle_basis"),
        request(3, 0, "sample"),
        request(4, 0, "sample_repeat"),
        request(5, 0, "perforation", "reverse"),
        request(6, 0, "position_basis", "reverse"),
        request(10, 1, "perforation"),
        request(11, 1, "position_basis"),
    )
    specifications = {
        0: ((1.0, 2.0), ("centroid_x_px", "centroid_y_px"), (1.0, 2.0)),
        1: ((4.0, 7.0), ("centroid_x_px", "centroid_y_px"), (3.0, 4.0)),
        2: ((0.0, 8.0), ("centroid_x_px", "centroid_y_px"), (5.0, 6.0)),
        3: (
            (999.0, 10.0, 20.0),
            ("total_counts_dark_corrected", "fourier_g_y_px", "fourier_g_x_px"),
            (100.0, 1.0, 1.0),
        ),
        4: (
            (888.0, 14.0, 24.0),
            ("total_counts_dark_corrected", "fourier_g_y_px", "fourier_g_x_px"),
            (100.0, 3.0, 3.0),
        ),
        5: ((100.0, 100.0), ("centroid_x_px", "centroid_y_px"), (1.0, 1.0)),
        6: ((150.0, 160.0), ("centroid_x_px", "centroid_y_px"), (1.0, 1.0)),
        10: ((2.0, 3.0), ("centroid_x_px", "centroid_y_px"), (0.5, 0.5)),
        11: ((3.0, 5.0), ("centroid_x_px", "centroid_y_px"), (0.7, 0.8)),
    }
    blocks = tuple(
        FrameObservationBlock(
            frame_id=frame_id,
            optical_state_index=next(
                item.optical_state_index for item in requests if item.frame_id == frame_id
            ),
            values=np.asarray(values),
            covariance=np.diag(variances),
            observable_names=names,
        )
        for frame_id, (values, names, variances) in specifications.items()
    )
    feature_metadata = {
        "certificate_id": "certificate-v1",
        "labels": ["g"],
        "local_reciprocal_vectors_xy_per_m": [[1e8, 0.0]],
        "world_reciprocal_vectors_xy_per_m": [[1e8, 0.0]],
        "covariance_world_per_m2": [[1e8, 0.0], [0.0, 1e8]],
    }
    frame_metadata = {
        str(item.frame_id): {
            "known_fourier_features": (
                feature_metadata if item.measurement in {"sample", "sample_repeat"} else None
            )
        }
        for item in requests
    }
    dataset = HeterogeneousInversionDataset(
        plan,
        blocks,
        metadata={"frame_metadata": frame_metadata},
    )
    noise = DetectorNoiseModel(100.0)
    noisy_frames = tuple(
        NoisyDetectorFrame(item, np.ones((2, 2)), item.frame_id + 100, noise)
        for item in requests
    )
    acquisition = CheckpointedAcquisition(
        checkpoint_directory=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        manifest_sha256="fixture",
        records=(),
        pending_frame_ids=(),
    )
    return CheckpointObservableDataset(acquisition, dataset, noisy_frames, 4)


def test_joint_aggregation_uses_shared_blank_covariance_and_forward_only(tmp_path):
    product = _joint_aggregation_product(tmp_path)
    aggregated = aggregate_checkpoint_observables_for_fit(
        product,
        material_repeat_policy="pool",
    )
    assert [block.optical_state_index for block in aggregated.blocks] == [0, 1]
    nominal = aggregated.blocks[0]
    np.testing.assert_allclose(nominal.values[:4], [3.0, 5.0, -1.0, 6.0])
    np.testing.assert_allclose(nominal.covariance[:2, :2], np.diag([4.0, 6.0]))
    np.testing.assert_allclose(nominal.covariance[2:4, 2:4], np.diag([6.0, 8.0]))
    np.testing.assert_allclose(nominal.covariance[:2, 2:4], np.diag([1.0, 2.0]))
    np.testing.assert_allclose(nominal.covariance[2:4, :2], np.diag([1.0, 2.0]))
    np.testing.assert_allclose(nominal.values[-2:], [11.0, 21.0])
    np.testing.assert_allclose(nominal.covariance[-2:, -2:], np.eye(2) * 0.75)
    assert all("total_counts" not in name for name in nominal.observable_names)

    aggregation = aggregated.metadata["aggregation"]
    assert aggregation["approach_policy"] == "forward_only"
    assert aggregation["excluded_reverse_frame_ids"] == [5, 6]
    state_metadata = aggregation["state_blocks"]
    assert state_metadata["0"]["baseline_frame_id"] == 0
    assert state_metadata["0"]["response_frame_ids"] == [1, 2]
    assert state_metadata["0"]["material"][0]["member_frame_ids"] == [3, 4]
    assert state_metadata["0"]["material"][0]["known_fourier_features"][
        "certificate_id"
    ] == "certificate-v1"
    assert state_metadata["10"]["held_out"] is True
    assert aggregated.plan.states[aggregated.blocks[1].optical_state_index].role == "held_out"


def test_joint_aggregation_repeat_policies_are_exact(tmp_path):
    product = _joint_aggregation_product(tmp_path)
    retained = aggregate_checkpoint_observables_for_fit(
        product,
        material_repeat_policy="retain",
    )
    primary = aggregate_checkpoint_observables_for_fit(
        product,
        material_repeat_policy="primary_only",
    )
    retained_names = retained.blocks[0].observable_names
    primary_names = primary.blocks[0].observable_names
    assert any(name.startswith("material_frame_3") for name in retained_names)
    assert any(name.startswith("material_frame_4") for name in retained_names)
    assert any(name.startswith("material_frame_3") for name in primary_names)
    assert not any(name.startswith("material_frame_4") for name in primary_names)


@pytest.mark.parametrize(
    ("family", "expected_vectors"),
    (("multilayer_period", 1), ("silicon_lattice_vectors", 2)),
)
def test_known_fourier_features_follow_certificate_and_sample_pose(
    family, expected_vectors
):
    certificate = SimpleNamespace(
        version="certificate-v3",
        multilayer_period_m=23e-9,
        silicon_layer_width_uncertainty_m=0.2e-9,
        silicon_germanium_layer_width_uncertainty_m=0.3e-9,
        silicon_lattice_metadata={"spacing_m": 0.31356e-9},
        silicon_lattice_spacing_uncertainty_m=0.00005e-9,
    )
    theta = 0.23
    projection = np.asarray(
        ((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta)))
    )
    features = known_fourier_features_from_sample_view(
        SimpleNamespace(certificate=certificate),
        SimpleNamespace(
            selection=SimpleNamespace(feature_family=family),
            projection_world_from_local=projection,
        ),
    )
    assert features is not None
    assert len(features.labels) == expected_vectors
    expected_world = np.linalg.inv(projection).T @ (
        features.local_reciprocal_vectors_xy_per_m.T
    )
    np.testing.assert_allclose(
        features.world_reciprocal_vectors_xy_per_m,
        expected_world.T,
    )
    assert features.certificate_id == "certificate-v3"
    eigenvalues = np.linalg.eigvalsh(features.covariance_world_per_m2)
    assert eigenvalues.min() >= -1e-10 * np.max(np.abs(eigenvalues))


def test_fourier_affine_recovers_known_scale_rotation_and_requires_two_axes():
    reference = np.asarray(((1.0, 0.0), (0.0, 2.0), (1.0, 1.0)))
    angle = 0.27
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    matrix = rotation @ np.diag((3.0, 2.0))
    measured = reference @ matrix.T
    calibration = fit_fourier_affine(reference, measured)
    np.testing.assert_allclose(calibration.matrix, matrix, atol=1e-12)
    assert calibration.rotation_rad == pytest.approx(angle)
    assert calibration.isotropic_scale == pytest.approx(np.sqrt(6.0))
    assert calibration.rms_residual < 1e-12
    with pytest.raises(ValueError, match="non-collinear"):
        fit_fourier_affine(reference[:1], measured[:1])


def test_held_out_acceptance_metrics_apply_requested_limits():
    accepted = evaluate_held_out_metrics(
        predicted_magnification=100400.0,
        reference_magnification=100000.0,
        predicted_rotation_deg=12.4,
        reference_rotation_deg=12.0,
        predicted_centroid_xy_px=(127.7, 128.1),
        reference_centroid_xy_px=(128.0, 128.0),
        predicted_beam_diameter_px=20.3,
        reference_beam_diameter_px=20.0,
        predicted_diffraction_vectors_yx_px=np.asarray(((-4.1, -1.9), (-3.2, -8.1))),
        reference_diffraction_vectors_yx_px=np.asarray(((3.0, 8.0), (4.0, 2.0))),
    )
    assert accepted.passed
    rejected = evaluate_held_out_metrics(
        predicted_magnification=100600.0,
        reference_magnification=100000.0,
        predicted_rotation_deg=12.0,
        reference_rotation_deg=12.0,
        predicted_centroid_xy_px=(128.0, 128.0),
        reference_centroid_xy_px=(128.0, 128.0),
        predicted_beam_diameter_px=20.0,
        reference_beam_diameter_px=20.0,
        predicted_diffraction_vectors_yx_px=np.empty((0, 2)),
        reference_diffraction_vectors_yx_px=np.empty((0, 2)),
    )
    assert not rejected.passed


def test_held_out_diffraction_gate_fails_closed_when_vectors_are_required():
    shared = dict(
        predicted_magnification=100000.0,
        reference_magnification=100000.0,
        predicted_rotation_deg=12.0,
        reference_rotation_deg=12.0,
        predicted_centroid_xy_px=(128.0, 128.0),
        reference_centroid_xy_px=(128.0, 128.0),
        predicted_beam_diameter_px=20.0,
        reference_beam_diameter_px=20.0,
    )
    optional = evaluate_held_out_metrics(
        **shared,
        predicted_diffraction_vectors_yx_px=np.empty((0, 2)),
        reference_diffraction_vectors_yx_px=np.empty((0, 2)),
    )
    assert optional.passed

    missing_both = evaluate_held_out_metrics(
        **shared,
        predicted_diffraction_vectors_yx_px=np.empty((0, 2)),
        reference_diffraction_vectors_yx_px=np.empty((0, 2)),
        require_diffraction_vectors=True,
    )
    assert not missing_both.passed
    assert np.isinf(missing_both.diffraction_vector_error_px)

    missing_prediction = evaluate_held_out_metrics(
        **shared,
        predicted_diffraction_vectors_yx_px=np.empty((0, 2)),
        reference_diffraction_vectors_yx_px=np.asarray(((3.0, 8.0),)),
        require_diffraction_vectors=True,
    )
    assert not missing_prediction.passed
    assert np.isinf(missing_prediction.diffraction_vector_error_px)
