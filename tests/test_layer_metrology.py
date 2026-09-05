from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import ndimage

from temgym_core.layer_metrology import (
    CertifiedLayerGeometry2D,
    LayerBoundaryRegistration2D,
    RegisteredLayerMetrology,
    RegisteredLayerPeriodMetrology,
    UnresolvedLayerBoundaryError,
    derive_certified_layer_geometry,
    extract_registered_layer_boundaries,
    freeze_layer_boundary_registration,
    layer_observable_covariance_augmentation,
    predict_registered_layer_observables,
    predict_registered_layer_period_observables,
    registered_layer_reference_residuals_px,
    registered_layer_period_metrology_from_boundary_metrology,
)
from temgym_core.microscope_inversion import (
    AcquisitionPlan,
    AcquisitionState,
    FrameObservationBlock,
    HeterogeneousInversionDataset,
    SamplePlaneState,
)
from temgym_core.multiscale_sample import MAGICALSampleAtlas, select_sample_view
from temgym_core.neoarm_calibration import (
    CheckpointObservableDataset,
    CheckpointedAcquisition,
    DetectorGeometry,
    DetectorNoiseModel,
    FrameRequest,
    FramePrediction,
    GaussianRenderResult,
    NoisyDetectorFrame,
    HardEdgeColumnParameterization,
    HardEdgeLensInitial,
    aggregate_checkpoint_observables_for_fit,
    build_heterogeneous_dataset_from_checkpoint,
    extract_image_observables,
    render_frame_sequence_checkpointed,
    simulate_detector_counts,
)
from temgym_core.physical_observables import (
    ImagingLayerBoundaries,
    ImagingLayerPeriods,
    PhysicalObservableBlockSpecification,
    predict_physical_observable_blocks,
    specifications_from_aggregated_dataset,
)


jax.config.update("jax_enable_x64", True)


def _rotation(theta):
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.asarray(((cosine, -sine), (sine, cosine)))


def _manual_geometry(*, translation=(3e-9, -2e-9), projection=None):
    x = np.asarray((-25.0, -15.0, -2.0, 8.0, 21.0, 31.0)) * 1e-9
    count = len(x)
    if projection is None:
        projection = _rotation(0.19) @ np.diag((np.cos(0.08), np.cos(0.11)))
    return CertifiedLayerGeometry2D(
        certificate_id="certificate-test-v4",
        certificate_source="traceable synthetic test certificate",
        region_name="multilayer_region_4",
        boundary_indices=tuple(range(3, 3 + count)),
        boundary_labels=tuple(f"b{index:02d}" for index in range(3, 3 + count)),
        boundary_points_local_xy_m=np.column_stack((x, np.zeros(count))),
        boundary_line_endpoints_local_xy_m=np.stack(
            (
                np.column_stack((x, np.full(count, -42e-9))),
                np.column_stack((x, np.full(count, 42e-9))),
            ),
            axis=1,
        ),
        boundary_standard_uncertainties_m=np.linspace(0.1e-9, 0.3e-9, count),
        interval_pairs=np.column_stack((np.arange(count - 1), np.arange(1, count))),
        interval_labels=tuple(f"l{index:02d}" for index in range(4, 3 + count)),
        interval_materials=tuple(
            "si" if index % 2 == 0 else "sige"
            for index in range(4, 3 + count)
        ),
        sample_translation_xy_m=np.asarray(translation),
        sample_projection_world_from_local=np.asarray(projection),
    )


def _synthetic_layer_image(
    geometry,
    h,
    shape_yx,
    pixel_size_yx_m,
    *,
    interface_width_m=0.7e-9,
    boundary_offsets_m=None,
):
    ny, nx = shape_yx
    pixel_xy = np.asarray(pixel_size_yx_m)[::-1]
    x_px, y_px = np.meshgrid(
        np.arange(nx) - 0.5 * (nx - 1),
        np.arange(ny) - 0.5 * (ny - 1),
        indexing="xy",
    )
    detector_xy_m = np.stack((x_px * pixel_xy[0], y_px * pixel_xy[1]), axis=-1)
    world_xy_m = np.einsum("ij,...j->...i", np.linalg.inv(h), detector_xy_m)
    local_xy_m = np.einsum(
        "ij,...j->...i",
        np.linalg.inv(geometry.sample_projection_world_from_local),
        world_xy_m - geometry.sample_translation_xy_m,
    )
    local_x = local_xy_m[..., 0]
    local_y = local_xy_m[..., 1]
    image = np.full(shape_yx, 4200.0)
    offsets = (
        np.zeros(geometry.n_boundaries)
        if boundary_offsets_m is None
        else np.asarray(boundary_offsets_m, dtype=float)
    )
    if offsets.shape != (geometry.n_boundaries,):
        raise ValueError("boundary_offsets_m has the wrong shape")
    for index, boundary_x in enumerate(geometry.boundary_points_local_xy_m[:, 0]):
        sign = 1.0 if index % 2 == 0 else -1.0
        image += sign * 420.0 * np.tanh(
            (local_x - boundary_x - offsets[index]) / interface_width_m
        )
    envelope = np.exp(-((local_y / 39e-9) ** 12))
    image = 4200.0 + (image - 4200.0) * envelope
    return ndimage.gaussian_filter(image, 0.45)


def _registered_fixture(*, mode="interval_widths", minimum_peak_snr=2.0):
    geometry = _manual_geometry()
    shape = (196, 238)
    pixel_yx = (70e-6, 55e-6)
    h = _rotation(0.31) @ np.asarray(((153_000.0, 11_000.0), (0.0, 132_000.0)))
    image = _synthetic_layer_image(geometry, h, shape, pixel_yx)
    registration = freeze_layer_boundary_registration(
        image,
        geometry,
        detector_position_map_xy_m_per_m=h,
        detector_pixel_size_yx_m=pixel_yx,
        smoothing_sigma_px=0.7,
        minimum_peak_snr=minimum_peak_snr,
    )
    return geometry, h, shape, pixel_yx, image, RegisteredLayerMetrology(
        geometry, registration, mode
    )


def test_geometry_derives_visible_registered_certificate_intervals_from_real_atlas():
    atlas = MAGICALSampleAtlas.nominal(
        shape_yx=(32, 32),
        region_shape_yx=(64, 64),
        authoritative_multilayer_shape_yx=(128, 128),
        max_levels=1,
        pose_translation_xy_m=(2.0e-9, -1.0e-9),
        pose_rotation_rad=0.23,
        pose_tilt_xy_rad=(0.07, -0.04),
    )
    view = select_sample_view(
        atlas,
        transfer_jacobian_xy=200_000.0 * _rotation(-0.17),
        detector_shape_yx=(256, 256),
        detector_pixel_size_yx_m=(55e-6, 55e-6),
        gaussian_radius_min_m=2e-9,
    )
    assert view.selection.feature_family == "individual_layer_widths"
    geometry = derive_certified_layer_geometry(atlas, view)

    assert geometry.n_intervals >= 2
    assert geometry.boundary_uncertainty_model == "cumulative_layer_widths"
    assert all(
        right == left + 1
        for left, right in zip(
            geometry.boundary_indices, geometry.boundary_indices[1:]
        )
    )
    expected_widths = np.asarray(
        [10e-9 if int(label[1:3]) % 2 else 13e-9 for label in geometry.interval_labels]
    )
    np.testing.assert_allclose(geometry.interval_widths_local_m, expected_widths)
    expected_world = np.asarray(view.centre_xy_m) + np.einsum(
        "ij,nj->ni",
        np.asarray(view.projection_world_from_local),
        geometry.boundary_points_local_xy_m,
    )
    np.testing.assert_allclose(geometry.boundary_points_world_xy_m, expected_world)
    round_trip = CertifiedLayerGeometry2D.from_metadata(geometry.to_metadata())
    np.testing.assert_array_equal(round_trip.boundary_indices, geometry.boundary_indices)
    np.testing.assert_allclose(
        round_trip.boundary_points_local_xy_m,
        geometry.boundary_points_local_xy_m,
    )
    assert not round_trip.boundary_points_local_xy_m.flags.writeable


def test_noisy_rotated_anisotropic_stripes_recover_subpixel_boundaries_and_widths():
    geometry, h, _shape, pixel_yx, image, metrology = _registered_fixture(
        mode="boundary_positions_and_interval_widths"
    )
    rng = np.random.default_rng(14)
    noisy = rng.poisson(image).astype(float) + rng.normal(0.0, 1.2, image.shape)
    extraction = metrology.extract(noisy)
    predicted = np.asarray(
        predict_registered_layer_observables(h, pixel_yx, metrology)
    )
    measured = metrology.values(extraction)

    assert measured.shape == predicted.shape
    np.testing.assert_allclose(
        extraction.boundary_positions_axis_px,
        predicted[: geometry.n_boundaries],
        atol=0.25,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        extraction.interval_widths_px,
        predicted[geometry.n_boundaries :],
        atol=0.35,
        rtol=0.0,
    )
    assert np.min(extraction.peak_snr) > 2.0
    assert metrology.observable_names[:2] == (
        "layer_boundary_b03_axis_px",
        "layer_boundary_b04_axis_px",
    )
    assert metrology.observable_names[-1] == "layer_interval_l08_width_px"

    # Search centres are reference-only association metadata.  They do not
    # enter the forward prediction.
    shifted_registration = replace(
        metrology.registration,
        search_centres_axis_px=metrology.registration.search_centres_axis_px + 0.03,
    )
    shifted = RegisteredLayerMetrology(
        geometry, shifted_registration, metrology.observable_mode
    )
    np.testing.assert_array_equal(
        predict_registered_layer_observables(h, pixel_yx, shifted), predicted
    )
    assert (
        metrology.registration.to_metadata()["reference_role"]
        == "peak_identity_order_sign_and_window_only"
    )


def test_same_transition_periods_survive_opposite_ctf_shoulder_shifts():
    geometry = _manual_geometry()
    shape = (196, 238)
    pixel_yx = (70e-6, 55e-6)
    h = _rotation(0.31) @ np.asarray(
        ((153_000.0, 11_000.0), (0.0, 132_000.0))
    )
    # A defocused contrast transfer can displace opposite transition classes
    # in opposite directions.  The displacement is large enough to fail the
    # ordinary 0.35 discovery gate, but remains well inside each disjoint
    # identity window.
    shifts_m = 1.55e-9 * np.where(np.arange(geometry.n_boundaries) % 2, -1.0, 1.0)
    image = _synthetic_layer_image(
        geometry,
        h,
        shape,
        pixel_yx,
        boundary_offsets_m=shifts_m,
    )
    with pytest.raises(UnresolvedLayerBoundaryError, match="offset"):
        freeze_layer_boundary_registration(
            image,
            geometry,
            detector_position_map_xy_m_per_m=h,
            detector_pixel_size_yx_m=pixel_yx,
            smoothing_sigma_px=0.7,
            minimum_peak_snr=2.0,
            minimum_step_improvement=0.02,
        )

    registration = freeze_layer_boundary_registration(
        image,
        geometry,
        detector_position_map_xy_m_per_m=h,
        detector_pixel_size_yx_m=pixel_yx,
        smoothing_sigma_px=0.7,
        minimum_peak_snr=2.0,
        minimum_step_improvement=0.02,
        maximum_reference_offset_fraction=0.499,
    )
    boundary = RegisteredLayerMetrology(
        geometry, registration, "boundary_positions"
    )
    period = registered_layer_period_metrology_from_boundary_metrology(boundary)
    extraction = period.extract(image)
    measured = period.values(extraction)
    predicted = np.asarray(
        predict_registered_layer_period_observables(h, pixel_yx, period)
    )

    assert isinstance(period, RegisteredLayerPeriodMetrology)
    assert period.span_pairs.shape == (geometry.n_boundaries - 2, 2)
    assert np.min(extraction.peak_snr) > 2.0
    assert np.min(extraction.step_model_improvement) > 0.02
    np.testing.assert_allclose(measured, predicted, atol=0.25, rtol=0.0)
    restored = RegisteredLayerPeriodMetrology.from_metadata(period.to_metadata())
    assert restored.observable_names == period.observable_names
    np.testing.assert_allclose(
        restored.span_certificate_covariance_local_m2,
        period.span_certificate_covariance_local_m2,
    )


def test_same_transition_period_selection_refuses_pruned_or_wrong_polarity_triplets():
    geometry, _h, _shape, _pixel_yx, _image, metrology = _registered_fixture(
        mode="boundary_positions"
    )
    polarity = np.asarray(metrology.registration.edge_polarities).copy()
    polarity[2] = polarity[1]
    damaged = RegisteredLayerMetrology(
        geometry,
        replace(metrology.registration, edge_polarities=polarity),
        "boundary_positions",
    )
    periods = registered_layer_period_metrology_from_boundary_metrology(
        damaged,
        minimum_period_spans=1,
    )
    assert (0, 2) not in map(tuple, periods.span_pairs)
    assert (1, 3) not in map(tuple, periods.span_pairs)
    with pytest.raises(UnresolvedLayerBoundaryError, match="at least"):
        registered_layer_period_metrology_from_boundary_metrology(
            damaged,
            minimum_period_spans=geometry.n_boundaries,
        )


def test_registration_schema_is_invariant_across_bootstrap_noise_and_round_trip():
    _geometry, _h, _shape, _pixel_yx, image, metrology = _registered_fixture(
        minimum_peak_snr=1.0
    )
    noise_model = SimpleNamespace(dark_offset_e=3.5)
    schemas = []
    values = []
    rng = np.random.default_rng(44)
    for _ in range(12):
        counts = rng.poisson(image + noise_model.dark_offset_e)
        value, schema = metrology(counts, object(), noise_model)
        schemas.append(schema)
        values.append(value)
    assert all(schema == metrology.observable_names for schema in schemas)
    covariance = np.cov(np.asarray(values), rowvar=False)
    assert covariance.shape == (metrology.geometry.n_intervals,) * 2
    assert np.all(np.isfinite(covariance))

    restored = RegisteredLayerMetrology.from_metadata(metrology.to_metadata())
    assert restored.observable_names == metrology.observable_names
    np.testing.assert_allclose(
        restored.registration.detector_normal_xy,
        metrology.registration.detector_normal_xy,
    )
    bad = dict(metrology.to_metadata())
    bad["observable_names"] = ["changed"]
    with pytest.raises(ValueError, match="observable schema"):
        RegisteredLayerMetrology.from_metadata(bad)


def test_reference_discovery_gate_does_not_censor_noisy_draws():
    geometry, _h, _shape, _pixel_yx, image, metrology = _registered_fixture(
        minimum_peak_snr=2.0
    )
    low_dose_draws = []
    for seed in range(24):
        counts = simulate_detector_counts(
            image,
            dose_electrons=2.0e5,
            incident_intensity_sum=float(np.sum(image)),
            read_noise_e=0.5,
            dark_offset_e=0.0,
            seed=seed,
        )
        low_dose_draws.append(metrology.extract(counts))
    assert any(np.min(result.peak_snr) < 2.0 for result in low_dose_draws)
    assert all(np.all(np.isfinite(metrology.values(result))) for result in low_dose_draws)

    weakest_index = int(
        np.argmin([np.min(result.peak_snr) for result in low_dose_draws])
    )
    policy_registration = replace(
        metrology.registration,
        extraction_rejection_minimum_peak_snr=2.0,
        extraction_rejection_minimum_step_improvement=0.02,
    )
    policy_metrology = RegisteredLayerMetrology(
        geometry, policy_registration, metrology.observable_mode
    )
    weak_counts = simulate_detector_counts(
        image,
        dose_electrons=2.0e5,
        incident_intensity_sum=float(np.sum(image)),
        read_noise_e=0.5,
        dark_offset_e=0.0,
        seed=weakest_index,
    )
    policy_metrology.extract(weak_counts)
    with pytest.raises(ValueError, match="extraction-rejection|polarity"):
        policy_metrology.extract(weak_counts, enforce_quality_policy=True)

    metadata = policy_registration.to_metadata()
    assert metadata["reference_discovery_gates"]["minimum_peak_snr"] == 2.0
    assert metadata["extraction_rejection_policy"]["minimum_peak_snr"] == 2.0
    assert metadata["extraction_rejection_policy"]["enforcement"] == (
        "explicit_opt_in_only"
    )


def test_certificate_and_validated_model_covariance_are_schema_safe():
    geometry, _h, _shape, _pixel_yx, _image, metrology = _registered_fixture()
    residuals = registered_layer_reference_residuals_px(metrology)
    augmentation = layer_observable_covariance_augmentation(metrology)
    assert augmentation.observable_names == metrology.observable_names
    np.testing.assert_allclose(augmentation.reference_residuals_px, residuals)
    assert augmentation.total_covariance_px2.shape == (
        geometry.n_intervals,
        geometry.n_intervals,
    )
    assert np.linalg.eigvalsh(augmentation.certificate_covariance_px2).min() >= -1e-12

    axis_scale = np.median(
        np.diff(metrology.registration.search_centres_axis_px)
        / np.diff(geometry.boundary_points_local_xy_m[:, 0])
    )
    expected_sigma = axis_scale * geometry.interval_standard_uncertainties_m
    np.testing.assert_allclose(
        np.sqrt(np.diag(augmentation.certificate_covariance_px2)),
        expected_sigma,
        rtol=1e-12,
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="provenance"):
        layer_observable_covariance_augmentation(
            metrology,
            validated_model_discrepancy_standard_uncertainty_px=0.2,
        )
    validated = layer_observable_covariance_augmentation(
        metrology,
        validated_model_discrepancy_standard_uncertainty_px=np.maximum(
            np.abs(residuals), 0.1
        ),
        model_discrepancy_provenance="independent matched-grid validation",
    )
    assert np.all(
        np.diag(validated.total_covariance_px2)
        > np.diag(augmentation.total_covariance_px2)
    )
    assert validated.to_metadata()["shared_across_same_specimen_repeats"] is True


def test_first_order_prediction_has_exact_xy_yx_units_and_pose_derivatives():
    geometry, h, _shape, pixel_yx, _image, metrology = _registered_fixture(
        mode="boundary_positions_and_interval_widths"
    )
    normal = metrology.registration.detector_normal_xy
    pixel_xy = np.asarray(pixel_yx)[::-1]
    world = geometry.sample_translation_xy_m + np.einsum(
        "ij,nj->ni",
        geometry.sample_projection_world_from_local,
        geometry.boundary_points_local_xy_m,
    )
    detector_px = (h @ world.T).T / pixel_xy
    offset = (
        metrology.registration.detector_optical_axis_xy_px
        - metrology.registration.coordinate_origin_xy_px
    )
    positions = (detector_px + offset) @ normal
    pairs = geometry.interval_pairs
    expected = np.concatenate(
        (positions, positions[pairs[:, 1]] - positions[pairs[:, 0]])
    )
    predicted = predict_registered_layer_observables(h, pixel_yx, metrology)
    np.testing.assert_allclose(predicted, expected, rtol=1e-13, atol=1e-13)

    def prediction(matrix, translation, projection):
        return predict_registered_layer_observables(
            matrix,
            pixel_yx,
            metrology,
            sample_translation_xy_m=translation,
            sample_projection_world_from_local=projection,
        )

    jac_h, jac_translation, jac_projection = jax.jacfwd(
        prediction, argnums=(0, 1, 2)
    )(
        jnp.asarray(h),
        jnp.asarray(geometry.sample_translation_xy_m),
        jnp.asarray(geometry.sample_projection_world_from_local),
    )
    assert jac_h.shape == (len(expected), 2, 2)
    assert jac_translation.shape == (len(expected), 2)
    assert jac_projection.shape == (len(expected), 2, 2)
    assert np.all(np.isfinite(jac_h))
    assert np.linalg.norm(jac_h) > 0.0
    assert np.linalg.norm(jac_projection) > 0.0
    # Translation cancels from interval widths but not absolute boundaries.
    assert np.linalg.norm(jac_translation[: geometry.n_boundaries]) > 0.0
    np.testing.assert_allclose(jac_translation[geometry.n_boundaries :], 0.0, atol=1e-12)


def _column():
    return HardEdgeColumnParameterization(
        lenses=(
            HardEdgeLensInitial(
                "IL1",
                z_center_m=0.42,
                length_m=0.08,
                q_bias=0.62,
                q_slope_per_dac=0.17,
                channel_index=0,
            ),
            HardEdgeLensInitial(
                "PL1",
                z_center_m=0.74,
                length_m=0.08,
                q_bias=0.51,
                q_slope_per_dac=-0.13,
                channel_index=1,
            ),
        ),
        sample_z_m=0.2,
        source_z_m=0.0,
        detector_z_m=1.0,
        channel_names=("IL1", "PL1"),
    )


def _request(frame_id, measurement, *, dacs=(0.25, -0.35), feature="auto"):
    return FrameRequest(
        frame_id=frame_id,
        optical_state_index=0,
        optical_state_label="state",
        measurement=measurement,
        sample_state=SamplePlaneState(label=measurement),
        raw_dac_values=dacs,
        feature_family=feature,
        interaction_route=(
            "local_action" if measurement in {"sample", "sample_repeat"} else "vacuum"
        ),
        frozen_topology_id=(
            "layer:L0:test:0123456789abcdef"
            if measurement in {"sample", "sample_repeat"}
            else None
        ),
    )


def test_physical_layer_part_uses_h_and_differentiates_through_hard_edge_column():
    geometry, _h, shape, pixel_yx, _image, metrology = _registered_fixture()
    detector = DetectorGeometry(
        shape_yx=shape,
        pixel_size_yx_m=pixel_yx,
        z_m=1.0,
        affine_xy=_rotation(-0.21) @ np.asarray(((1.03, 0.04), (0.0, 0.97))),
    )
    column = _column()
    request = _request(2, "sample", feature="individual_layer_widths")
    part = ImagingLayerBoundaries(request, metrology)
    specification = PhysicalObservableBlockSpecification(2, 0, (part,))
    predicted = np.asarray(
        predict_physical_observable_blocks(
            column, column.nominal_parameters, detector, (specification,)
        )[0]
    )
    transfer = np.asarray(
        column.matrix(
            column.nominal_parameters,
            request.raw_dac_values,
            segment="sample_to_detector",
        )
    )
    h = detector.affine_xy @ transfer[np.ix_((0, 2), (0, 2))]
    expected = np.asarray(
        predict_registered_layer_observables(h, pixel_yx, metrology)
    )
    np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-12)

    def forward(parameters):
        return predict_physical_observable_blocks(
            column, parameters, detector, (specification,)
        )[0]

    jacobian = np.asarray(jax.jacfwd(forward)(jnp.asarray(column.nominal_parameters)))
    assert jacobian.shape == (geometry.n_intervals, column.n_parameters)
    assert np.all(np.isfinite(jacobian))
    assert np.linalg.norm(jacobian) > 0.0
    assert np.linalg.norm(jacobian[:, column.parameter_names.index("IL1.q_bias")]) > 0.0
    assert np.linalg.norm(jacobian[:, column.parameter_names.index("PL1.q_bias")]) > 0.0


def test_physical_layer_period_part_uses_h_and_is_jittable():
    geometry, _h, shape, pixel_yx, _image, boundary = _registered_fixture(
        mode="boundary_positions"
    )
    period = registered_layer_period_metrology_from_boundary_metrology(boundary)
    detector = DetectorGeometry(
        shape_yx=shape,
        pixel_size_yx_m=pixel_yx,
        z_m=1.0,
        affine_xy=_rotation(-0.21) @ np.asarray(((1.03, 0.04), (0.0, 0.97))),
    )
    column = _column()
    request = _request(2, "sample", feature="multilayer_period")
    part = ImagingLayerPeriods(request, period)
    specification = PhysicalObservableBlockSpecification(2, 0, (part,))

    transfer = np.asarray(
        column.matrix(
            column.nominal_parameters,
            request.raw_dac_values,
            segment="sample_to_detector",
        )
    )
    h = detector.affine_xy @ transfer[np.ix_((0, 2), (0, 2))]
    expected = np.asarray(
        predict_registered_layer_period_observables(h, pixel_yx, period)
    )

    def forward(parameters):
        return predict_physical_observable_blocks(
            column, parameters, detector, (specification,)
        )[0]

    predicted = np.asarray(jax.jit(forward)(jnp.asarray(column.nominal_parameters)))
    np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-12)
    assert specification.observable_names == period.observable_names
    jacobian = np.asarray(jax.jacfwd(forward)(jnp.asarray(column.nominal_parameters)))
    assert jacobian.shape == (len(period.span_pairs), column.n_parameters)
    assert np.all(np.isfinite(jacobian))
    assert np.linalg.norm(jacobian) > 0.0
    assert geometry.n_boundaries > len(period.span_pairs)


def test_aggregated_specification_rebuilds_explicit_layer_period_schema():
    _geometry, _h, shape, pixel_yx, _image, boundary = _registered_fixture(
        mode="boundary_positions"
    )
    period = registered_layer_period_metrology_from_boundary_metrology(boundary)
    plan = AcquisitionPlan(
        channel_names=("IL1", "PL1"),
        states=(AcquisitionState("state", (0.25, -0.35), "held_out", 2),),
        reference_dacs=(0.25, -0.35),
    )
    baseline = _request(0, "perforation")
    material = _request(1, "sample", feature="multilayer_period")
    names = tuple(f"material_frame_1.{name}" for name in period.observable_names)
    block = FrameObservationBlock(0, 0, np.zeros(len(names)), np.eye(len(names)), names)
    dataset = HeterogeneousInversionDataset(
        plan,
        (block,),
        metadata={
            "aggregation": {
                "state_blocks": {
                    "0": {
                        "baseline_frame_id": 0,
                        "response_frame_ids": [],
                        "material": [
                            {
                                "mode": "retained",
                                "member_frame_ids": [1],
                                "known_fourier_features": None,
                                "known_layer_metrology": None,
                                "known_layer_period_metrology": period.to_metadata(),
                            }
                        ],
                    }
                }
            }
        },
    )

    specifications = specifications_from_aggregated_dataset(
        dataset, (baseline, material)
    )
    assert specifications[0].observable_names == names
    assert isinstance(specifications[0].parts[0], ImagingLayerPeriods)
    predicted = predict_physical_observable_blocks(
        _column(),
        _column().nominal_parameters,
        DetectorGeometry(shape_yx=shape, pixel_size_yx_m=pixel_yx),
        specifications,
        dataset=dataset,
    )[0]
    assert predicted.shape == (len(period.span_pairs),)
    assert np.all(np.isfinite(np.asarray(predicted)))


def test_layer_material_aggregation_pools_covariance_and_rebuilds_exact_schema(tmp_path):
    _geometry, _h, shape, _pixel_yx, _image, metrology = _registered_fixture()
    plan = AcquisitionPlan(
        channel_names=("IL1", "PL1"),
        states=(AcquisitionState("state", (0.25, -0.35), "nominal", 3),),
        reference_dacs=(0.25, -0.35),
    )
    requests = (
        _request(0, "perforation"),
        _request(1, "sample", feature="individual_layer_widths"),
        _request(2, "sample_repeat", feature="individual_layer_widths"),
    )
    count = metrology.geometry.n_intervals
    names = metrology.observable_names
    metadata = metrology.to_metadata()
    augmentation = layer_observable_covariance_augmentation(metrology)
    shared_covariance = augmentation.total_covariance_px2
    blocks = (
        FrameObservationBlock(
            0,
            0,
            np.asarray((0.0, 0.0)),
            np.eye(2),
            ("centroid_x_px", "centroid_y_px"),
        ),
        FrameObservationBlock(
            1,
            0,
            np.arange(count),
            np.eye(count) * 2.0 + shared_covariance,
            names,
        ),
        FrameObservationBlock(
            2,
            0,
            np.arange(count) + 1.0,
            np.eye(count) * 6.0 + shared_covariance,
            names,
        ),
    )
    dataset = HeterogeneousInversionDataset(
        plan,
        blocks,
        metadata={
            "frame_metadata": {
                "0": {"known_layer_metrology": None},
                "1": {
                    "known_layer_metrology": metadata,
                    "layer_covariance_augmentation": augmentation.to_metadata(),
                },
                "2": {
                    "known_layer_metrology": metadata,
                    "layer_covariance_augmentation": augmentation.to_metadata(),
                },
            }
        },
    )
    noise = DetectorNoiseModel(1000.0)
    noisy_frames = tuple(
        NoisyDetectorFrame(request, np.ones(shape), 10 + request.frame_id, noise)
        for request in requests
    )
    product = CheckpointObservableDataset(
        CheckpointedAcquisition(
            checkpoint_directory=tmp_path,
            manifest_path=tmp_path / "manifest.json",
            manifest_sha256="fixture",
            records=(),
            pending_frame_ids=(),
        ),
        dataset,
        noisy_frames,
        8,
    )

    aggregated = aggregate_checkpoint_observables_for_fit(product)
    block = aggregated.blocks[0]
    independent_pooled_covariance = np.eye(count) * 1.5
    expected_covariance = independent_pooled_covariance + shared_covariance
    expected_value = independent_pooled_covariance @ (
        np.arange(count) / 2.0 + (np.arange(count) + 1.0) / 6.0
    )
    np.testing.assert_allclose(block.values, expected_value)
    np.testing.assert_allclose(block.covariance, expected_covariance)
    assert block.observable_names == tuple(
        f"material_pool.{name}" for name in names
    )
    material_metadata = aggregated.metadata["aggregation"]["state_blocks"]["0"][
        "material"
    ][0]
    assert material_metadata["known_fourier_features"] is None
    assert material_metadata["known_layer_metrology"]["geometry"]["certificate_id"] == (
        "certificate-test-v4"
    )
    np.testing.assert_allclose(
        material_metadata["layer_covariance_augmentation"]["total_covariance_px2"],
        shared_covariance,
    )

    specifications = specifications_from_aggregated_dataset(aggregated, requests)
    assert specifications[0].observable_names == block.observable_names
    assert isinstance(specifications[0].parts[0], ImagingLayerBoundaries)
    prediction = predict_physical_observable_blocks(
        _column(),
        _column().nominal_parameters,
        DetectorGeometry(shape_yx=shape, pixel_size_yx_m=(70e-6, 55e-6)),
        specifications,
        dataset=aggregated,
    )[0]
    assert prediction.shape == (count,)


def test_checkpoint_builder_runs_registered_extractor_for_actual_and_bootstraps(
    tmp_path,
):
    _geometry, _h, shape, pixel_yx, image, metrology = _registered_fixture(
        minimum_peak_snr=2.0
    )
    plan = AcquisitionPlan(
        channel_names=("IL1", "PL1"),
        states=(AcquisitionState("state", (0.25, -0.35), "nominal", 3),),
        reference_dacs=(0.25, -0.35),
    )
    requests = (
        _request(0, "perforation"),
        _request(1, "sample", feature="individual_layer_widths"),
        _request(2, "sample_repeat", feature="individual_layer_widths"),
    )
    detector = DetectorGeometry(shape_yx=shape, pixel_size_yx_m=pixel_yx)
    yy, xx = np.indices(shape, dtype=float)
    blank = 4200.0 * np.exp(
        -0.5
        * (
            ((xx - 0.5 * (shape[1] - 1)) / 55.0) ** 2
            + ((yy - 0.5 * (shape[0] - 1)) / 48.0) ** 2
        )
    ) + 200.0

    def render(request):
        intensity = (
            image
            if request.measurement in {"sample", "sample_repeat"}
            else blank
        )
        return FramePrediction(
            request=request,
            render=GaussianRenderResult(np.sqrt(intensity).astype(complex), intensity, 0.0),
            observables=extract_image_observables(intensity),
            interaction_route=request.interaction_route,
        )

    directory = tmp_path / "layer-checkpoint"
    render_frame_sequence_checkpointed(
        requests,
        detector,
        render,
        directory,
        render_signature="registered-layer-test:v1",
        render_metadata={"sample": "fixed"},
    )
    product = build_heterogeneous_dataset_from_checkpoint(
        plan,
        requests,
        detector,
        directory,
        render_signature="registered-layer-test:v1",
        render_metadata={"sample": "fixed"},
        default_noise_model=DetectorNoiseModel(
            dose_electrons=2.0e5,
            incident_intensity_sum=float(np.sum(image)),
            read_noise_e=0.5,
            dark_offset_e=2.0,
        ),
        bootstrap_replicates=12,
        seed=72,
        layer_metrology_by_family={"individual_layer_widths": metrology},
    )

    for frame_id in (1, 2):
        block = next(block for block in product.dataset.blocks if block.frame_id == frame_id)
        assert block.observable_names == metrology.observable_names
        assert np.linalg.eigvalsh(block.covariance).min() > 0.0
        frame_metadata = product.dataset.metadata["frame_metadata"][str(frame_id)]
        assert frame_metadata["known_fourier_features"] is None
        assert frame_metadata["known_layer_metrology"]["observable_names"] == list(
            metrology.observable_names
        )
        assert "RegisteredLayerMetrology" in frame_metadata["observable_extractor"]
        augmentation_metadata = frame_metadata["layer_covariance_augmentation"]
        certificate_covariance = np.asarray(
            augmentation_metadata["certificate_covariance_px2"]
        )
        assert np.all(np.diag(block.covariance) >= np.diag(certificate_covariance))
    assert not np.array_equal(product.frame(1).counts_yx, product.frame(2).counts_yx)

    aggregated = aggregate_checkpoint_observables_for_fit(product)
    assert aggregated.blocks[0].observable_names == tuple(
        f"material_pool.{name}" for name in metrology.observable_names
    )


@pytest.mark.parametrize(
    "bad_shape",
    ((196, 237), (238, 196)),
)
def test_extraction_rejects_shape_or_yx_transposition(bad_shape):
    geometry, _h, _shape, _pixel_yx, _image, metrology = _registered_fixture()
    with pytest.raises(ValueError, match="image_yx"):
        extract_registered_layer_boundaries(
            np.ones(bad_shape), geometry, metrology.registration
        )


def test_registration_rejects_overlapping_windows_and_low_signal():
    geometry, h, _shape, pixel_yx, image, metrology = _registered_fixture()
    with pytest.raises(ValueError, match="overlap"):
        LayerBoundaryRegistration2D(
            **{
                **metrology.registration.__dict__,
                "search_half_widths_px": np.full(geometry.n_boundaries, 100.0),
            }
        )
    flat = np.ones_like(image)
    with pytest.raises(ValueError, match="peak|boundary"):
        freeze_layer_boundary_registration(
            flat,
            geometry,
            detector_position_map_xy_m_per_m=h,
            detector_pixel_size_yx_m=pixel_yx,
            minimum_peak_snr=2.0,
        )
