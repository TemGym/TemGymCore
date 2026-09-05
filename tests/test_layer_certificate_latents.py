from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import linalg

from temgym_core.layer_metrology import (
    CertifiedLayerGeometry2D,
    LayerBoundaryRegistration2D,
    RegisteredLayerMetrology,
    RegisteredLayerPeriodMetrology,
    layer_observable_covariance_augmentation,
    predict_registered_layer_observables,
)
from temgym_core.microscope_inversion import (
    AcquisitionPlan,
    AcquisitionState,
    FrameObservationBlock,
    HeterogeneousInversionDataset,
    SamplePlaneState,
)
from temgym_core.neoarm_calibration import (
    DetectorGeometry,
    FrameRequest,
    HardEdgeColumnParameterization,
    HardEdgeLensInitial,
)
from temgym_core.physical_nuisance import (
    GlobalObservationLayout,
    SharedLatentNuisanceLikelihood,
)
from temgym_core.physical_observables import (
    ImagingLayerBoundaries,
    ImagingLayerPeriods,
    ImagingReciprocalVectors,
    PhysicalObservableBlockSpecification,
    calibration_priors_for_physical_observables,
    fourier_observable_covariance_augmentation,
    layer_certificate_calibrations_for_physical_observables,
    layer_certificate_covariance_augmentations_for_physical_observables,
    make_calibration_aware_physical_predictor,
    predict_physical_observable_blocks,
    strip_shared_layer_certificate_covariance,
    strip_shared_multilayer_certificate_covariance,
)


jax.config.update("jax_enable_x64", True)


_FULL_X_M = np.asarray((-30.0, -20.0, -7.0, 3.0, 16.0, 26.0)) * 1e-9
_FULL_SIGMA_M = np.asarray((0.0, 0.18, 0.25, 0.31, 0.37, 0.42)) * 1e-9


def _geometry(indices):
    indices = tuple(indices)
    x = _FULL_X_M[np.asarray(indices)]
    sigma = _FULL_SIGMA_M[np.asarray(indices)]
    count = len(indices)
    return CertifiedLayerGeometry2D(
        certificate_id="magical-layer-certificate-v1",
        certificate_source="traceable synthetic layer certificate",
        region_name="multilayer_region_2",
        boundary_indices=indices,
        boundary_labels=tuple(f"b{index:02d}" for index in indices),
        boundary_points_local_xy_m=np.column_stack((x, np.zeros(count))),
        boundary_line_endpoints_local_xy_m=np.stack(
            (
                np.column_stack((x, np.full(count, -40e-9))),
                np.column_stack((x, np.full(count, 40e-9))),
            ),
            axis=1,
        ),
        boundary_standard_uncertainties_m=sigma,
        interval_pairs=np.column_stack(
            (np.arange(count - 1), np.arange(1, count))
        ),
        interval_labels=tuple(
            f"l{right:02d}" for right in indices[1:]
        ),
        interval_materials=tuple(
            "si" if right % 2 == 0 else "sige" for right in indices[1:]
        ),
        sample_translation_xy_m=np.asarray((1.5e-9, -0.8e-9)),
        sample_projection_world_from_local=np.asarray(
            ((0.985, -0.12), (0.15, 0.97))
        ),
        boundary_uncertainty_model="cumulative_layer_widths",
    )


def _registration(indices):
    indices = tuple(indices)
    count = len(indices)
    centres = 20.0 + 14.0 * np.arange(count)
    polarity = np.asarray(
        tuple(1.0 if index % 2 == 0 else -1.0 for index in indices)
    )
    return LayerBoundaryRegistration2D(
        image_shape_yx=(128, 160),
        detector_normal_xy=np.asarray((1.0, 0.0)),
        detector_tangent_xy=np.asarray((0.0, 1.0)),
        coordinate_origin_xy_px=np.asarray((79.5, 63.5)),
        detector_optical_axis_xy_px=np.asarray((79.5, 63.5)),
        search_centres_axis_px=centres,
        search_half_widths_px=np.full(count, 5.0),
        reference_peak_centres_axis_px=centres,
        peak_tracking_half_widths_px=np.full(count, 2.0),
        edge_model_widths_px=np.full(count, 0.8),
        tangent_centres_axis_px=np.zeros(count),
        tangent_half_widths_px=np.full(count, 20.0),
        edge_polarities=polarity,
    )


def _column():
    return HardEdgeColumnParameterization(
        lenses=(
            HardEdgeLensInitial(
                "IL1", 0.42, 0.08, q_bias=0.61, q_slope_per_dac=0.18,
                channel_index=0,
            ),
            HardEdgeLensInitial(
                "PL1", 0.74, 0.08, q_bias=0.48, q_slope_per_dac=-0.14,
                channel_index=1,
            ),
        ),
        sample_z_m=0.2,
        source_z_m=0.0,
        detector_z_m=1.0,
        channel_names=("IL1", "PL1"),
    )


def _detector():
    return DetectorGeometry(
        shape_yx=(128, 160),
        pixel_size_yx_m=(58e-6, 53e-6),
        z_m=1.0,
        affine_xy=np.asarray(((0.98, -0.09), (0.11, 1.02))),
    )


def _problem():
    states = (
        AcquisitionState("state_a", (0.3, -0.2), "nominal", 2),
        AcquisitionState("state_b", (-0.4, 0.5), "d_optimal", 2),
    )
    plan = AcquisitionPlan(
        channel_names=("IL1", "PL1"),
        states=states,
        reference_dacs=(0.0, 0.0),
    )
    requests = (
        FrameRequest(
            frame_id=20,
            optical_state_index=0,
            optical_state_label="state_a",
            measurement="sample",
            sample_state=SamplePlaneState(),
            raw_dac_values=states[0].dac_values,
        ),
        FrameRequest(
            frame_id=21,
            optical_state_index=1,
            optical_state_label="state_b",
            measurement="sample",
            sample_state=SamplePlaneState(),
            raw_dac_values=states[1].dac_values,
        ),
    )
    first_metrology = RegisteredLayerMetrology(
        _geometry((0, 1, 2, 3)),
        _registration((0, 1, 2, 3)),
        "boundary_positions_and_interval_widths",
    )
    second_boundary = RegisteredLayerMetrology(
        _geometry((2, 3, 4, 5)),
        _registration((2, 3, 4, 5)),
        "boundary_positions",
    )
    second_metrology = RegisteredLayerPeriodMetrology(
        second_boundary,
        np.asarray(((0, 2), (1, 3))),
        ("b02_to_b04", "b03_to_b05"),
    )
    parts = (
        ImagingLayerBoundaries(
            requests[0], first_metrology, name_prefix="material_pool."
        ),
        ImagingLayerPeriods(
            requests[1],
            second_metrology,
            name_prefix="material_frame_21.",
        ),
    )
    specifications = (
        PhysicalObservableBlockSpecification(100, 0, (parts[0],)),
        PhysicalObservableBlockSpecification(101, 1, (parts[1],)),
    )
    return plan, requests, parts, specifications


def _dataset_with_standalone_certificate_covariance():
    plan, _requests, parts, specifications = _problem()
    column = _column()
    detector = _detector()
    parameters = column.nominal_parameters
    augmentations = (
        layer_certificate_covariance_augmentations_for_physical_observables(
            column, parameters, detector, specifications
        )
    )
    prediction = predict_physical_observable_blocks(
        column, parameters, detector, specifications
    )
    statistical = (
        np.diag(np.linspace(0.08, 0.14, len(parts[0].observable_names))),
        np.diag(np.linspace(0.05, 0.07, len(parts[1].observable_names))),
    )
    model_discrepancy = (
        np.diag(np.full(len(parts[0].observable_names), 0.012)),
        np.zeros_like(statistical[1]),
    )
    blocks = tuple(
        FrameObservationBlock(
            frame_id=spec.frame_id,
            optical_state_index=spec.optical_state_index,
            values=np.asarray(values),
            covariance=(
                statistical[index]
                + model_discrepancy[index]
                + augmentations[index].certificate_covariance_px2
            ),
            observable_names=spec.observable_names,
        )
        for index, (spec, values) in enumerate(
            zip(specifications, prediction, strict=True)
        )
    )
    first_metadata = {
        "observable_names": list(parts[0].metrology.observable_names),
        "certificate_covariance_px2": (
            augmentations[0].certificate_covariance_px2.tolist()
        ),
        "validated_model_discrepancy_covariance_px2": (
            model_discrepancy[0].tolist()
        ),
        "total_covariance_px2": (
            augmentations[0].certificate_covariance_px2
            + model_discrepancy[0]
        ).tolist(),
    }
    second_metadata = {
        "observable_names": list(parts[1].metrology.observable_names),
        "bootstrap_covariance_px2": statistical[1].tolist(),
        "certificate_covariance_px2": (
            augmentations[1].certificate_covariance_px2.tolist()
        ),
        "total_covariance_px2": (
            statistical[1] + augmentations[1].certificate_covariance_px2
        ).tolist(),
        "certificate_covariance_added_once": True,
    }
    dataset = HeterogeneousInversionDataset(
        plan,
        blocks,
        metadata={
            "aggregation": {
                "state_blocks": {
                    "100": {
                        "material": [
                            {
                                "mode": "precision_pool",
                                "member_frame_ids": [20],
                                "layer_covariance_augmentation": first_metadata,
                            }
                        ]
                    },
                    "101": {
                        "material": [
                            {
                                "mode": "retained",
                                "member_frame_ids": [21],
                                "layer_period_covariance_augmentation": second_metadata,
                            }
                        ]
                    },
                }
            }
        },
    )
    return (
        dataset,
        column,
        detector,
        parameters,
        specifications,
        augmentations,
        statistical,
        model_discrepancy,
    )


def test_one_certificate_prior_aligns_overlapping_crops_and_drives_both_states():
    _plan, _requests, _parts, specifications = _problem()
    calibrations = layer_certificate_calibrations_for_physical_observables(
        specifications
    )
    assert len(calibrations) == 1
    calibration = calibrations[0]
    assert calibration.group_name == "specimen_layers:magical-layer-certificate-v1"
    assert calibration.boundary_indices == (0, 1, 2, 3, 4, 5)
    assert calibration.rows_for_boundary_indices((2, 3)) == (2, 3)
    np.testing.assert_allclose(
        calibration.boundary_coordinates_local_x_m, _FULL_X_M
    )
    np.testing.assert_allclose(
        calibration.boundary_covariance_local_m2,
        np.minimum.outer(_FULL_SIGMA_M**2, _FULL_SIGMA_M**2),
    )

    column = _column()
    detector = _detector()
    priors, callback = make_calibration_aware_physical_predictor(
        column,
        column.nominal_parameters,
        detector,
        specifications,
    )
    assert tuple(prior.group_name for prior in priors) == (
        calibration.group_name,
    )
    nominal_values = (jnp.asarray(priors[0].mean),)
    nominal = callback(jnp.asarray(column.nominal_parameters), nominal_values)
    direct = predict_physical_observable_blocks(
        column, column.nominal_parameters, detector, specifications
    )
    for actual, expected in zip(nominal, direct, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    moved = np.asarray(priors[0].mean).copy()
    moved[2] += 0.4e-9
    changed = jax.jit(callback)(
        jnp.asarray(column.nominal_parameters), (jnp.asarray(moved),)
    )
    assert np.linalg.norm(np.asarray(changed[0]) - np.asarray(nominal[0])) > 0.0
    assert np.linalg.norm(np.asarray(changed[1]) - np.asarray(nominal[1])) > 0.0


def test_shared_layer_pushforward_has_cross_state_covariance_and_exact_marginals():
    (
        _dataset,
        _column_value,
        _detector_value,
        _parameters,
        _specifications,
        augmentations,
        _statistical,
        _model_discrepancy,
    ) = _dataset_with_standalone_certificate_covariance()
    first, second = augmentations
    joint_factor = np.vstack(
        (first.certificate_output_factor, second.certificate_output_factor)
    )
    joint_covariance = joint_factor @ joint_factor.T
    split = len(first.observable_names)
    np.testing.assert_allclose(
        joint_covariance[:split, :split], first.certificate_covariance_px2
    )
    np.testing.assert_allclose(
        joint_covariance[split:, split:], second.certificate_covariance_px2
    )
    assert np.linalg.norm(joint_covariance[:split, split:]) > 1e-12


def test_shared_pushforward_matches_legacy_standalone_layer_covariance():
    plan, requests, _parts, _specifications = _problem()
    column = _column()
    detector = DetectorGeometry(
        shape_yx=(128, 160),
        pixel_size_yx_m=(0.11e-9, 0.09e-9),
        z_m=1.0,
        affine_xy=np.asarray(((0.98, -0.09), (0.11, 1.02))),
    )
    geometry = _geometry((0, 1, 2, 3))
    provisional = RegisteredLayerMetrology(
        geometry, _registration((0, 1, 2, 3)), "interval_widths"
    )
    matrix = np.asarray(
        column.matrix(
            column.nominal_parameters,
            np.asarray(requests[0].raw_dac_values),
            segment="sample_to_detector",
        )
    )
    H = np.asarray(detector.affine_xy) @ matrix[np.ix_((0, 2), (0, 2))]
    boundary_metrology = replace(provisional, observable_mode="boundary_positions")
    predicted_centres = np.asarray(
        predict_registered_layer_observables(
            H, detector.pixel_size_yx_m, boundary_metrology
        )
    )
    if np.mean(np.diff(predicted_centres)) < 0.0:
        registration_with_forward_normal = replace(
            provisional.registration,
            detector_normal_xy=-provisional.registration.detector_normal_xy,
        )
        provisional = replace(provisional, registration=registration_with_forward_normal)
        boundary_metrology = replace(
            provisional, observable_mode="boundary_positions"
        )
        predicted_centres = np.asarray(
            predict_registered_layer_observables(
                H, detector.pixel_size_yx_m, boundary_metrology
            )
        )
    separation = float(np.min(np.diff(predicted_centres)))
    assert separation > 0.0
    registration = replace(
        provisional.registration,
        search_centres_axis_px=predicted_centres,
        reference_peak_centres_axis_px=predicted_centres,
        search_half_widths_px=np.full(geometry.n_boundaries, 0.35 * separation),
        peak_tracking_half_widths_px=np.full(
            geometry.n_boundaries, 0.15 * separation
        ),
    )
    metrology = RegisteredLayerMetrology(
        geometry, registration, "interval_widths"
    )
    part = ImagingLayerBoundaries(requests[0], metrology)
    spec = PhysicalObservableBlockSpecification(100, 0, (part,))
    exact = layer_certificate_covariance_augmentations_for_physical_observables(
        column, column.nominal_parameters, detector, (spec,)
    )[0]
    standalone = layer_observable_covariance_augmentation(metrology)
    np.testing.assert_allclose(
        exact.certificate_covariance_px2,
        standalone.certificate_covariance_px2,
        rtol=2e-10,
        atol=1e-14,
    )


def test_stripping_removes_certificate_once_and_retains_noise_and_discrepancy():
    (
        dataset,
        column,
        detector,
        parameters,
        specifications,
        augmentations,
        statistical,
        model_discrepancy,
    ) = _dataset_with_standalone_certificate_covariance()
    stripped = strip_shared_layer_certificate_covariance(
        dataset,
        column,
        parameters,
        detector,
        specifications,
    )
    for index, block in enumerate(stripped.blocks):
        np.testing.assert_allclose(
            block.covariance,
            statistical[index] + model_discrepancy[index],
            rtol=1e-12,
            atol=1e-14,
        )
    policy = stripped.metadata["layer_certificate_covariance"]
    assert policy["stripped_for_shared_latent_likelihood"] is True
    assert policy["stripped_subblock_count"] == 2
    with pytest.raises(ValueError, match="already been stripped"):
        strip_shared_layer_certificate_covariance(
            stripped,
            column,
            parameters,
            detector,
            specifications,
        )

    joint_factor = np.vstack(
        tuple(value.certificate_output_factor for value in augmentations)
    )
    latent_covariance = joint_factor @ joint_factor.T
    correct_marginal = (
        linalg.block_diag(*(block.covariance for block in stripped.blocks))
        + latent_covariance
    )
    standalone_block_diagonal = linalg.block_diag(
        *(block.covariance for block in dataset.blocks)
    )
    for block_index, block in enumerate(dataset.blocks):
        start = sum(value.values.size for value in dataset.blocks[:block_index])
        stop = start + block.values.size
        np.testing.assert_allclose(
            correct_marginal[start:stop, start:stop],
            block.covariance,
            rtol=1e-12,
            atol=1e-14,
        )
    double_counted = standalone_block_diagonal + latent_covariance
    expected_duplicate = linalg.block_diag(
        *(value.certificate_covariance_px2 for value in augmentations)
    )
    np.testing.assert_allclose(
        double_counted - correct_marginal,
        expected_duplicate,
        rtol=2e-6,
        atol=2e-17,
    )
    assert np.linalg.norm(expected_duplicate) > 0.0

    priors, callback = make_calibration_aware_physical_predictor(
        column,
        parameters,
        detector,
        specifications,
        dataset=stripped,
    )
    likelihood = SharedLatentNuisanceLikelihood(
        GlobalObservationLayout.from_dataset(stripped),
        column.n_parameters,
        callback,
        calibration_priors=priors,
    )
    assert likelihood.n_calibration_latents == priors[0].n_latents
    assert np.allclose(
        likelihood.joint_residual(
            likelihood.join_parameters(parameters)
        ),
        0.0,
        atol=1e-12,
    )


def test_inconsistent_overlapping_certificate_crop_fails_closed():
    _plan, requests, parts, specifications = _problem()
    geometry = parts[1].metrology.geometry
    points = np.asarray(geometry.boundary_points_local_xy_m).copy()
    endpoints = np.asarray(geometry.boundary_line_endpoints_local_xy_m).copy()
    points[0, 0] += 0.5e-9
    endpoints[0, :, 0] += 0.5e-9
    inconsistent_geometry = replace(
        geometry,
        boundary_points_local_xy_m=points,
        boundary_line_endpoints_local_xy_m=endpoints,
    )
    inconsistent_boundary = RegisteredLayerMetrology(
        inconsistent_geometry,
        parts[1].metrology.registration,
        "boundary_positions",
    )
    inconsistent_period = RegisteredLayerPeriodMetrology(
        inconsistent_boundary,
        parts[1].metrology.span_pairs,
        parts[1].metrology.span_labels,
    )
    inconsistent_part = ImagingLayerPeriods(
        requests[1], inconsistent_period, name_prefix="material_frame_21."
    )
    inconsistent_specs = (
        specifications[0],
        PhysicalObservableBlockSpecification(101, 1, (inconsistent_part,)),
    )
    with pytest.raises(ValueError, match="inconsistent coordinates"):
        layer_certificate_calibrations_for_physical_observables(
            inconsistent_specs
        )


_UNIFIED_X_M = np.asarray((-30.0, -20.0, -7.0, 3.0, 16.0, 26.0)) * 1e-9
_UNIFIED_INCREMENT_VARIANCE_M2 = (
    np.asarray((0.0, 0.18, 0.25, 0.18, 0.25, 0.18)) * 1e-9
) ** 2
_UNIFIED_VARIANCE_M2 = np.cumsum(_UNIFIED_INCREMENT_VARIANCE_M2)


def _unified_geometry(region_name, offset_m):
    indices = tuple(range(len(_UNIFIED_X_M)))
    x = _UNIFIED_X_M + float(offset_m)
    count = len(indices)
    return CertifiedLayerGeometry2D(
        certificate_id="magical-unified-v1",
        certificate_source="traceable shared-increment certificate",
        region_name=region_name,
        boundary_indices=indices,
        boundary_labels=tuple(f"b{index:02d}" for index in indices),
        boundary_points_local_xy_m=np.column_stack((x, np.zeros(count))),
        boundary_line_endpoints_local_xy_m=np.stack(
            (
                np.column_stack((x, np.full(count, -35e-9))),
                np.column_stack((x, np.full(count, 35e-9))),
            ),
            axis=1,
        ),
        boundary_standard_uncertainties_m=np.sqrt(_UNIFIED_VARIANCE_M2),
        interval_pairs=np.column_stack(
            (np.arange(count - 1), np.arange(1, count))
        ),
        interval_labels=tuple(f"l{index:02d}" for index in indices[1:]),
        interval_materials=tuple(
            "sige" if index % 2 else "si" for index in indices[1:]
        ),
        sample_translation_xy_m=np.asarray((0.7e-9, -1.1e-9)),
        sample_projection_world_from_local=np.asarray(
            ((0.985, -0.12), (0.15, 0.97))
        ),
        boundary_uncertainty_model="cumulative_layer_widths",
    )


def _unified_problem():
    plan, requests, _parts, _specifications = _problem()
    period = _UNIFIED_X_M[2] - _UNIFIED_X_M[0]
    period_variance = _UNIFIED_VARIANCE_M2[2] - _UNIFIED_VARIANCE_M2[0]
    direction_world = np.asarray((0.82, -0.37))
    world_vector = direction_world[None, :] / period
    derivative = -world_vector.reshape(2) / period
    fourier_covariance = np.outer(derivative, derivative) * period_variance
    low_request = FrameRequest(
        frame_id=22,
        optical_state_index=0,
        optical_state_label="state_a",
        measurement="sample",
        sample_state=SamplePlaneState(),
        raw_dac_values=plan.states[0].dac_values,
    )
    region_2_request = FrameRequest(
        frame_id=23,
        optical_state_index=0,
        optical_state_label="state_a",
        measurement="sample",
        sample_state=SamplePlaneState(),
        raw_dac_values=plan.states[0].dac_values,
    )
    region_4_request = FrameRequest(
        frame_id=24,
        optical_state_index=1,
        optical_state_label="state_b",
        measurement="sample",
        sample_state=SamplePlaneState(),
        raw_dac_values=plan.states[1].dac_values,
    )
    low_part = ImagingReciprocalVectors(
        request=low_request,
        labels=("multilayer_period",),
        world_reciprocal_vectors_xy_per_m=world_vector,
        name_prefix="material_frame_22.",
        covariance_world_per_m2=fourier_covariance,
        certificate_id="magical-unified-v1",
        covariance_provenance="shared SiGe+Si certificate widths",
        calibration_group_name=(
            "specimen_multilayer_period:magical-unified-v1"
        ),
    )
    region_2_metrology = RegisteredLayerMetrology(
        _unified_geometry("multilayer_region_2", -4e-9),
        _registration(range(len(_UNIFIED_X_M))),
        "interval_widths",
    )
    region_4_boundary = RegisteredLayerMetrology(
        _unified_geometry("multilayer_region_4", 7e-9),
        _registration(range(len(_UNIFIED_X_M))),
        "boundary_positions",
    )
    region_4_metrology = RegisteredLayerPeriodMetrology(
        region_4_boundary,
        np.asarray(((0, 2), (2, 4))),
        ("b00_to_b02", "b02_to_b04"),
    )
    layer_parts = (
        ImagingLayerBoundaries(
            region_2_request,
            region_2_metrology,
            name_prefix="material_frame_23.",
        ),
        ImagingLayerPeriods(
            region_4_request,
            region_4_metrology,
            name_prefix="material_frame_24.",
        ),
    )
    specs = (
        PhysicalObservableBlockSpecification(200, 0, (low_part,)),
        PhysicalObservableBlockSpecification(201, 0, (layer_parts[0],)),
        PhysicalObservableBlockSpecification(202, 1, (layer_parts[1],)),
    )
    return plan, (low_part, *layer_parts), specs


def test_unified_multilayer_prior_drives_reciprocal_and_two_registered_regions():
    _plan_value, parts, specifications = _unified_problem()
    calibration = layer_certificate_calibrations_for_physical_observables(
        specifications
    )[0]
    assert calibration.multilayer_period_boundary_indices == (0, 2)
    assert calibration.multilayer_fourier_group_names == (
        "specimen_multilayer_period:magical-unified-v1",
    )
    # Region 4 is deliberately the canonical frame; its current coordinates
    # must remain bit-for-bit the prior mean.
    np.testing.assert_array_equal(
        calibration.boundary_coordinates_local_x_m,
        parts[2].metrology.geometry.boundary_points_local_xy_m[:, 0],
    )
    assert calibration.region_origin_offset_m("multilayer_region_4") == 0.0
    assert calibration.region_origin_offset_m(
        "multilayer_region_2"
    ) == pytest.approx(-11e-9)

    column = _column()
    detector = _detector()
    priors = calibration_priors_for_physical_observables(
        column, column.nominal_parameters, detector, specifications
    )
    assert len(priors) == 1
    assert priors[0].group_name == "specimen_layers:magical-unified-v1"
    assert all(
        "width_increment" in name for name in priors[0].latent_names
    )
    callback_priors, callback = make_calibration_aware_physical_predictor(
        column, column.nominal_parameters, detector, specifications
    )
    nominal = jax.jit(callback)(
        jnp.asarray(column.nominal_parameters),
        tuple(jnp.asarray(prior.mean) for prior in callback_priors),
    )
    direct = predict_physical_observable_blocks(
        column, column.nominal_parameters, detector, specifications
    )
    for actual, expected in zip(nominal, direct, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    values = np.asarray(callback_priors[0].mean).copy()
    values[2] += 0.7e-9
    changed = callback(
        jnp.asarray(column.nominal_parameters), (jnp.asarray(values),)
    )
    nominal_period = _UNIFIED_X_M[2] - _UNIFIED_X_M[0]
    np.testing.assert_allclose(
        np.asarray(changed[0]) / np.asarray(nominal[0]),
        nominal_period / (nominal_period + 0.7e-9),
        rtol=2e-12,
        atol=2e-12,
    )
    assert np.linalg.norm(np.asarray(changed[1]) - np.asarray(nominal[1])) > 0.0
    assert np.linalg.norm(np.asarray(changed[2]) - np.asarray(nominal[2])) > 0.0

    def predictions_from_latents(z):
        calibrated = callback_priors[0].values(z)
        return jnp.concatenate(
            callback(jnp.asarray(column.nominal_parameters), (calibrated,))
        )

    jacobian = np.asarray(
        jax.jacfwd(predictions_from_latents)(
            jnp.zeros(callback_priors[0].n_latents)
        )
    )
    covariance = jacobian @ jacobian.T
    split_0 = len(parts[0].observable_names)
    split_1 = split_0 + len(parts[1].observable_names)
    fourier_augmentation = fourier_observable_covariance_augmentation(
        column, column.nominal_parameters, detector, parts[0]
    )
    layer_augmentations = (
        layer_certificate_covariance_augmentations_for_physical_observables(
            column, column.nominal_parameters, detector, specifications
        )
    )
    np.testing.assert_allclose(
        covariance[:split_0, :split_0],
        fourier_augmentation.certificate_covariance_observable2,
        rtol=2e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        covariance[split_0:split_1, split_0:split_1],
        layer_augmentations[0].certificate_covariance_px2,
        rtol=2e-10,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        covariance[split_1:, split_1:],
        layer_augmentations[1].certificate_covariance_px2,
        rtol=2e-10,
        atol=1e-14,
    )
    assert np.linalg.norm(covariance[:split_0, split_0:split_1]) > 0.0
    assert np.linalg.norm(covariance[:split_0, split_1:]) > 0.0


def test_unified_multilayer_stripping_removes_both_standalone_terms_once():
    plan, parts, specifications = _unified_problem()
    column = _column()
    detector = _detector()
    parameters = column.nominal_parameters
    predictions = predict_physical_observable_blocks(
        column, parameters, detector, specifications
    )
    fourier_augmentation = fourier_observable_covariance_augmentation(
        column, parameters, detector, parts[0]
    )
    layer_augmentations = (
        layer_certificate_covariance_augmentations_for_physical_observables(
            column, parameters, detector, specifications
        )
    )
    statistical = tuple(
        np.diag(np.linspace(0.09, 0.14, len(spec.observable_names)))
        for spec in specifications
    )
    calibration_covariances = (
        fourier_augmentation.total_covariance_observable2,
        layer_augmentations[0].certificate_covariance_px2,
        layer_augmentations[1].certificate_covariance_px2,
    )
    blocks = tuple(
        FrameObservationBlock(
            frame_id=spec.frame_id,
            optical_state_index=spec.optical_state_index,
            values=np.asarray(prediction),
            covariance=statistical[index] + calibration_covariances[index],
            observable_names=spec.observable_names,
        )
        for index, (spec, prediction) in enumerate(
            zip(specifications, predictions, strict=True)
        )
    )

    def retained_entry(part, **metadata):
        return {
            "mode": "retained",
            "member_frame_ids": [part.request.frame_id],
            **metadata,
        }

    layer_metadata = []
    for part, augmentation, key in zip(
        parts[1:],
        layer_augmentations,
        ("layer_covariance_augmentation", "layer_period_covariance_augmentation"),
        strict=True,
    ):
        payload = {
            "observable_names": list(part.metrology.observable_names),
            "certificate_covariance_px2": (
                augmentation.certificate_covariance_px2.tolist()
            ),
        }
        if key == "layer_period_covariance_augmentation":
            payload["certificate_covariance_added_once"] = True
        layer_metadata.append(retained_entry(part, **{key: payload}))
    dataset = HeterogeneousInversionDataset(
        plan,
        blocks,
        metadata={
            "aggregation": {
                "state_blocks": {
                    "200": {
                        "material": [
                            retained_entry(
                                parts[0],
                                fourier_covariance_augmentation={
                                    **fourier_augmentation.to_metadata(),
                                    "observable_names": [
                                        name.removeprefix("material_frame_22.")
                                        for name in fourier_augmentation.observable_names
                                    ],
                                },
                            )
                        ]
                    },
                    "201": {"material": [layer_metadata[0]]},
                    "202": {"material": [layer_metadata[1]]},
                }
            }
        },
    )
    with pytest.raises(ValueError, match="must be prepared with"):
        make_calibration_aware_physical_predictor(
            column,
            parameters,
            detector,
            specifications,
            dataset=dataset,
        )
    stripped = strip_shared_multilayer_certificate_covariance(
        dataset,
        column,
        parameters,
        detector,
        specifications,
    )
    for actual, expected in zip(stripped.blocks, statistical, strict=True):
        np.testing.assert_allclose(actual.covariance, expected, rtol=3e-9, atol=1e-14)
    policy = stripped.metadata["multilayer_certificate_covariance"]
    assert policy["stripped_for_shared_latent_likelihood"] is True
    assert policy["period_boundary_indices"] == {"magical-unified-v1": [0, 2]}
    shared_priors, _callback = make_calibration_aware_physical_predictor(
        column,
        parameters,
        detector,
        specifications,
        dataset=stripped,
    )
    assert tuple(prior.group_name for prior in shared_priors) == (
        "specimen_layers:magical-unified-v1",
    )
    with pytest.raises(ValueError, match="already been stripped"):
        strip_shared_multilayer_certificate_covariance(
            stripped,
            column,
            parameters,
            detector,
            specifications,
        )


def test_unified_multilayer_relation_rejects_inconsistent_fourier_covariance():
    _plan_value, parts, specifications = _unified_problem()
    bad_fourier = replace(
        parts[0],
        covariance_world_per_m2=1.3 * parts[0].covariance_world_per_m2,
    )
    bad_specs = (
        replace(specifications[0], parts=(bad_fourier,)),
        *specifications[1:],
    )
    with pytest.raises(ValueError, match="not the push-forward"):
        layer_certificate_calibrations_for_physical_observables(bad_specs)
