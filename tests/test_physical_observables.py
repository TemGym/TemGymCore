import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
from temgym_core.physical_observables import (
    AngularDistributionReciprocalVectors,
    CentroidDifference,
    ImagingReciprocalVectors,
    IndexedDiffractionBraggSpots,
    NativeDiffractionReciprocalVectors,
    PhysicalObservableBlockSpecification,
    SingularObservableMapError,
    UnsupportedPhysicalObservableError,
    predict_physical_observable_blocks,
    specifications_from_aggregated_dataset,
    validate_physical_observable_schema,
)


jax.config.update("jax_enable_x64", True)


def _column(*, powered: bool) -> HardEdgeColumnParameterization:
    return HardEdgeColumnParameterization(
        lenses=(
            HardEdgeLensInitial(
                name="IL1",
                z_center_m=0.42,
                length_m=0.08,
                q_bias=0.62 if powered else 0.0,
                q_slope_per_dac=0.17 if powered else 0.0,
                channel_index=0,
            ),
            HardEdgeLensInitial(
                name="PL1",
                z_center_m=0.74,
                length_m=0.08,
                q_bias=0.51 if powered else 0.0,
                q_slope_per_dac=-0.13 if powered else 0.0,
                channel_index=1,
            ),
        ),
        sample_z_m=0.2,
        source_z_m=0.0,
        detector_z_m=1.0,
        channel_names=("IL1", "PL1"),
    )


def _request(
    frame_id: int,
    measurement: str,
    sample_state: SamplePlaneState | None = None,
    *,
    dacs: tuple[float, float] = (0.25, -0.35),
) -> FrameRequest:
    return FrameRequest(
        frame_id=frame_id,
        optical_state_index=0,
        optical_state_label="state",
        measurement=measurement,
        sample_state=SamplePlaneState() if sample_state is None else sample_state,
        raw_dac_values=dacs,
    )


def _detector(*, angle_rad: float = 0.0) -> DetectorGeometry:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return DetectorGeometry(
        shape_yx=(80, 120),
        pixel_size_yx_m=(6.0e-6, 4.0e-6),
        z_m=1.0,
        mtf_sigma_px=0.4,
        affine_xy=np.asarray(((cosine, -sine), (sine, cosine))),
    )


def _plan() -> AcquisitionPlan:
    return AcquisitionPlan(
        channel_names=("IL1", "PL1"),
        states=(AcquisitionState("state", (0.25, -0.35), "nominal", 4),),
        reference_dacs=(0.0, 0.0),
    )


def test_imaging_reciprocal_vectors_apply_rotation_units_and_yx_order():
    column = _column(powered=False)
    detector = _detector(angle_rad=0.31)
    request = _request(2, "sample")
    vectors = np.asarray(((2.0e6, -3.0e6), (5.0e6, 1.0e6)))
    part = ImagingReciprocalVectors(request, ("a", "b"), vectors)
    spec = PhysicalObservableBlockSpecification(2, 0, (part,))

    predicted = np.asarray(
        predict_physical_observable_blocks(
            column, column.nominal_parameters, detector, (spec,)
        )[0]
    ).reshape(-1, 2)

    # With zero lens strength A=I, so H is exactly the detector affine.
    transformed_xy_per_m = np.linalg.solve(detector.affine_xy.T, vectors.T).T
    fov_xy_m = np.asarray(
        (
            detector.shape_yx[1] * detector.pixel_size_yx_m[1],
            detector.shape_yx[0] * detector.pixel_size_yx_m[0],
        )
    )
    expected_yx_bins = (transformed_xy_per_m * fov_xy_m)[:, ::-1]
    np.testing.assert_allclose(predicted, expected_yx_bins, rtol=1e-12, atol=1e-12)
    assert spec.observable_names == (
        "fourier_a_y_px",
        "fourier_a_x_px",
        "fourier_b_y_px",
        "fourier_b_x_px",
    )


def test_centroid_difference_and_declared_probe_covariance_only():
    column = _column(powered=False)
    detector = _detector(angle_rad=-0.18)
    baseline = _request(0, "perforation")
    state = SamplePlaneState(x=1.2e-6, theta_x=-2.0e-5, y=0.7e-6, theta_y=3e-5)
    request = _request(1, "position_basis", state)

    no_covariance = CentroidDifference(request, baseline)
    no_cov_spec = PhysicalObservableBlockSpecification(0, 0, (no_covariance,))
    no_cov_prediction = predict_physical_observable_blocks(
        column, column.nominal_parameters, detector, (no_cov_spec,)
    )[0]
    assert no_cov_prediction.shape == (2,)

    covariance = np.diag((0.2e-6**2, 1.0e-5**2, 0.3e-6**2, 1.5e-5**2))
    with_covariance = CentroidDifference(request, baseline, covariance)
    cov_spec = PhysicalObservableBlockSpecification(0, 0, (with_covariance,))
    predicted = np.asarray(
        predict_physical_observable_blocks(
            column, column.nominal_parameters, detector, (cov_spec,)
        )[0]
    )

    matrix = np.asarray(
        column.matrix(
            column.nominal_parameters,
            request.raw_dac_values,
            segment="sample_to_detector",
        )
    )
    pixel_xy = np.asarray(
        (detector.pixel_size_yx_m[1], detector.pixel_size_yx_m[0])
    )
    response = (detector.affine_xy / pixel_xy[:, None]) @ matrix[(0, 2), :]
    expected_centroid = response @ state.as_array()
    expected_covariance = (
        response @ covariance @ response.T
        + detector.mtf_sigma_px**2 * np.eye(2)
    )
    expected = np.asarray(
        (
            expected_centroid[0],
            expected_centroid[1],
            expected_covariance[0, 0],
            expected_covariance[0, 1],
            expected_covariance[1, 1],
        )
    )
    np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-12)


def test_native_diffraction_uses_actual_dacs_b_inverse_transpose_and_yx_bins():
    column = _column(powered=False)
    detector = _detector(angle_rad=0.23)
    request = _request(3, "indexed_diffraction", dacs=(0.77, -0.81))
    angular_vectors = np.asarray(((0.9, -0.4), (-0.2, 0.75)))
    part = NativeDiffractionReciprocalVectors(
        request, ("g1", "g2"), angular_vectors
    )
    spec = PhysicalObservableBlockSpecification(3, 0, (part,))

    predicted = np.asarray(
        predict_physical_observable_blocks(
            column, column.nominal_parameters, detector, (spec,)
        )[0]
    ).reshape(-1, 2)
    matrix = np.asarray(
        column.matrix(
            column.nominal_parameters,
            request.raw_dac_values,
            segment="sample_to_detector",
        )
    )
    detector_from_angle = detector.affine_xy @ matrix[np.ix_((0, 2), (1, 3))]
    detector_per_m = np.linalg.solve(detector_from_angle.T, angular_vectors.T).T
    fov_xy_m = np.asarray(
        (
            detector.shape_yx[1] * detector.pixel_size_yx_m[1],
            detector.shape_yx[0] * detector.pixel_size_yx_m[0],
        )
    )
    expected = (detector_per_m * fov_xy_m)[:, ::-1]
    np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-12)


def test_indexed_bragg_spots_use_direct_b_map_wavelength_units_and_yx_pixels():
    column = _column(powered=False)
    detector = _detector(angle_rad=-0.27)
    request = _request(3, "indexed_diffraction", dacs=(0.71, -0.63))
    vectors = np.asarray(((3.2e9, -1.1e9), (-0.8e9, 2.4e9)))
    wavelength_m = 2.5079e-12
    part = IndexedDiffractionBraggSpots(
        request,
        ("si_111_a", "si_111_b"),
        vectors,
        wavelength_m,
        "200_kV_relativistic_wavelength",
    )
    spec = PhysicalObservableBlockSpecification(3, 0, (part,))

    predicted = np.asarray(
        predict_physical_observable_blocks(
            column, column.nominal_parameters, detector, (spec,)
        )[0]
    ).reshape(-1, 2)
    matrix = np.asarray(
        column.matrix(
            column.nominal_parameters,
            request.raw_dac_values,
            segment="sample_to_detector",
        )
    )
    detector_from_angle = detector.affine_xy @ matrix[np.ix_((0, 2), (1, 3))]
    detector_xy_m = (detector_from_angle @ (wavelength_m * vectors).T).T
    pixel_xy_m = np.asarray(
        (detector.pixel_size_yx_m[1], detector.pixel_size_yx_m[0])
    )
    expected_yx_px = (detector_xy_m / pixel_xy_m)[:, ::-1]
    np.testing.assert_allclose(predicted, expected_yx_px, rtol=1e-12, atol=1e-12)
    assert "200_kV_relativistic_wavelength" in spec.observable_provenance[0][1]
    assert spec.observable_names[:2] == (
        "diffraction_frame_3.bragg_si_111_a_y_px",
        "diffraction_frame_3.bragg_si_111_a_x_px",
    )


def test_bragg_and_angular_covector_maps_are_distinct_and_native_dacs_matter():
    column = _column(powered=True)
    detector = _detector()
    first_request = _request(3, "indexed_diffraction", dacs=(0.45, -0.55))
    second_request = _request(4, "indexed_diffraction", dacs=(-0.35, 0.65))
    vector = np.asarray(((3.0e9, 1.0e9),))
    first_bragg = IndexedDiffractionBraggSpots(
        first_request, ("g",), vector, 2.5e-12, "declared_test_wavelength"
    )
    second_bragg = IndexedDiffractionBraggSpots(
        second_request, ("g",), vector, 2.5e-12, "declared_test_wavelength"
    )
    angular_covector = AngularDistributionReciprocalVectors(
        first_request, ("q",), np.asarray(((0.7, -0.2),))
    )
    specs = (
        PhysicalObservableBlockSpecification(3, 0, (first_bragg,)),
        PhysicalObservableBlockSpecification(4, 0, (second_bragg,)),
        PhysicalObservableBlockSpecification(5, 0, (angular_covector,)),
    )
    predictions = predict_physical_observable_blocks(
        column, column.nominal_parameters, detector, specs
    )
    assert not np.allclose(predictions[0], predictions[1], rtol=1e-8, atol=1e-8)
    # Both have two components, but direct B spot positions and B^-T covectors
    # are deliberately different physical observables and numerical maps.
    assert not np.allclose(predictions[0], predictions[2], rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize(
    ("wavelength", "provenance"),
    ((0.0, "source"), (-1e-12, "source"), (np.nan, "source"), (2.5e-12, "")),
)
def test_indexed_bragg_spots_require_valid_wavelength_provenance(
    wavelength, provenance
):
    with pytest.raises(ValueError, match="wavelength|provenance"):
        IndexedDiffractionBraggSpots(
            _request(3, "indexed_diffraction"),
            ("g",),
            np.asarray(((3.0e9, 0.0),)),
            wavelength,
            provenance,
        )


def test_prediction_is_jax_differentiable_and_sensitive_to_il_and_pl():
    column = _column(powered=True)
    detector = _detector(angle_rad=0.11)
    request = _request(3, "indexed_diffraction", dacs=(0.43, -0.58))
    part = IndexedDiffractionBraggSpots(
        request,
        ("g1", "g2"),
        np.asarray(((3.0e9, 0.8e9), (-1.1e9, 2.7e9))),
        2.5079e-12,
        "200_kV_relativistic_wavelength",
    )
    spec = PhysicalObservableBlockSpecification(3, 0, (part,))

    def predict(parameters):
        return predict_physical_observable_blocks(
            column, parameters, detector, (spec,)
        )[0]

    jacobian = np.asarray(jax.jacfwd(predict)(jnp.asarray(column.nominal_parameters)))
    assert jacobian.shape == (4, column.n_parameters)
    assert np.all(np.isfinite(jacobian))
    names = column.parameter_names
    for parameter_name in (
        "IL1.q_bias",
        "IL1.q_slope_per_dac",
        "PL1.q_bias",
        "PL1.q_slope_per_dac",
    ):
        assert np.linalg.norm(jacobian[:, names.index(parameter_name)]) > 1e-8


def test_singular_position_map_is_rejected():
    column = HardEdgeColumnParameterization(
        lenses=(
            HardEdgeLensInitial(
                "IL1",
                z_center_m=0.25,
                length_m=0.1,
                q_bias=np.pi / 2.0,
            ),
        ),
        sample_z_m=0.2,
        source_z_m=0.0,
        detector_z_m=0.3,
        channel_names=("unused",),
    )
    request = FrameRequest(
        frame_id=0,
        optical_state_index=0,
        optical_state_label="singular",
        measurement="sample",
        sample_state=SamplePlaneState(),
        raw_dac_values=(0.0,),
    )
    part = ImagingReciprocalVectors(request, ("g",), np.asarray(((1.0e6, 0.0),)))
    spec = PhysicalObservableBlockSpecification(0, 0, (part,))
    with pytest.raises(SingularObservableMapError, match="position map H"):
        predict_physical_observable_blocks(
            column, column.nominal_parameters, DetectorGeometry(z_m=0.3), (spec,)
        )


def test_aggregated_specification_builder_matches_schema_exactly():
    baseline = _request(0, "perforation")
    response = _request(
        1,
        "position_basis",
        SamplePlaneState(x=1.0e-6),
    )
    material = _request(2, "sample")
    feature = {
        "certificate_id": "certificate-7",
        "labels": ["layer_period"],
        "world_reciprocal_vectors_xy_per_m": [[2.5e7, 0.0]],
    }
    names = (
        "position_basis_frame_1.delta_centroid_x_px",
        "position_basis_frame_1.delta_centroid_y_px",
        "material_pool.fourier_layer_period_y_px",
        "material_pool.fourier_layer_period_x_px",
    )
    block = FrameObservationBlock(0, 0, np.zeros(4), np.eye(4), names)
    dataset = HeterogeneousInversionDataset(
        _plan(),
        (block,),
        metadata={
            "aggregation": {
                "state_blocks": {
                    "0": {
                        "baseline_frame_id": 0,
                        "response_frame_ids": [1],
                        "material": [
                            {
                                "mode": "precision_pool",
                                "member_frame_ids": [2],
                                "known_fourier_features": feature,
                            }
                        ],
                    }
                }
            }
        },
    )

    specifications = specifications_from_aggregated_dataset(
        dataset, (baseline, response, material)
    )
    assert specifications[0].observable_names == names
    predicted = predict_physical_observable_blocks(
        _column(powered=True),
        _column(powered=True).nominal_parameters,
        _detector(),
        specifications,
        dataset=dataset,
    )
    assert predicted[0].shape == (4,)


def test_schema_rejects_unsupported_intensity_or_material_moments():
    request = _request(2, "sample")
    part = ImagingReciprocalVectors(request, ("g",), np.asarray(((1.0e6, 0.0),)))
    spec = PhysicalObservableBlockSpecification(2, 0, (part,))
    bad_block = FrameObservationBlock(
        2,
        0,
        np.zeros(2),
        np.eye(2),
        ("total_intensity", "covariance_xx_px2"),
    )
    bad_dataset = HeterogeneousInversionDataset(_plan(), (bad_block,))
    with pytest.raises(
        UnsupportedPhysicalObservableError, match="Unsupported dataset names"
    ):
        validate_physical_observable_schema((spec,), bad_dataset)


def test_centroid_difference_rejects_cross_state_or_cross_dac_baseline():
    request = _request(1, "angle_basis", SamplePlaneState(theta_x=1e-4))
    baseline = _request(0, "perforation", dacs=(0.26, -0.35))
    with pytest.raises(ValueError, match="same-state, same-DAC"):
        CentroidDifference(request, baseline)
