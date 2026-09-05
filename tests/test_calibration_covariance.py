from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.calibration_covariance import (
    deduplicate_calibration_priors,
    gaussian_prior_from_metadata,
)
from temgym_core.microscope_inversion import (
    AcquisitionPlan,
    AcquisitionState,
    FrameObservationBlock,
    HeterogeneousInversionDataset,
    SamplePlaneState,
)
from temgym_core.neoarm_calibration import (
    CheckpointObservableDataset,
    CheckpointedAcquisition,
    DetectorGeometry,
    DetectorNoiseModel,
    FramePrediction,
    FrameRequest,
    GaussianRenderResult,
    HardEdgeColumnParameterization,
    HardEdgeLensInitial,
    KnownFourierFeatureSet,
    NoisyDetectorFrame,
    aggregate_checkpoint_observables_for_fit,
    build_heterogeneous_dataset_from_checkpoint,
    extract_image_observables,
    render_frame_sequence_checkpointed,
    shared_fourier_calibration_priors_from_dataset,
    strip_shared_fourier_calibration_covariance,
)
from temgym_core.physical_observables import (
    ImagingReciprocalVectors,
    IndexedDiffractionBraggSpots,
    PhysicalObservableBlockSpecification,
    calibration_covariance_augmentations_for_physical_observables,
    fourier_observable_covariance_augmentation,
    imaging_reciprocal_vectors_from_feature_set,
    indexed_diffraction_bragg_spots_from_feature_set,
    make_calibration_aware_physical_predictor,
    predict_physical_observable_blocks,
)


jax.config.update("jax_enable_x64", True)


def _column(*, powered: bool = False) -> HardEdgeColumnParameterization:
    return HardEdgeColumnParameterization(
        lenses=(
            HardEdgeLensInitial(
                "IL1",
                z_center_m=0.45,
                length_m=0.08,
                q_bias=0.7 if powered else 0.0,
                q_slope_per_dac=0.2 if powered else 0.0,
                channel_index=0,
            ),
            HardEdgeLensInitial(
                "PL1",
                z_center_m=0.78,
                length_m=0.06,
                q_bias=0.5 if powered else 0.0,
                q_slope_per_dac=-0.15 if powered else 0.0,
                channel_index=1,
            ),
        ),
        source_z_m=0.0,
        sample_z_m=0.2,
        detector_z_m=1.0,
        channel_names=("IL1", "PL1"),
    )


def _detector() -> DetectorGeometry:
    angle = 0.21
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    return DetectorGeometry(
        shape_yx=(96, 128),
        pixel_size_yx_m=(6.5e-6, 4.5e-6),
        z_m=1.0,
        affine_xy=rotation @ np.asarray(((1.02, 0.03), (0.0, 0.97))),
    )


def _request(
    frame_id: int,
    measurement: str,
    *,
    dacs: tuple[float, float] = (0.2, -0.3),
) -> FrameRequest:
    return FrameRequest(
        frame_id=frame_id,
        optical_state_index=0,
        optical_state_label="state",
        measurement=measurement,
        sample_state=SamplePlaneState(),
        raw_dac_values=dacs,
    )


def _vector_covariance() -> np.ndarray:
    factor = np.asarray(
        (
            (1.6e5, 0.0, 0.0),
            (-0.4e5, 1.1e5, 0.0),
            (0.7e5, 0.2e5, 0.8e5),
            (0.3e5, -0.6e5, 0.5e5),
        )
    )
    return factor @ factor.T


def _feature_set() -> KnownFourierFeatureSet:
    vectors = np.asarray(((2.7e9, -0.8e9), (0.6e9, 3.0e9)))
    return KnownFourierFeatureSet(
        labels=("g0", "g1"),
        local_reciprocal_vectors_xy_per_m=vectors,
        world_reciprocal_vectors_xy_per_m=vectors,
        covariance_world_per_m2=_vector_covariance(),
        certificate_id="magical-certificate-v1",
        covariance_provenance="traceable certified reciprocal-vector covariance",
        calibration_group_name="magical-si111-vectors",
    )


def _imaging_part(request: FrameRequest) -> ImagingReciprocalVectors:
    return imaging_reciprocal_vectors_from_feature_set(
        request,
        _feature_set(),
    )


def _diffraction_part(
    request: FrameRequest,
    *,
    wavelength_variance_m2: float | None = (2.0e-16) ** 2,
) -> IndexedDiffractionBraggSpots:
    return indexed_diffraction_bragg_spots_from_feature_set(
        request,
        _feature_set(),
        2.5079e-12,
        "200 kV calibrated high-voltage readback",
        wavelength_variance_m2=wavelength_variance_m2,
        wavelength_calibration_group_name=(
            None
            if wavelength_variance_m2 is None
            else "neoarm-electron-wavelength"
        ),
    )


def test_imaging_certificate_covariance_uses_exact_h_inverse_transpose_map():
    column = _column()
    detector = _detector()
    part = _imaging_part(_request(1, "sample"))
    augmentation = fourier_observable_covariance_augmentation(
        column, column.nominal_parameters, detector, part
    )

    matrix = np.asarray(
        column.matrix(
            column.nominal_parameters,
            np.asarray(part.request.raw_dac_values),
            segment="sample_to_detector",
        )
    )
    h = detector.affine_xy @ matrix[np.ix_((0, 2), (0, 2))]
    fov_xy = np.asarray(
        (
            detector.shape_yx[1] * detector.pixel_size_yx_m[1],
            detector.shape_yx[0] * detector.pixel_size_yx_m[0],
        )
    )
    yx_from_xy = np.asarray(((0.0, 1.0), (1.0, 0.0)))
    one_vector = yx_from_xy @ np.diag(fov_xy) @ np.linalg.inv(h).T
    jacobian = np.kron(np.eye(2), one_vector)
    expected = jacobian @ _vector_covariance() @ jacobian.T

    np.testing.assert_allclose(
        augmentation.world_vector_jacobian, jacobian, rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        augmentation.certificate_covariance_observable2,
        expected,
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        augmentation.certificate_prior.covariance,
        _vector_covariance(),
        rtol=1e-12,
        atol=1e-6,
    )
    metadata = augmentation.to_metadata()
    assert metadata["shared_across_same_specimen_repeats"] is True
    assert "must_use_shared_input_latents" in metadata[
        "cross_state_calibration_correlation"
    ]
    rebuilt = gaussian_prior_from_metadata(metadata["shared_input_priors"][0])
    np.testing.assert_allclose(rebuilt.covariance, _vector_covariance(), rtol=1e-12)


def test_diffraction_decomposes_certificate_wavelength_and_bilinear_covariance():
    column = _column(powered=True)
    detector = _detector()
    part = _diffraction_part(_request(8, "indexed_diffraction"))
    augmentation = fourier_observable_covariance_augmentation(
        column, column.nominal_parameters, detector, part
    )

    matrix = np.asarray(
        column.matrix(
            column.nominal_parameters,
            np.asarray(part.request.raw_dac_values),
            segment="sample_to_detector",
        )
    )
    pixel_xy = np.asarray(detector.pixel_size_yx_m)[::-1]
    b = detector.affine_xy @ matrix[np.ix_((0, 2), (1, 3))]
    yx_from_xy = np.asarray(((0.0, 1.0), (1.0, 0.0)))
    unit = np.kron(np.eye(2), yx_from_xy @ np.diag(1.0 / pixel_xy) @ b)
    j_g = part.wavelength_m * unit
    j_wavelength = (
        unit @ part.world_reciprocal_vectors_xy_per_m.reshape(-1)
    )[:, None]
    variance = float(part.wavelength_variance_m2)

    expected_certificate = j_g @ _vector_covariance() @ j_g.T
    expected_wavelength = variance * j_wavelength @ j_wavelength.T
    expected_product = variance * unit @ _vector_covariance() @ unit.T
    np.testing.assert_allclose(
        augmentation.certificate_covariance_observable2,
        expected_certificate,
        rtol=1e-11,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        augmentation.wavelength_covariance_observable2,
        expected_wavelength,
        rtol=1e-11,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        augmentation.bilinear_product_covariance_observable2,
        expected_product,
        rtol=1e-11,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        augmentation.total_covariance_observable2,
        expected_certificate + expected_wavelength + expected_product,
        rtol=1e-11,
        atol=1e-13,
    )


def test_diffraction_covariance_fails_closed_without_explicit_wavelength_variance():
    part = _diffraction_part(
        _request(8, "indexed_diffraction"), wavelength_variance_m2=None
    )
    with pytest.raises(ValueError, match="explicit wavelength_variance_m2"):
        fourier_observable_covariance_augmentation(
            _column(), _column().nominal_parameters, _detector(), part
        )


def test_shared_priors_are_deduplicated_and_drive_all_states_in_one_callback():
    column = _column(powered=True)
    detector = _detector()
    parts = (
        _diffraction_part(
            _request(8, "indexed_diffraction", dacs=(0.25, -0.35))
        ),
        _diffraction_part(
            _request(9, "indexed_diffraction", dacs=(-0.45, 0.55))
        ),
    )
    specs = tuple(
        PhysicalObservableBlockSpecification(part.request.frame_id, 0, (part,))
        for part in parts
    )
    augmentations = calibration_covariance_augmentations_for_physical_observables(
        column, column.nominal_parameters, detector, specs
    )
    priors = deduplicate_calibration_priors(augmentations)
    assert [prior.group_name for prior in priors] == [
        "magical-si111-vectors",
        "neoarm-electron-wavelength",
    ]

    callback_priors, callback = make_calibration_aware_physical_predictor(
        column, column.nominal_parameters, detector, specs
    )
    assert tuple(prior.group_name for prior in callback_priors) == tuple(
        prior.group_name for prior in priors
    )
    means = tuple(jnp.asarray(prior.mean) for prior in callback_priors)
    nominal = callback(jnp.asarray(column.nominal_parameters), means)
    direct = predict_physical_observable_blocks(
        column, column.nominal_parameters, detector, specs
    )
    for actual, expected in zip(nominal, direct, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    perturbed = list(means)
    vector_index = next(
        index
        for index, prior in enumerate(callback_priors)
        if prior.group_name == "magical-si111-vectors"
    )
    wavelength_index = next(
        index
        for index, prior in enumerate(callback_priors)
        if prior.group_name == "neoarm-electron-wavelength"
    )
    perturbed[vector_index] = perturbed[vector_index] * 1.0002
    perturbed[wavelength_index] = perturbed[wavelength_index] * 0.9998
    changed = jax.jit(callback)(
        jnp.asarray(column.nominal_parameters), tuple(perturbed)
    )
    assert all(
        not np.allclose(actual, expected, rtol=1e-10, atol=1e-10)
        for actual, expected in zip(changed, direct, strict=True)
    )


def test_same_state_fourier_repeat_pooling_preserves_shared_covariance(tmp_path):
    column = _column()
    detector = _detector()
    requests = (
        _request(0, "perforation"),
        _request(1, "sample"),
        _request(2, "sample_repeat"),
    )
    plan = AcquisitionPlan(
        channel_names=("IL1", "PL1"),
        states=(AcquisitionState("state", (0.2, -0.3), "nominal", 3),),
        reference_dacs=(0.2, -0.3),
    )
    parts = (_imaging_part(requests[1]), _imaging_part(requests[2]))
    augmentations = tuple(
        fourier_observable_covariance_augmentation(
            column, column.nominal_parameters, detector, part
        )
        for part in parts
    )
    shared = augmentations[0].total_covariance_observable2
    np.testing.assert_allclose(
        augmentations[1].total_covariance_observable2, shared
    )
    feature_metadata = {
        "certificate_id": parts[0].certificate_id,
        "labels": list(parts[0].labels),
        "local_reciprocal_vectors_xy_per_m": (
            parts[0].world_reciprocal_vectors_xy_per_m.tolist()
        ),
        "world_reciprocal_vectors_xy_per_m": (
            parts[0].world_reciprocal_vectors_xy_per_m.tolist()
        ),
        "covariance_world_per_m2": parts[0].covariance_world_per_m2.tolist(),
        "covariance_provenance": parts[0].covariance_provenance,
        "calibration_group_name": parts[0].calibration_group_name,
    }
    blocks = (
        FrameObservationBlock(
            0,
            0,
            np.zeros(2),
            np.eye(2),
            ("centroid_x_px", "centroid_y_px"),
        ),
        FrameObservationBlock(
            1,
            0,
            np.arange(4, dtype=float),
            2.0 * np.eye(4) + shared,
            parts[0].observable_names,
        ),
        FrameObservationBlock(
            2,
            0,
            np.arange(4, dtype=float) + 1.0,
            6.0 * np.eye(4) + shared,
            parts[1].observable_names,
        ),
    )
    frame_metadata = {
        "0": {"known_fourier_features": None},
        "1": {
            "known_fourier_features": feature_metadata,
            "fourier_covariance_augmentation": augmentations[0].to_metadata(),
        },
        "2": {
            "known_fourier_features": feature_metadata,
            "fourier_covariance_augmentation": augmentations[1].to_metadata(),
        },
    }
    dataset = HeterogeneousInversionDataset(
        plan, blocks, metadata={"frame_metadata": frame_metadata}
    )
    noise = DetectorNoiseModel(1000.0)
    product = CheckpointObservableDataset(
        CheckpointedAcquisition(
            checkpoint_directory=Path(tmp_path),
            manifest_path=Path(tmp_path) / "manifest.json",
            manifest_sha256="fixture",
            records=(),
            pending_frame_ids=(),
        ),
        dataset,
        tuple(
            NoisyDetectorFrame(request, np.ones((4, 4)), 10 + request.frame_id, noise)
            for request in requests
        ),
        8,
    )

    aggregated = aggregate_checkpoint_observables_for_fit(product)
    expected_independent = 1.5 * np.eye(4)
    np.testing.assert_allclose(
        aggregated.blocks[0].covariance,
        expected_independent + shared,
        rtol=1e-12,
        atol=1e-12,
    )
    metadata = aggregated.metadata["aggregation"]["state_blocks"]["0"][
        "material"
    ][0]["fourier_covariance_augmentation"]
    np.testing.assert_allclose(metadata["total_covariance_observable2"], shared)
    assert "must_use_shared_input_latents" in metadata[
        "cross_state_calibration_correlation"
    ]
    statistical = strip_shared_fourier_calibration_covariance(aggregated)
    np.testing.assert_allclose(
        statistical.blocks[0].covariance,
        expected_independent,
        rtol=1e-12,
        atol=1e-12,
    )


def test_checkpoint_builder_adds_decomposed_fourier_covariance_and_global_flag(
    tmp_path,
):
    column = _column()
    detector = _detector()
    requests = (_request(1, "sample"), _request(2, "sample_repeat"))
    plan = AcquisitionPlan(
        channel_names=("IL1", "PL1"),
        states=(AcquisitionState("state", (0.2, -0.3), "nominal", 2),),
        reference_dacs=(0.2, -0.3),
    )

    def render(request):
        yy, xx = np.indices(detector.shape_yx, dtype=float)
        intensity = (
            2.0
            + 0.45 * np.cos(2.0 * np.pi * xx / 8.0)
            + 0.35 * np.cos(2.0 * np.pi * yy / 10.0)
        )
        return FramePrediction(
            request=request,
            render=GaussianRenderResult(
                np.sqrt(intensity).astype(complex), intensity, 0.0
            ),
            observables=extract_image_observables(intensity),
            interaction_route="local_action",
        )

    checkpoint = Path(tmp_path) / "fourier-checkpoint"
    render_frame_sequence_checkpointed(
        requests,
        detector,
        render,
        checkpoint,
        render_signature="fourier-covariance-test:v1",
    )
    parts = tuple(_imaging_part(request) for request in requests)
    augmentations = {
        request.frame_id: fourier_observable_covariance_augmentation(
            column, column.nominal_parameters, detector, part
        )
        for request, part in zip(requests, parts, strict=True)
    }
    feature = {
        "labels": parts[0].labels,
        "local_reciprocal_vectors_xy_per_m": (
            parts[0].world_reciprocal_vectors_xy_per_m
        ),
        "world_reciprocal_vectors_xy_per_m": (
            parts[0].world_reciprocal_vectors_xy_per_m
        ),
        "covariance_world_per_m2": parts[0].covariance_world_per_m2,
        "certificate_id": parts[0].certificate_id,
        "covariance_provenance": parts[0].covariance_provenance,
        "calibration_group_name": parts[0].calibration_group_name,
    }
    known = KnownFourierFeatureSet(**feature)
    common = dict(
        plan=plan,
        requests=requests,
        detector=detector,
        checkpoint_directory=checkpoint,
        render_signature="fourier-covariance-test:v1",
        default_noise_model=DetectorNoiseModel(
            dose_electrons=1.0e5,
            incident_intensity_sum=float(np.prod(detector.shape_yx) * 2.0),
        ),
        seed=91,
        bootstrap_replicates=12,
        fourier_features_by_family={"auto": known},
        moment_observables_by_measurement={"sample": (), "sample_repeat": ()},
    )
    statistical_only = build_heterogeneous_dataset_from_checkpoint(**common)
    augmented = build_heterogeneous_dataset_from_checkpoint(
        **common,
        fourier_covariance_augmentation_by_frame=augmentations,
    )

    for request in requests:
        plain = next(
            block
            for block in statistical_only.dataset.blocks
            if block.frame_id == request.frame_id
        )
        calibrated = next(
            block
            for block in augmented.dataset.blocks
            if block.frame_id == request.frame_id
        )
        np.testing.assert_allclose(plain.values, calibrated.values)
        np.testing.assert_allclose(
            calibrated.covariance - plain.covariance,
            augmentations[request.frame_id].total_covariance_observable2,
            rtol=1e-10,
            atol=1e-14,
        )
        metadata = augmented.dataset.metadata["frame_metadata"][
            str(request.frame_id)
        ]["fourier_covariance_augmentation"]
        assert metadata["certificate_id"] == "magical-certificate-v1"
    policy = augmented.dataset.metadata["fourier_calibration_covariance"]
    assert policy["per_frame_output_covariance_included"] is True
    assert policy["unpropagated_frame_ids"] == []
    assert len(policy["shared_input_priors"]) == 1
    assert "cross-state" in policy["global_fit_policy"]
    assert statistical_only.dataset.metadata["fourier_calibration_covariance"][
        "unpropagated_frame_ids"
    ] == [1, 2]

    shared_priors = shared_fourier_calibration_priors_from_dataset(
        augmented.dataset
    )
    assert tuple(prior.group_name for prior in shared_priors) == (
        "magical-si111-vectors",
    )
    statistical_for_latents = strip_shared_fourier_calibration_covariance(
        augmented.dataset
    )
    for plain, stripped in zip(
        statistical_only.dataset.blocks,
        statistical_for_latents.blocks,
        strict=True,
    ):
        np.testing.assert_allclose(stripped.covariance, plain.covariance)
    stripped_policy = statistical_for_latents.metadata[
        "fourier_calibration_covariance"
    ]
    assert stripped_policy["stripped_for_shared_latent_likelihood"] is True
    assert stripped_policy["stripped_group_names"] == ["magical-si111-vectors"]
