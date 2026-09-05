import numpy as np
import pytest

from temgym_core.diffraction_metrology import (
    DiffractionSpotRegistrationError,
    extract_registered_bragg_spot_offsets,
)


def _spots(
    shape=(129, 151),
    *,
    centre=(63.4, 74.1),
    offsets=((18.35, -7.2), (-11.6, -23.4)),
    sigma=1.45,
    background=8.0,
):
    yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    image = np.full(shape, background, dtype=float)
    for position, amplitude in ((centre, 900.0),):
        image += amplitude * np.exp(
            -((yy - position[0]) ** 2 + (xx - position[1]) ** 2) / (2.0 * sigma**2)
        )
    for index, offset in enumerate(np.asarray(offsets, dtype=float)):
        for sign in (-1.0, 1.0):
            position = np.asarray(centre) + sign * offset
            image += (380.0 + 70.0 * index) * np.exp(
                -((yy - position[0]) ** 2 + (xx - position[1]) ** 2)
                / (2.0 * sigma**2)
            )
    return image, np.asarray(offsets, dtype=float)


def test_friedel_pair_measurement_recovers_subpixel_yx_offsets_and_schema():
    image, offsets = _spots()
    result = extract_registered_bragg_spot_offsets(
        image,
        offsets + np.asarray(((0.35, -0.25), (-0.2, 0.3))),
        ("g0", "g1"),
        search_radius_px=5.5,
    )
    np.testing.assert_allclose(result.offsets_yx_px, offsets, atol=0.035)
    np.testing.assert_allclose(result.values, offsets.ravel(), atol=0.035)
    assert result.observable_names == (
        "bragg_g0_y_px",
        "bragg_g0_x_px",
        "bragg_g1_y_px",
        "bragg_g1_x_px",
    )
    assert not result.offsets_yx_px.flags.writeable


def test_detector_noise_bootstrap_keeps_fixed_registration_and_is_unbiased():
    mean, offsets = _spots(background=3.0)
    values = []
    for seed in range(48):
        draw = np.random.default_rng(seed).poisson(mean)
        values.append(
            extract_registered_bragg_spot_offsets(
                draw,
                offsets,
                ("g0", "g1"),
                search_radius_px=5.5,
            ).values
        )
    values = np.asarray(values)
    np.testing.assert_allclose(values.mean(axis=0), offsets.ravel(), atol=0.08)
    covariance = np.cov(values, rowvar=False)
    assert covariance.shape == (4, 4)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12
    assert np.all(np.diag(covariance) > 0.0)


def test_registration_window_uses_friedel_difference_not_direct_beam_position():
    image, offsets = _spots(centre=(59.7, 80.2))
    result = extract_registered_bragg_spot_offsets(
        image,
        offsets,
        ("g0", "g1"),
        detector_centre_yx_px=(60.0, 80.0),
        search_radius_px=5.5,
    )
    np.testing.assert_allclose(result.offsets_yx_px, offsets, atol=0.04)
    np.testing.assert_allclose(result.direct_beam_centroid_yx_px, (59.7, 80.2), atol=0.04)


def test_edge_intersection_and_missing_signal_fail_closed():
    image, offsets = _spots()
    with pytest.raises(DiffractionSpotRegistrationError, match="detector edge"):
        extract_registered_bragg_spot_offsets(
            image,
            np.asarray(((62.0, 0.0),)),
            ("edge",),
            search_radius_px=5.0,
        )
    with pytest.raises(DiffractionSpotRegistrationError, match="insufficient"):
        extract_registered_bragg_spot_offsets(
            np.ones((65, 65)),
            np.asarray(((15.0, 0.0),)),
            ("empty",),
            search_radius_px=4.0,
        )


def test_input_validation_rejects_wrong_order_shape_and_nonfinite_values():
    image, offsets = _spots()
    with pytest.raises(ValueError, match="shape"):
        extract_registered_bragg_spot_offsets(image, offsets.ravel(), ("g0", "g1"))
    with pytest.raises(ValueError, match="match"):
        extract_registered_bragg_spot_offsets(image, offsets, ("only_one",))
    damaged = image.copy()
    damaged[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        extract_registered_bragg_spot_offsets(damaged, offsets, ("g0", "g1"))
