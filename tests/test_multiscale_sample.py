import copy
from pathlib import Path
import tomllib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.constants import energy2wavelength
from temgym_core.gaussian import make_gaussian
from temgym_core.multiscale_sample import (
    CubicActionField2D,
    CubicInterpolatedSample2D,
    MAGICALSampleAtlas,
    SampleLevel2D,
    SamplePyramid2D,
    SampleView,
    render_sample_view,
    select_sample_view,
    validate_sample_level,
    validate_sample_view_derivatives,
)


def _linear_level():
    x = np.linspace(-1.2, 1.2, 9)
    y = np.linspace(-0.6, 0.6, 7)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    pathlength = 3.0 * xx + 5.0 * yy
    log_amplitude = -0.2 + 0.1 * xx - 0.05 * yy
    return SampleLevel2D(pathlength, log_amplitude, x, y, name="linear")


def _constant_level(pathlength=2e-12, log_amplitude=np.log(0.7)):
    return SampleLevel2D.from_action_channels(
        np.full((8, 10), pathlength),
        np.full((8, 10), log_amplitude),
        pixel_size_yx_m=(0.2e-6, 0.15e-6),
        name="constant",
    )


@pytest.fixture(scope="module")
def nominal_atlas():
    return MAGICALSampleAtlas.nominal(
        field_width_m=352e-9,
        shape_yx=(640, 640),
        region_shape_yx=(1024, 1024),
        max_levels=6,
    )


def test_level_preserves_yx_contract_and_c2_explicit_derivatives():
    level = _linear_level()
    assert level.shape_yx == (7, 9)
    assert level.log_field_amplitude_yx is level.log_amplitude_yx
    view = SampleView.from_level(level)
    xy = jnp.array([0.17, -0.11])

    actions = view.evaluate_actions(xy)
    jet = view.action_jet(xy)

    np.testing.assert_allclose(actions[0], 3.0 * 0.17 + 5.0 * -0.11, atol=1e-12)
    np.testing.assert_allclose(actions[1], -0.2 + 0.1 * 0.17 - 0.05 * -0.11, atol=1e-12)
    np.testing.assert_allclose(jet.pathlength.gradient_xy, [3.0, 5.0], atol=1e-11)
    np.testing.assert_allclose(jet.log_amplitude.gradient_xy, [0.1, -0.05], atol=1e-11)
    np.testing.assert_allclose(jet.pathlength.hessian_xy, 0.0, atol=1e-10)

    validation = validate_sample_view_derivatives(view, xy)
    assert float(validation.gradient_max_abs_error) < 1e-10
    assert float(validation.hessian_max_abs_error) < 1e-9


def test_off_grid_quadratic_values_gradients_and_hessians_are_exact():
    x = np.linspace(-1.1, 1.3, 13)
    y = np.linspace(-0.9, 1.1, 11)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    pathlength = 0.7 + 2.0 * xx - 3.0 * yy + 1.5 * xx**2 + 0.8 * xx * yy - 0.4 * yy**2
    log_amplitude = -0.2 + 0.3 * xx + 0.5 * yy - 0.2 * xx**2 + 0.6 * xx * yy + 0.9 * yy**2
    view = SampleView.from_level(
        SampleLevel2D(pathlength, log_amplitude, x, y, name="quadratic")
    )
    qx, qy = 0.137, -0.224
    jet = view.action_jet(jnp.asarray((qx, qy)))

    expected_path = 0.7 + 2 * qx - 3 * qy + 1.5 * qx**2 + 0.8 * qx * qy - 0.4 * qy**2
    expected_log = -0.2 + 0.3 * qx + 0.5 * qy - 0.2 * qx**2 + 0.6 * qx * qy + 0.9 * qy**2
    np.testing.assert_allclose(
        (jet.pathlength.value, jet.log_amplitude.value),
        (expected_path, expected_log),
        atol=2e-11,
    )
    np.testing.assert_allclose(
        jet.pathlength.gradient_xy,
        (2 + 3 * qx + 0.8 * qy, -3 + 0.8 * qx - 0.8 * qy),
        atol=2e-10,
    )
    np.testing.assert_allclose(
        jet.log_amplitude.gradient_xy,
        (0.3 - 0.4 * qx + 0.6 * qy, 0.5 + 0.6 * qx + 1.8 * qy),
        atol=2e-10,
    )
    np.testing.assert_allclose(
        jet.pathlength.hessian_xy, ((3.0, 0.8), (0.8, -0.8)), atol=2e-9
    )
    np.testing.assert_allclose(
        jet.log_amplitude.hessian_xy, ((-0.4, 0.6), (0.6, 1.8)), atol=2e-9
    )


def test_pose_rotates_coordinates_gradients_and_hessians():
    level = _linear_level()
    centre = np.array([2.5, -1.7])
    theta = np.pi / 2
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    local_xy = np.array([0.2, -0.1])
    world_xy = centre + rotation @ local_xy
    view = SampleView.from_level(
        level, centre_xy_m=centre, rotation_rad=theta
    )

    jet = view.action_jet(jnp.asarray(world_xy))

    np.testing.assert_allclose(jet.pathlength.value, 3 * 0.2 + 5 * -0.1, atol=1e-11)
    np.testing.assert_allclose(
        jet.pathlength.gradient_xy, rotation @ np.array([3.0, 5.0]), atol=1e-10
    )
    np.testing.assert_allclose(jet.pathlength.hessian_xy, 0.0, atol=1e-9)


def test_tilt_foreshortens_coordinates_and_scales_projected_action():
    tilt_x = 0.31
    cosine = np.cos(tilt_x)
    local_xy = np.asarray((0.2, -0.1))
    world_xy = np.asarray((local_xy[0], cosine * local_xy[1]))
    view = SampleView.from_level(_linear_level(), tilt_xy_rad=(tilt_x, 0.0))

    jet = view.action_jet(jnp.asarray(world_xy))

    np.testing.assert_allclose(
        jet.pathlength.value,
        (3.0 * local_xy[0] + 5.0 * local_xy[1]) / cosine,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        jet.pathlength.gradient_xy,
        (3.0 / cosine, 5.0 / cosine**2),
        atol=1e-9,
    )
    assert isinstance(view, CubicInterpolatedSample2D)


def test_boundary_modes_are_explicit_and_have_correct_derivatives():
    level = _linear_level()
    query = jnp.array([2.0, 0.1])

    vacuum = SampleView.from_level(level, boundary="vacuum")
    np.testing.assert_allclose(vacuum.evaluate_actions(query), [0.0, 0.0], atol=0.0)
    np.testing.assert_allclose(
        vacuum.action_jet(query).pathlength.gradient_xy, 0.0, atol=0.0
    )

    clamped = SampleView.from_level(level, boundary="constant")
    clamped_jet = clamped.action_jet(query)
    np.testing.assert_allclose(clamped_jet.pathlength.value, 3 * 1.2 + 5 * 0.1, atol=1e-11)
    np.testing.assert_allclose(clamped_jet.pathlength.gradient_xy, [0.0, 5.0], atol=1e-10)

    nan_view = SampleView.from_level(level, boundary="nan")
    assert np.isnan(np.asarray(nan_view.evaluate_actions(query))).all()

    with pytest.raises(ValueError, match="boundary"):
        SampleView.from_level(level, boundary="wrap")


def test_periodic_boundary_wraps_values_and_derivatives():
    ny, nx = 8, 12
    pixel = 0.1
    x = (np.arange(nx) - 0.5 * (nx - 1)) * pixel
    y = (np.arange(ny) - 0.5 * (ny - 1)) * pixel
    period_x = nx * pixel
    period_y = ny * pixel
    xx, yy = np.meshgrid(x, y, indexing="xy")
    pathlength = (
        np.cos(2 * np.pi * xx / period_x)
        + 0.2 * np.sin(2 * np.pi * xx / period_x)
        + 0.3 * np.sin(2 * np.pi * yy / period_y)
    )
    level = SampleLevel2D(pathlength, np.zeros_like(pathlength), x, y)
    view = SampleView.from_level(level, boundary="periodic")
    point = jnp.array([0.17, -0.13])

    base = view.action_jet(point).pathlength
    wrapped = view.action_jet(point + jnp.array([period_x, -period_y])).pathlength

    np.testing.assert_allclose(wrapped.value, base.value, atol=1e-11)
    np.testing.assert_allclose(wrapped.gradient_xy, base.gradient_xy, atol=1e-10)
    np.testing.assert_allclose(wrapped.hessian_xy, base.hessian_xy, atol=1e-8)
    assert view.interpolation_halo_knots == 1

    # Exercise the actual pixel-edge seam, not merely two queries that modulo
    # to the same coordinate.  Value and the first two derivatives converge
    # from opposite sides, which is the C2 contract needed by Gaussian jets.
    xmin, xmax, _, _ = level.bounds_xyxy_m
    epsilon = 1e-8
    left = view.action_jet(jnp.array([xmin + epsilon, 0.13])).pathlength
    right = view.action_jet(jnp.array([xmax - epsilon, 0.13])).pathlength
    np.testing.assert_allclose(left.value, right.value, atol=2e-7)
    np.testing.assert_allclose(left.gradient_xy, right.gradient_xy, atol=2e-6)
    np.testing.assert_allclose(left.hessian_xy, right.hessian_xy, atol=2e-5)


def test_amplitude_phase_and_complex_constructors_use_physical_pathlength():
    voltage = 200e3
    wavelength = float(np.asarray(energy2wavelength(voltage)))
    phase_line = np.linspace(0.0, 4.0 * np.pi, 10)
    phase = np.broadcast_to(phase_line, (8, 10))
    amplitude = np.full_like(phase, 0.4)

    level = SampleLevel2D.from_amplitude_phase(
        amplitude,
        phase,
        pixel_size_yx_m=(1e-9, 2e-9),
        voltage=voltage,
    )
    np.testing.assert_allclose(level.pathlength_yx_m, phase * wavelength / (2 * np.pi))
    np.testing.assert_allclose(level.log_amplitude_yx, np.log(0.4))

    exact_named = SampleLevel2D.from_pathlength_and_log_amplitude(
        level.pathlength_yx_m,
        level.log_amplitude_yx,
        pixel_size_yx_m=(1e-9, 2e-9),
    )
    np.testing.assert_allclose(exact_named.pathlength_yx_m, level.pathlength_yx_m)
    exact_phase_named = SampleLevel2D.from_amplitude_and_unwrapped_phase(
        amplitude,
        phase,
        pixel_size_yx_m=(1e-9, 2e-9),
        voltage=voltage,
    )
    np.testing.assert_allclose(
        exact_phase_named.pathlength_yx_m, level.pathlength_yx_m
    )

    from_complex = SampleLevel2D.from_complex_transmission(
        amplitude * np.exp(1j * phase),
        pixel_size_yx_m=(1e-9, 2e-9),
        voltage=voltage,
        phase_unwrap="unwrap",
    )
    np.testing.assert_allclose(from_complex.pathlength_yx_m, level.pathlength_yx_m, atol=1e-24)

    with pytest.raises(ValueError, match="exactly one"):
        SampleLevel2D.from_amplitude_phase(
            amplitude, phase, pixel_size_yx_m=(1e-9, 2e-9)
        )
    with pytest.raises(ValueError, match="near_zero_threshold"):
        SampleLevel2D.from_complex_transmission(
            np.zeros_like(amplitude, dtype=np.complex128),
            pixel_size_yx_m=(1e-9, 2e-9),
            voltage=voltage,
            phase_unwrap="principal",
            near_zero_threshold=1e-6,
        )


def test_near_zero_amplitude_floor_keeps_actions_and_gaussian_jet_finite():
    amplitude = np.zeros((8, 10), dtype=np.float64)
    phase = np.zeros_like(amplitude)
    level = SampleLevel2D.from_amplitude_and_unwrapped_phase(
        amplitude,
        phase,
        pixel_size_yx_m=(1e-9, 1e-9),
        voltage=200e3,
        amplitude_floor=1e-40,
    )
    view = SampleView.from_level(level, log_floor=-50.0)
    jet = view.action_jet(jnp.zeros(2))
    assert np.isfinite(np.asarray(jet.log_amplitude.value)).all()
    assert np.isfinite(np.asarray(jet.log_amplitude.gradient_xy)).all()
    assert np.isfinite(np.asarray(jet.log_amplitude.hessian_xy)).all()

    beam = make_gaussian(voltage=200e3, waist_x=1e-9, waist_y=1e-9)
    output = jax.jit(lambda value: view(value))(beam)
    assert np.isfinite(np.asarray(output.amplitude)).all()
    assert np.isfinite(np.asarray(output.Q_inv)).all()
    assert 0.0 < float(np.abs(np.asarray(output.amplitude))) < 1e-15


def test_level_validation_rejects_ambiguous_or_invalid_rasters():
    with pytest.raises(ValueError, match="yx array shape"):
        SampleLevel2D(np.ones((5, 6)), np.ones((5, 6)), np.arange(5), np.arange(5))
    with pytest.raises(ValueError, match="strictly increasing"):
        SampleLevel2D(np.ones((5, 5)), np.ones((5, 5)), np.arange(5)[::-1], np.arange(5))
    with pytest.raises(ValueError, match="equal shape"):
        SampleLevel2D.from_action_channels(
            np.ones((6, 8)), np.ones((6, 7)), pixel_size_yx_m=(1.0, 1.0)
        )


def test_factor_two_pyramid_preserves_physical_pixel_edge_extent():
    finest = SampleLevel2D.from_action_channels(
        np.ones((64, 32)),
        -0.2 * np.ones((64, 32)),
        pixel_size_yx_m=(0.25, 0.5),
        name="finest",
    )
    pyramid = SamplePyramid2D.from_finest(finest, max_levels=4)

    assert [level.shape_yx for level in pyramid.levels] == [
        (64, 32),
        (32, 16),
        (16, 8),
        (8, 4),
    ]
    assert [level.pixel_size_yx_m for level in pyramid.levels] == [
        pytest.approx((0.25, 0.5)),
        pytest.approx((0.5, 1.0)),
        pytest.approx((1.0, 2.0)),
        pytest.approx((2.0, 4.0)),
    ]
    for level in pyramid.levels:
        np.testing.assert_allclose(level.bounds_xyxy_m, finest.bounds_xyxy_m)


def test_pyramid_antialiases_one_registered_low_frequency_landmark():
    x = (np.arange(128) - 63.5) * 1e-9
    y = (np.arange(96) - 47.5) * 1e-9
    xx, yy = np.meshgrid(x, y, indexing="xy")
    target = np.asarray((11.2e-9, -7.7e-9))
    landmark = np.exp(
        -0.5 * (((xx - target[0]) / 13e-9) ** 2 + ((yy - target[1]) / 9e-9) ** 2)
    )
    pyramid = SamplePyramid2D.from_finest(
        SampleLevel2D(landmark, -0.2 * landmark, x, y), max_levels=5
    )

    centroids = []
    for level in pyramid.levels:
        weights = np.maximum(level.pathlength_yx_m, 0.0)
        total = weights.sum()
        centroids.append(
            np.asarray(
                (
                    (weights * level.x_coords_m[None, :]).sum() / total,
                    (weights * level.y_coords_m[:, None]).sum() / total,
                )
            )
        )
        np.testing.assert_allclose(
            centroids[-1], target, atol=level.max_spacing_m
        )
    for centroid in centroids[1:]:
        np.testing.assert_allclose(
            centroid, centroids[0], atol=pyramid.levels[-1].max_spacing_m
        )
    assert np.max(pyramid.levels[1].pathlength_yx_m) < np.max(
        pyramid.levels[0].pathlength_yx_m
    )


def test_transfer_jacobian_selects_coarsest_adequate_lod_and_can_flag_finest():
    finest = SampleLevel2D.from_action_channels(
        np.zeros((64, 64)),
        np.zeros((64, 64)),
        pixel_size_yx_m=(0.25, 0.25),
    )
    pyramid = SamplePyramid2D.from_finest(finest, max_levels=4)

    view = select_sample_view(
        pyramid,
        transfer_jacobian_xy=np.eye(2),
        detector_shape_yx=(4, 4),
        detector_pixel_size_yx_m=(2.0, 2.0),
        texels_per_detector_pixel=2.0,
        margin_m=0.01,
        lens_wobble_margin_m=0.02,
        calibrated_input_margin_m=0.03,
        gaussian_support_margin_m=0.04,
        pose_uncertainty_margin_m=0.05,
        interpolation_halo_margin_m=0.06,
    )
    assert view.level_index == 2
    assert view.selection.meets_resolution
    assert view.selection.specimen_pixel_min_m == pytest.approx(2.0)
    assert view.selection.total_margin_m == pytest.approx(0.21)

    fine_view = select_sample_view(
        pyramid,
        transfer_jacobian_xy=np.eye(2),
        detector_shape_yx=(4, 4),
        detector_pixel_size_yx_m=(2.0, 2.0),
        gaussian_radius_min_m=0.3,
        samples_per_gaussian_radius=3.0,
    )
    assert fine_view.level_index == 0
    assert not fine_view.selection.meets_resolution

    with pytest.raises(ValueError, match="nonsingular"):
        select_sample_view(
            pyramid,
            transfer_jacobian_xy=np.zeros((2, 2)),
            detector_shape_yx=(4, 4),
            detector_pixel_size_yx_m=(1.0, 1.0),
        )


def test_periodic_lod_support_tiles_beyond_finite_source_bounds(nominal_atlas):
    level = SampleLevel2D.from_action_channels(
        np.zeros((8, 8)),
        np.zeros((8, 8)),
        pixel_size_yx_m=(0.25, 0.25),
    )
    pyramid = SamplePyramid2D((level,))
    selection_kwargs = dict(
        transfer_jacobian_xy=np.eye(2),
        detector_shape_yx=(16, 16),
        detector_pixel_size_yx_m=(1.0, 1.0),
    )

    periodic_view = select_sample_view(
        pyramid,
        boundary="periodic",
        **selection_kwargs,
    )
    vacuum_view = select_sample_view(
        pyramid,
        boundary="vacuum",
        **selection_kwargs,
    )

    requested = periodic_view.selection.local_bounds_xyxy_m
    stored = periodic_view.selection.selected_source_bounds_xyxy_m
    assert requested[1] - requested[0] > stored[1] - stored[0]
    assert requested[3] - requested[2] > stored[3] - stored[2]
    assert periodic_view.boundary == "periodic"
    assert periodic_view.selection.meets_resolution
    assert vacuum_view.boundary == "vacuum"
    assert (
        vacuum_view.selection.selected_spacing_m
        <= vacuum_view.selection.required_spacing_m
    )
    assert not vacuum_view.selection.meets_resolution

    automatic_lattice_view = select_sample_view(
        nominal_atlas,
        transfer_jacobian_xy=600_000.0 * np.eye(2),
        detector_shape_yx=(128, 128),
        detector_pixel_size_yx_m=(55e-6, 55e-6),
        margin_m=25e-9,
    )
    requested = automatic_lattice_view.selection.local_bounds_xyxy_m
    stored = automatic_lattice_view.selection.selected_source_bounds_xyxy_m
    assert requested[1] - requested[0] > stored[1] - stored[0]
    assert requested[3] - requested[2] > stored[3] - stored[2]
    assert (
        automatic_lattice_view.selection.selected_region_name
        == "silicon_lattice_region"
    )
    assert automatic_lattice_view.boundary == "periodic"
    assert (
        automatic_lattice_view.selection.selected_spacing_m
        <= automatic_lattice_view.selection.required_spacing_m * (1.0 + 1e-12)
    )
    assert automatic_lattice_view.selection.meets_resolution


def test_transfer_envelope_unions_rotated_fovs_and_uses_finest_pixel_bound():
    finest = SampleLevel2D.from_action_channels(
        np.zeros((256, 256)),
        np.zeros((256, 256)),
        pixel_size_yx_m=(1.0, 1.0),
    )
    pyramid = SamplePyramid2D.from_finest(finest, max_levels=6)
    nominal = select_sample_view(
        pyramid,
        transfer_jacobian_xy=np.eye(2),
        detector_shape_yx=(4, 4),
        detector_pixel_size_yx_m=(8.0, 8.0),
    )
    angle = np.deg2rad(31.0)
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    envelope = np.stack(
        (
            0.5 * rotation @ np.diag((1.08, 1.0 / 1.08)),
            2.0 * rotation.T @ np.diag((0.94, 1.0 / 0.94)),
        )
    )
    frozen = select_sample_view(
        pyramid,
        transfer_jacobian_xy=np.eye(2),
        transfer_jacobian_envelope_xy=envelope,
        detector_shape_yx=(4, 4),
        detector_pixel_size_yx_m=(8.0, 8.0),
    )

    nominal_bounds = nominal.selection.local_bounds_xyxy_m
    frozen_bounds = frozen.selection.local_bounds_xyxy_m
    assert frozen_bounds[1] - frozen_bounds[0] > nominal_bounds[1] - nominal_bounds[0]
    assert frozen_bounds[3] - frozen_bounds[2] > nominal_bounds[3] - nominal_bounds[2]
    assert frozen.selection.specimen_pixel_min_m < nominal.selection.specimen_pixel_min_m
    assert frozen.selection.selected_spacing_m <= nominal.selection.selected_spacing_m
    assert frozen.level_index <= nominal.level_index

    with pytest.raises(ValueError, match="shape"):
        select_sample_view(
            pyramid,
            transfer_jacobian_xy=np.eye(2),
            transfer_jacobian_envelope_xy=np.eye(2),
            detector_shape_yx=(4, 4),
            detector_pixel_size_yx_m=(8.0, 8.0),
        )
    with pytest.raises(ValueError, match="singular"):
        select_sample_view(
            pyramid,
            transfer_jacobian_xy=np.eye(2),
            transfer_jacobian_envelope_xy=np.zeros((1, 2, 2)),
            detector_shape_yx=(4, 4),
            detector_pixel_size_yx_m=(8.0, 8.0),
        )


def test_nominal_magical_atlas_is_certificate_driven_multiscale_and_asymmetric(
    nominal_atlas,
):
    atlas = nominal_atlas

    assert len(atlas.certificate.features) == 7
    assert len({feature.name for feature in atlas.certificate.features}) == 7
    assert {feature.kind for feature in atlas.certificate.features} == {
        "gabor",
        "fiducial",
    }
    finest = atlas.pyramid.levels[0]
    assert np.ptp(finest.pathlength_yx_m) > 0.0
    assert np.ptp(finest.log_amplitude_yx) > 0.0
    assert np.max(finest.log_amplitude_yx) <= 1e-15
    assert set(atlas.regions) == {
        "perforation",
        "multilayer_region_1",
        "multilayer_region_2",
        "multilayer_region_3",
        "multilayer_region_4",
        "silicon_lattice_region",
    }
    assert atlas.perforation.kind == "perforation"
    assert atlas.multilayer_region_4.layer_count == 4
    assert atlas.certificate.layers["multilayer_region_3"] == 3
    assert atlas.certificate.lattice_metadata["silicon_lattice_region"] == pytest.approx(
        0.31356e-9
    )
    assert set(atlas.region_pyramids) == set(atlas.regions)
    assert atlas.global_support_margin_m == pytest.approx(8e-9)
    assert atlas.certificate.multilayer_metadata == {
        "silicon_layer_width_m": pytest.approx(13e-9),
        "silicon_germanium_layer_width_m": pytest.approx(10e-9),
        "silicon_layer_width_uncertainty_m": pytest.approx(0.5e-9),
        "silicon_germanium_layer_width_uncertainty_m": pytest.approx(0.5e-9),
        "interface_width_m": pytest.approx(1e-9),
        "multilayer_period_m": pytest.approx(23e-9),
        "layer_phase_step_rad": pytest.approx(0.06),
        "layer_log_field_amplitude_step": pytest.approx(0.035),
    }

    region = atlas.multilayer_region_4
    multilayer = min(
        (
            level
            for level in atlas.pyramid_for_region(region.name).levels
            if level.bounds_xyxy_m[1] - level.bounds_xyxy_m[0]
            >= region.size_xy_m[0]
        ),
        key=lambda level: level.max_spacing_m,
    )
    centre_row = np.argmin(
        np.abs(multilayer.y_coords_m - atlas.multilayer_region_4.centre_xy_m[1])
    )
    cx = region.centre_xy_m[0]
    origin = cx - 0.5 * region.size_xy_m[0]
    sige_index = np.argmin(np.abs(multilayer.x_coords_m - (origin + 5e-9)))
    silicon_index = np.argmin(np.abs(multilayer.x_coords_m - (origin + 16.5e-9)))
    assert (
        multilayer.pathlength_yx_m[centre_row, sige_index]
        > multilayer.pathlength_yx_m[centre_row, silicon_index]
    )

    view = select_sample_view(
        atlas,
        transfer_jacobian_xy=100.0 * np.eye(2),
        detector_shape_yx=(32, 32),
        detector_pixel_size_yx_m=(1e-6, 1e-6),
        texels_per_detector_pixel=1.0,
        samples_per_feature_period=2.0,
    )
    assert view.selection.feature_names
    assert all(
        name in {feature.name for feature in atlas.certificate.features}
        for name in view.selection.feature_names
    )


def test_global_wide_and_compact_tiles_are_registered_views_of_one_scene(
    nominal_atlas,
):
    atlas = nominal_atlas
    region4 = atlas.multilayer_region_4
    region4_pyramid = atlas.pyramid_for_region(region4.name)
    wide = min(
        (
            level
            for level in region4_pyramid.levels
            if level.bounds_xyxy_m[1] - level.bounds_xyxy_m[0] > 250e-9
        ),
        key=lambda level: level.max_spacing_m,
    )
    compact = region4_pyramid.levels[0]
    global_view = SampleView.from_level(atlas.pyramid.levels[0])
    wide_view = SampleView.from_level(wide)
    compact_view = SampleView.from_level(compact)

    # Region 3 lies inside the wide region-4 tile.  Agreement here proves the
    # tile contains overlapping scene content rather than only redrawing its
    # nominal target region.
    overlap_point = jnp.asarray((5e-9, atlas.multilayer_region_3.centre_xy_m[1]))
    np.testing.assert_allclose(
        wide_view.evaluate_actions(overlap_point),
        global_view.evaluate_actions(overlap_point),
        rtol=2e-6,
        atol=1e-18,
    )

    # The compact and wide region-4 rasters use different pitches/support but
    # share physical coordinates and the same analytic certificate renderer.
    for x_m in (5e-9, 16.5e-9):
        point = jnp.asarray((x_m, region4.centre_xy_m[1]))
        np.testing.assert_allclose(
            compact_view.evaluate_actions(point),
            wide_view.evaluate_actions(point),
            rtol=2e-6,
            atol=1e-18,
        )


def test_toml_factory_consumes_certificate_pose_halo_and_vacuum_perforation():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "neoarm_synthetic_column.toml"
    )
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    atlas = MAGICALSampleAtlas.from_toml(
        config_path,
        shape_yx=(128, 128),
        region_shape_yx=(64, 64),
        authoritative_multilayer_shape_yx=(128, 128),
        max_levels=2,
        global_support_margin_m=0.0,
    )
    certificate = atlas.certificate

    assert certificate.certificate_source == config["sample"]["certificate"]["source"]
    assert certificate.silicon_layer_width_m == pytest.approx(13e-9)
    assert certificate.silicon_germanium_layer_width_m == pytest.approx(10e-9)
    assert certificate.silicon_layer_width_uncertainty_m == pytest.approx(0.25e-9)
    assert certificate.silicon_germanium_layer_width_uncertainty_m == pytest.approx(
        0.25e-9
    )
    assert certificate.interface_width_m == pytest.approx(1e-9)
    assert certificate.layer_phase_step_rad == pytest.approx(0.06)
    assert certificate.layer_log_field_amplitude_step == pytest.approx(0.035)
    assert certificate.silicon_lattice_metadata == {
        "spacing_m": pytest.approx(0.31356e-9),
        "spacing_standard_uncertainty_m": pytest.approx(0.00005e-9),
        "plane_hkl": (1, 1, 1),
        "zone_axis_uvw": (1, -1, 0),
        "certificate_source": config["sample"]["certificate"]["source"],
    }
    assert len(certificate.layer_boundaries) == 16
    first, second = certificate.layer_boundaries[:2]
    assert (first.index, first.left_material, first.right_material) == (
        1,
        "SiGe",
        "Si",
    )
    assert first.coordinate_m == pytest.approx(10e-9)
    assert first.standard_uncertainty_m == pytest.approx(0.25e-9)
    assert second.coordinate_m == pytest.approx(23e-9)
    assert second.standard_uncertainty_m == pytest.approx(np.sqrt(2) * 0.25e-9)
    assert certificate.layer_boundary_coordinates_m[-1] == pytest.approx(184e-9)
    assert atlas.perforation.centre_xy_m == pytest.approx((0.0, 0.0))
    assert certificate.perforation_diameter_m == pytest.approx(40e-9)
    assert atlas.pose_translation_xy_m == pytest.approx((2e-9, -1e-9))
    assert atlas.pose_rotation_rad == pytest.approx(np.deg2rad(3.25))
    assert atlas.pose_tilt_xy_rad == pytest.approx((1.2e-3, -0.8e-3))
    assert atlas.boundary_default == "vacuum"
    assert atlas.interpolation_halo_knots == 4

    finest = atlas.pyramid.levels[0]
    xx, yy = np.meshgrid(finest.x_coords_m, finest.y_coords_m, indexing="xy")
    strict_interior = np.hypot(xx, yy) < 0.95 * 20e-9
    assert strict_interior.any()
    np.testing.assert_array_equal(finest.pathlength_yx_m[strict_interior], 0.0)
    np.testing.assert_array_equal(finest.log_amplitude_yx[strict_interior], 0.0)

    perforation_view = select_sample_view(
        atlas,
        transfer_jacobian_xy=150_000.0 * np.eye(2),
        detector_shape_yx=(128, 128),
        detector_pixel_size_yx_m=(55e-6, 55e-6),
        region_name="perforation",
    )
    projection = np.asarray(perforation_view.projection_world_from_local)
    just_inside_local = np.asarray((0.999 * 20e-9, 0.0))
    just_inside_world = np.asarray(perforation_view.centre_xy_m) + projection @ just_inside_local
    inside_jet = perforation_view.action_jet(jnp.asarray(just_inside_world))
    np.testing.assert_array_equal(
        np.asarray((inside_jet.pathlength.value, inside_jet.log_amplitude.value)),
        0.0,
    )
    np.testing.assert_array_equal(inside_jet.pathlength.gradient_xy, 0.0)
    np.testing.assert_array_equal(inside_jet.pathlength.hessian_xy, 0.0)
    assert perforation_view.selection.crop_halo_knots == 4

    # The analytic guard is a C2-flat quintic, not a hard value mask.  Check
    # the limiting value/gradient/Hessian on both sides of the certified rim;
    # derivative units are normalized by the interface width.
    interface = certificate.interface_width_m
    epsilon = 1e-7 * interface
    rim_jets = []
    for local_radius in (20e-9 - epsilon, 20e-9, 20e-9 + epsilon):
        local_point = np.asarray((local_radius, 0.0))
        world_point = (
            np.asarray(perforation_view.centre_xy_m) + projection @ local_point
        )
        rim_jets.append(perforation_view.action_jet(jnp.asarray(world_point)))
    for channel_name in ("pathlength", "log_amplitude"):
        inside = getattr(rim_jets[0], channel_name)
        edge = getattr(rim_jets[1], channel_name)
        outside = getattr(rim_jets[2], channel_name)
        np.testing.assert_allclose(outside.value, inside.value, atol=1e-16)
        np.testing.assert_allclose(edge.value, inside.value, atol=1e-16)
        np.testing.assert_allclose(
            np.asarray(outside.gradient_xy) * interface,
            np.asarray(inside.gradient_xy) * interface,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(outside.hessian_xy) * interface**2,
            np.asarray(inside.hessian_xy) * interface**2,
            atol=1e-5,
        )

    outside_local = np.asarray((22e-9, 0.0))
    outside_world = np.asarray(perforation_view.centre_xy_m) + projection @ outside_local
    assert np.max(np.abs(np.asarray(perforation_view.evaluate_actions(outside_world)))) > 0.0

    periodic_view = select_sample_view(
        atlas,
        transfer_jacobian_xy=600_000.0 * np.eye(2),
        detector_shape_yx=(128, 128),
        detector_pixel_size_yx_m=(55e-6, 55e-6),
        region_name="silicon_lattice_region",
    )
    assert periodic_view.selection.crop_halo_knots == 0
    assert periodic_view.selection.interpolation_halo_knots == 1

    shifted_boundaries = copy.deepcopy(config)
    shifted_boundaries["sample"]["certificate"][
        "layer_boundary_coordinates_m"
    ][0] = 6e-9
    shifted_atlas = MAGICALSampleAtlas.from_config(
        shifted_boundaries,
        shape_yx=(32, 32),
        region_shape_yx=(64, 64),
        authoritative_multilayer_shape_yx=(16, 16),
        max_levels=1,
        global_support_margin_m=0.0,
    )
    region = atlas.multilayer_region_1
    probe = jnp.asarray(
        (
            region.centre_xy_m[0] - 0.5 * region.size_xy_m[0] + 8e-9,
            region.centre_xy_m[1],
        )
    )
    baseline_action = SampleView.from_level(
        atlas.pyramid_for_region(region.name).levels[0]
    ).evaluate_actions(probe)
    shifted_action = SampleView.from_level(
        shifted_atlas.pyramid_for_region(region.name).levels[0]
    ).evaluate_actions(probe)
    assert abs(float(shifted_action[1] - baseline_action[1])) > 1e-3

    unpaired_boundaries = copy.deepcopy(config)
    del unpaired_boundaries["sample"]["certificate"][
        "layer_boundary_standard_uncertainties_m"
    ]
    with pytest.raises(ValueError, match="provided together"):
        MAGICALSampleAtlas.from_config(unpaired_boundaries)

    incompatible = copy.deepcopy(config)
    incompatible["sample"]["certificate"]["perforation_diameter_m"] = 0.1e-3
    with pytest.raises(ValueError, match="incompatible"):
        MAGICALSampleAtlas.from_config(
            incompatible,
            shape_yx=(32, 32),
            region_shape_yx=(16, 16),
            authoritative_multilayer_shape_yx=(16, 16),
            max_levels=1,
        )


def test_rotated_anisotropic_views_cover_all_nominal_acquisition_bands(
    nominal_atlas,
):
    cases = (
        (
            np.asarray(((-37109.584846207195, -15561.516985291411),
                        (15402.309915947151, -36656.7477853378))),
            None,
            "multilayer_period",
        ),
        (
            np.asarray(((-71443.04168745542, -37054.28705885211),
                        (36663.29245362597, -70566.25414339473))),
            "multilayer_region_4",
            "multilayer_period",
        ),
        (
            np.asarray(((-125513.62255498732, -83770.4309355),
                        (82861.1734093722, -123960.12089751312))),
            "multilayer_region_4",
            "individual_layer_widths",
        ),
        (
            np.asarray(((-218243.08267694357, -208457.60941755935),
                        (206128.81091111834, -215497.6930701306))),
            "multilayer_region_4",
            "individual_layer_widths",
        ),
        (
            np.asarray(((-318559.6777843387, -512696.99036691384),
                        (506815.91337162483, -314405.78807917907))),
            "silicon_lattice_region",
            "silicon_lattice_vectors",
        ),
    )
    for jacobian, region_name, feature_family in cases:
        view = select_sample_view(
            nominal_atlas,
            transfer_jacobian_xy=jacobian,
            detector_shape_yx=(256, 256),
            detector_pixel_size_yx_m=(55e-6, 55e-6),
            sample_centre_xy_m=(2e-9, -1e-9),
            sample_rotation_rad=np.deg2rad(3.25),
            sample_tilt_xy_rad=(1.2e-3, -0.8e-3),
            gaussian_radius_min_m=3.4e-9,
            lens_wobble_margin_m=0.15e-9,
            calibrated_input_margin_m=1.10e-9,
            gaussian_support_margin_m=0.10e-9,
            pose_uncertainty_margin_m=0.10e-9,
            interpolation_halo_margin_m=0.05e-9,
        )
        selection = view.selection
        assert selection.feature_family == feature_family
        assert selection.selected_region_name == region_name
        assert selection.meets_resolution
        assert selection.selected_spacing_m <= selection.required_spacing_m * (
            1.0 + 2e-12
        )
        lx0, lx1, ly0, ly1 = selection.local_bounds_xyxy_m
        sx0, sx1, sy0, sy1 = selection.selected_source_bounds_xyxy_m
        assert sx0 <= lx0 and sx1 >= lx1 and sy0 <= ly0 and sy1 >= ly1

    multilayer_level = nominal_atlas.pyramid_for_region(
        "multilayer_region_2"
    ).levels[0]
    mx0, mx1, my0, my1 = multilayer_level.bounds_xyxy_m
    assert min(mx1 - mx0, my1 - my0) >= 260e-9
    lattice_level = nominal_atlas.pyramid_for_region(
        "silicon_lattice_region"
    ).levels[0]
    x0, x1, y0, y1 = lattice_level.bounds_xyxy_m
    assert min(x1 - x0, y1 - y0) >= 34e-9


def test_magical_routes_40k_to_600k_fields_with_stage_translation_and_fixed_crop(
    nominal_atlas,
):
    expected = {
        40_000: ("multilayer_period", None),
        80_000: ("multilayer_period", "multilayer_region_4"),
        150_000: ("individual_layer_widths", "multilayer_region_4"),
        300_000: ("individual_layer_widths", "multilayer_region_4"),
        400_000: ("individual_layer_widths", "multilayer_region_4"),
        600_000: ("silicon_lattice_vectors", "silicon_lattice_region"),
    }
    for magnification, (family, region_name) in expected.items():
        view = select_sample_view(
            nominal_atlas,
            transfer_jacobian_xy=magnification * np.eye(2),
            detector_shape_yx=(256, 256),
            detector_pixel_size_yx_m=(55e-6, 55e-6),
            interpolation_halo_margin_m=1e-9,
        )
        selection = view.selection

        assert selection.feature_family == family
        assert selection.selected_region_name == region_name
        assert selection.meets_resolution
        assert selection.meets_feature_sampling
        assert 3.0 <= selection.projected_feature_pixels <= 96.0
        assert selection.visible_feature_intervals >= 2.0
        assert selection.total_margin_m == pytest.approx(1e-9)
        assert dict(selection.margin_components_m)["interpolation_halo"] == pytest.approx(
            1e-9
        )
        np.testing.assert_allclose(
            view.world_to_local(jnp.zeros(2)),
            selection.field_centre_local_xy_m,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            view.centre_xy_m, selection.sample_translation_xy_m, atol=1e-15
        )
        if region_name is not None:
            assert region_name in selection.region_names
        if magnification == 600_000:
            assert view.boundary == "periodic"
            assert view.interpolation_halo_knots == 1
            assert selection.interpolation_halo_knots == 1
            assert selection.selected_spacing_m == pytest.approx(0.31356e-9 / 8)
        else:
            assert view.boundary == "vacuum"
        if magnification in (150_000, 300_000, 400_000):
            # The frozen interpolation crop is substantially smaller than the
            # 2688-square authoritative source tile.
            assert max(view.interpolator.f.shape[:2]) < 1100


def test_render_sample_view_returns_yx_complex_transmission():
    level = _constant_level()
    view = SampleView.from_level(level)
    x = level.x_coords_m[2:7]
    y = level.y_coords_m[1:5]
    wavelength = 2.5e-12

    render = render_sample_view(view, x, y, wavelength_m=wavelength)

    assert render.pathlength_yx_m.shape == (4, 5)
    np.testing.assert_allclose(render.pathlength_yx_m, 2e-12, atol=1e-24)
    np.testing.assert_allclose(render.log_amplitude_yx, np.log(0.7), atol=1e-12)
    expected = 0.7 * np.exp(1j * 2 * np.pi * 2e-12 / wavelength)
    np.testing.assert_allclose(render.transmission_yx, expected, rtol=1e-11)


def test_sample_view_uses_explicit_jet_in_scalar_and_batched_gaussian_action():
    level = _constant_level()
    view = SampleView.from_level(level)
    assert isinstance(view, CubicActionField2D)
    beam = make_gaussian(
        x=jnp.array([-0.1e-6, 0.1e-6]),
        y=0.0,
        voltage=200e3,
        waist_x=20e-9,
        waist_y=25e-9,
    )

    output = jax.jit(lambda value: view(value))(beam)

    np.testing.assert_allclose(np.abs(output.amplitude), 0.7, rtol=1e-11)
    np.testing.assert_allclose(output.pathlength, 2e-12, atol=1e-24)
    np.testing.assert_allclose(output.dx, 0.0, atol=1e-17)
    np.testing.assert_allclose(output.dy, 0.0, atol=1e-17)


def test_sampling_diagnostic_accepts_constant_sample_action():
    view = SampleView.from_level(_constant_level())
    beam = make_gaussian(
        x=jnp.array([-0.1e-6, 0.1e-6]),
        y=0.0,
        voltage=200e3,
        waist_x=20e-9,
        waist_y=20e-9,
    )

    diagnostic = view.sampling_diagnostic(
        beam, phase_tol=1e-10, amplitude_tol=1e-10
    )

    assert not np.asarray(diagnostic.needs_refinement).any()
    np.testing.assert_allclose(diagnostic.phase_max_abs, 0.0, atol=1e-10)
    np.testing.assert_allclose(diagnostic.amplitude_max_abs, 0.0, atol=1e-10)
    validation = validate_sample_level(view, beam)
    assert validation.use_local_action
    assert validation.recommended_route == "local_action"
    assert validation.flagged_beams == 0


def test_validation_routes_unresolved_sample_to_plane_reexpansion():
    voltage = 200e3
    x = np.linspace(-50e-9, 50e-9, 65)
    y = np.linspace(-50e-9, 50e-9, 65)
    xx, _ = np.meshgrid(x, y, indexing="xy")
    phase = 2.0 * np.sin(2 * np.pi * xx / 12e-9)
    level = SampleLevel2D.from_amplitude_and_unwrapped_phase(
        np.ones_like(phase),
        phase,
        pixel_size_yx_m=(y[1] - y[0], x[1] - x[0]),
        voltage=voltage,
    )
    view = SampleView.from_level(level)
    beam = make_gaussian(
        voltage=voltage,
        waist_x=15e-9,
        waist_y=15e-9,
    )

    validation = view.validate(beam, phase_tol=0.1, support_radius=2.0)

    assert not validation.use_local_action
    assert validation.recommended_route == "sample_plane_reexpansion"
    assert validation.flagged_beams == 1


def test_sample_view_actions_are_jittable_and_pose_fields_remain_differentiable():
    view = SampleView.from_level(_linear_level())
    points = jnp.array([[0.1, 0.2], [-0.3, 0.05]])

    eager = view.evaluate_actions(points)
    compiled = jax.jit(view.evaluate_actions)(points)

    np.testing.assert_allclose(compiled, eager, rtol=1e-12, atol=1e-12)
    leaves = jax.tree.leaves(view)
    assert any(np.asarray(leaf).shape == (2,) for leaf in leaves)
