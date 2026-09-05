import numpy as np
import pytest
from scipy import integrate

from temgym_core.grid_wave_reference import (
    GridWaveOptions,
    GridWaveSamplingError,
    assess_grid_reference_convergence,
    certify_radial_symplectic_map,
    poisson_read_total_standard_error,
    round_solenoid_grid_reference,
    scaled_collins_czt_1d,
    scaled_collins_czt_2d,
    simpson_uniform_weights,
)
from temgym_core.neoarm_calibration import collins_propagate_2d


@pytest.mark.parametrize(
    "radial_abcd",
    (
        (107047.63949368332, 246.60311799931839, 355171.47527343163, 818.20014566518648),
        (181925.87841668283, -168.8828205605983, 603620.72044943518, -560.34451890805269),
        (175541.42310797924, -153.23952002419668, 582437.45493211993, -508.44087656171996),
        (180284.3846039229, -167.35900527755234, 598150.29301235499, -555.26626593286176),
        (173957.57431962743, -151.8568859273438, 577159.13447384047, -503.83312246398413),
        (179297.85196695948, -166.45217536477037, 594901.09362777718, -552.27979072204425),
        (120131.25247618285, 164.88783656538575, 398589.34516076517, 547.08940801730591),
    ),
)
def test_executed_ill_conditioned_maps_pass_only_by_backward_error(radial_abcd):
    original = tuple(radial_abcd)
    certificate = certify_radial_symplectic_map(
        radial_abcd,
        absolute_tolerance=1e-8,
    )

    assert certificate.passed
    assert certificate.certification_basis == "floating_point_backward_error"
    assert certificate.determinant_error > certificate.absolute_tolerance
    assert certificate.determinant_error <= (
        certificate.absolute_tolerance
        + certificate.determinant_evaluation_roundoff_bound
    )
    assert certificate.normalized_backward_error < np.finfo(float).eps
    # Certification is provenance, not a hidden symplectic projection.
    assert certificate.radial_abcd == original


def test_backward_error_certificate_still_rejects_resolvable_defects():
    well_scaled = certify_radial_symplectic_map(
        (1.0, 0.0, 0.0, 1.0 + 5e-8),
        absolute_tolerance=1e-8,
    )
    assert not well_scaled.passed
    assert well_scaled.certification_basis == "failed_determinant_residual"

    executed = np.asarray(
        (180284.3846039229, -167.35900527755234, 598150.29301235499, -555.26626593286176)
    )
    executed[3] += 1e-6
    ill_scaled_but_resolvably_wrong = certify_radial_symplectic_map(
        executed,
        absolute_tolerance=1e-8,
    )
    assert not ill_scaled_but_resolvably_wrong.passed
    assert ill_scaled_but_resolvably_wrong.determinant_error > 0.1


def test_reference_convergence_preserves_absolute_count_changes():
    previous = np.full((3, 5), 100.0)
    refined = np.full((3, 5), 104.0)
    noise_sigma = np.full((3, 5), 20.0)

    diagnostics = assess_grid_reference_convergence(
        previous,
        refined,
        noise_sigma,
        observable_residuals_in_sigma={"intensity": 0.2},
        maximum_detector_rms_in_noise_sigma=0.1,
        maximum_observable_residual_in_sigma=0.1,
    )

    assert diagnostics.detector_rms_in_noise_sigma == pytest.approx(0.2)
    assert diagnostics.observable_residual_map == {"intensity": 0.2}
    assert diagnostics.failure_reasons == (
        "detector_rms",
        "observable:intensity",
    )
    assert diagnostics.passed is False


def test_reference_convergence_accepts_only_when_both_gates_pass():
    previous = np.asarray(((10.0, 20.0), (30.0, 40.0)))
    refined = previous + np.asarray(((0.005, -0.005), (0.002, -0.002)))
    diagnostics = assess_grid_reference_convergence(
        previous,
        refined,
        np.ones_like(previous),
        observable_residuals_in_sigma={"scale": 0.004, "intensity": 0.008},
        maximum_detector_rms_in_noise_sigma=0.01,
        maximum_observable_residual_in_sigma=0.01,
    )

    assert diagnostics.detector_rms_in_noise_sigma == pytest.approx(
        np.sqrt(np.mean((previous - refined) ** 2))
    )
    assert diagnostics.observable_residuals_in_sigma == (
        ("intensity", 0.008),
        ("scale", 0.004),
    )
    assert diagnostics.failure_reasons == ()
    assert diagnostics.passed is True


def test_analytic_poisson_read_error_preserves_the_raw_total():
    expected = np.asarray(((10.0, 20.0), (30.0, 40.0)))
    sigma = poisson_read_total_standard_error(expected, read_noise_e=0.6)
    assert sigma == pytest.approx(np.sqrt(expected.sum() + expected.size * 0.6**2))

    # Scaling the raw expectation changes its uncertainty; no flux
    # normalisation is hidden in the helper.
    doubled = poisson_read_total_standard_error(2.0 * expected, read_noise_e=0.6)
    assert doubled > sigma


@pytest.mark.parametrize(
    ("expected", "read_noise", "match"),
    (
        (np.ones(3), 0.0, "two-dimensional"),
        (np.full((2, 2), np.nan), 0.0, "finite"),
        (np.asarray(((1.0, -1.0), (2.0, 3.0))), 0.0, "non-negative"),
        (np.ones((2, 2)), -0.1, "non-negative"),
        (np.zeros((2, 2)), 0.0, "positive"),
    ),
)
def test_analytic_poisson_read_error_rejects_invalid_inputs(
    expected, read_noise, match
):
    with pytest.raises(ValueError, match=match):
        poisson_read_total_standard_error(expected, read_noise_e=read_noise)


@pytest.mark.parametrize(
    ("previous", "refined", "noise", "match"),
    (
        (np.ones(3), np.ones((2, 2)), np.ones((2, 2)), "two-dimensional"),
        (np.ones((2, 2)), np.ones((2, 2)), np.ones((3, 2)), "match"),
        (np.ones((2, 2)), np.full((2, 2), np.nan), np.ones((2, 2)), "finite"),
        (np.ones((2, 2)), np.ones((2, 2)), np.zeros((2, 2)), "strictly positive"),
    ),
)
def test_reference_convergence_rejects_invalid_arrays(previous, refined, noise, match):
    with pytest.raises(ValueError, match=match):
        assess_grid_reference_convergence(previous, refined, noise)


def _direct_collins_1d(field, x_in, x_out, abcd, wavelength):
    A, B, _C, D = abcd
    kernel = np.exp(
        1j
        * np.pi
        * (
            A * x_in[None, :] ** 2
            - 2.0 * x_out[:, None] * x_in[None, :]
            + D * x_out[:, None] ** 2
        )
        / (wavelength * B)
    ) / np.sqrt(1j * wavelength * B + 0j)
    return kernel @ (field * simpson_uniform_weights(len(x_in), x_in[1] - x_in[0]))


def _normalised_gaussian(x, y, *, centre=(0.0, 0.0), waist=(0.2, 0.16)):
    xx, yy = np.meshgrid(x, y, indexing="xy")
    amplitude = np.sqrt(2.0 / (np.pi * waist[0] * waist[1]))
    return amplitude * np.exp(
        -((xx - centre[0]) / waist[0]) ** 2
        - ((yy - centre[1]) / waist[1]) ** 2
    )


@pytest.mark.parametrize("n", range(2, 11))
def test_linear_simpson_weights_match_scipy_for_even_and_odd_counts(n):
    x = np.linspace(-0.7, 1.1, n)
    expected = integrate.simpson(np.eye(n), x=x, axis=1)
    np.testing.assert_allclose(
        simpson_uniform_weights(n, x[1] - x[0]), expected, rtol=1e-14, atol=1e-14
    )


def test_scaled_czt_1d_matches_direct_simpson_with_absolute_prefactor():
    wavelength = 0.037
    abcd = (0.8, 0.73, -0.410958904109589, 0.875)
    # AD-BC=1.  Offset, unequal grids exercise every CZT linear phase.
    x_in = np.linspace(-0.61, 0.83, 47)
    x_out = np.linspace(-0.77, 0.58, 39)
    field = np.exp(-((x_in - 0.08) / 0.21) ** 2) * np.exp(
        0.7j * x_in + 0.2j * x_in**2
    )
    expected = _direct_collins_1d(field, x_in, x_out, abcd, wavelength)
    actual = scaled_collins_czt_1d(
        field, x_in, x_out, abcd=abcd, wavelength_m=wavelength
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


def test_scaled_czt_supports_signed_b_with_the_same_branch_as_direct_collins():
    wavelength = 0.09
    abcd = (1.0, -0.65, 0.0, 1.0)
    x_in = np.linspace(-0.55, 0.62, 42)
    x_out = np.linspace(-0.71, 0.53, 37)
    field = np.exp(-12.0 * (x_in + 0.04) ** 2) * np.exp(-0.5j * x_in)
    expected = _direct_collins_1d(field, x_in, x_out, abcd, wavelength)
    actual = scaled_collins_czt_1d(
        field, x_in, x_out, abcd=abcd, wavelength_m=wavelength
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


def test_scaled_czt_2d_matches_existing_direct_simpson_reference():
    wavelength = 0.061
    abcd_x = (0.9, 0.8, -0.2625, 0.8777777777777778)
    abcd_y = (1.1, 0.7, 0.1571428571428573, 1.0)
    x_in = np.linspace(-0.5, 0.6, 19)
    y_in = np.linspace(-0.4, 0.55, 17)
    x_out = np.linspace(-0.7, 0.65, 23)
    y_out = np.linspace(-0.62, 0.58, 21)
    xx, yy = np.meshgrid(x_in, y_in, indexing="xy")
    field = np.exp(-7.0 * xx**2 - 9.0 * yy**2) * np.exp(
        0.3j * xx - 0.2j * yy
    )
    expected = collins_propagate_2d(
        field,
        x_in,
        y_in,
        x_out,
        y_out,
        abcd_x=abcd_x,
        abcd_y=abcd_y,
        wavelength_m=wavelength,
    )
    actual = scaled_collins_czt_2d(
        field,
        x_in,
        y_in,
        x_out,
        y_out,
        abcd_x=abcd_x,
        abcd_y=abcd_y,
        wavelength_m=wavelength,
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=3e-11)


def test_scaled_czt_free_space_gaussian_has_correct_width_and_power():
    wavelength = 0.08
    distance = 0.9
    waist = (0.18, 0.14)
    x_in = np.linspace(-1.25, 1.25, 401)
    y_in = np.linspace(-1.05, 1.05, 337)
    field = _normalised_gaussian(x_in, y_in, waist=waist)
    rayleigh = np.pi * np.asarray(waist) ** 2 / wavelength
    propagated_waist = np.asarray(waist) * np.sqrt(1.0 + (distance / rayleigh) ** 2)
    x_out = np.linspace(-5.5 * propagated_waist[0], 5.5 * propagated_waist[0], 421)
    y_out = np.linspace(-5.5 * propagated_waist[1], 5.5 * propagated_waist[1], 389)
    output = scaled_collins_czt_2d(
        field,
        x_in,
        y_in,
        x_out,
        y_out,
        abcd_x=(1.0, distance, 0.0, 1.0),
        wavelength_m=wavelength,
    )
    intensity = np.abs(output) ** 2
    wx = simpson_uniform_weights(x_out.size, x_out[1] - x_out[0])
    wy = simpson_uniform_weights(y_out.size, y_out[1] - y_out[0])
    power = float(wy @ intensity @ wx)
    xx, yy = np.meshgrid(x_out, y_out, indexing="xy")
    sigma_x2 = float(wy @ (intensity * xx**2) @ wx / power)
    sigma_y2 = float(wy @ (intensity * yy**2) @ wx / power)
    assert power == pytest.approx(1.0, rel=3e-5)
    # For intensity exp(-2*x²/w²), variance is w²/4.
    assert 2.0 * np.sqrt(sigma_x2) == pytest.approx(propagated_waist[0], rel=4e-4)
    assert 2.0 * np.sqrt(sigma_y2) == pytest.approx(propagated_waist[1], rel=4e-4)


def _centroid(x, y, intensity):
    total = float(np.sum(intensity))
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.asarray((np.sum(intensity * xx) / total, np.sum(intensity * yy) / total))


def test_conjugate_round_map_rotates_then_applies_detector_affine():
    x_in = np.linspace(-0.9, 0.9, 361)
    y_in = np.linspace(-0.8, 0.8, 321)
    centre = np.asarray((0.13, -0.09))
    field = _normalised_gaussian(x_in, y_in, centre=centre, waist=(0.11, 0.09))
    detector_x = np.linspace(-0.9, 0.9, 361)
    detector_y = np.linspace(-0.8, 0.8, 321)
    angle = np.deg2rad(27.0)
    affine = np.asarray(((1.05, 0.03), (-0.02, 0.92)))
    magnification = 1.35
    result = round_solenoid_grid_reference(
        field,
        x_in,
        y_in,
        detector_x,
        detector_y,
        radial_abcd=(magnification, 0.0, 0.0, 1.0 / magnification),
        cumulative_rotation_rad=angle,
        detector_affine_xy=affine,
        wavelength_m=0.05,
        options=GridWaveOptions(maximum_input_edge_power_fraction=1e-8),
    )
    expected_column = np.asarray(
        (
            np.cos(angle) * centre[0] - np.sin(angle) * centre[1],
            np.sin(angle) * centre[0] + np.cos(angle) * centre[1],
        )
    ) * magnification
    expected_detector = affine @ expected_column
    actual = _centroid(detector_x, detector_y, result.intensity_yx)
    pixel = max(detector_x[1] - detector_x[0], detector_y[1] - detector_y[0])
    np.testing.assert_allclose(actual, expected_detector, atol=0.35 * pixel)
    assert result.diagnostics.route == "conjugate"
    assert result.diagnostics.passed
    assert result.diagnostics.input_power == pytest.approx(1.0, rel=2e-8)


def test_dispatcher_routes_nonzero_b_to_czt_and_reports_power():
    wavelength = 0.08
    x_in = np.linspace(-1.2, 1.2, 301)
    y_in = np.linspace(-1.0, 1.0, 251)
    field = _normalised_gaussian(x_in, y_in, waist=(0.18, 0.15))
    detector_x = np.linspace(-1.1, 1.1, 281)
    detector_y = np.linspace(-0.95, 0.95, 243)
    result = round_solenoid_grid_reference(
        field,
        x_in,
        y_in,
        detector_x,
        detector_y,
        radial_abcd=(1.0, 0.8, 0.0, 1.0),
        cumulative_rotation_rad=0.0,
        detector_affine_xy=np.eye(2),
        wavelength_m=wavelength,
        options=GridWaveOptions(
            output_oversampling=1.0,
            interpolation_halo_pixels=0,
            czt_maximum_kernel_phase_step_rad=4.0,
            maximum_input_edge_power_fraction=1e-8,
            require_power_conservation=True,
            maximum_power_relative_error=2e-4,
        ),
    )
    assert result.diagnostics.route == "czt"
    assert result.diagnostics.b_displacement_px > 1e-3
    assert result.diagnostics.power_relative_error < 2e-4
    assert result.diagnostics.passed
    direct = scaled_collins_czt_2d(
        field,
        x_in,
        y_in,
        detector_x,
        detector_y,
        abcd_x=(1.0, 0.8, 0.0, 1.0),
        wavelength_m=wavelength,
    )
    relative_field_error = float(
        np.linalg.norm(result.field_yx - direct) / np.linalg.norm(direct)
    )
    relative_intensity_error = float(
        np.linalg.norm(result.intensity_yx - np.abs(direct) ** 2)
        / np.linalg.norm(np.abs(direct) ** 2)
    )
    # The remaining error is cubic resampling from the carrier-reduced
    # intermediate lattice onto detector coordinates, not Collins aliasing.
    assert relative_field_error < 2e-7
    assert relative_intensity_error < 2e-7


def test_scaled_fresnel_fft_factorization_matches_direct_collins():
    wavelength = 0.04
    A, B, C = 2.0, 2.0, 0.15
    D = (1.0 + B * C) / A
    x = np.linspace(-0.7, 0.7, 97)
    y = x.copy()
    field = _normalised_gaussian(x, y, waist=(0.16, 0.16))
    detector_x = np.linspace(-1.3, 1.3, 91)
    detector_y = detector_x.copy()
    options = GridWaveOptions(
        maximum_input_edge_power_fraction=1e-12,
        fft_padding_factor=6.0,
        fft_maximum_nyquist_shift_fraction_of_padding=3.0,
        fft_maximum_paraxial_angle_rad=3.0,
        fft_minimum_samples_per_detector_pixel=0.5,
        output_oversampling=1.0,
        interpolation_halo_pixels=0,
        maximum_intermediate_points=1_000_000,
    )
    actual = round_solenoid_grid_reference(
        field,
        x,
        y,
        detector_x,
        detector_y,
        radial_abcd=(A, B, C, D),
        cumulative_rotation_rad=0.0,
        detector_affine_xy=np.eye(2),
        wavelength_m=wavelength,
        route="scaled_fresnel_fft",
        options=options,
    )
    expected = collins_propagate_2d(
        field,
        x,
        y,
        detector_x,
        detector_y,
        abcd_x=(A, B, C, D),
        abcd_y=(A, B, C, D),
        wavelength_m=wavelength,
    )
    # The transfer-function Fresnel form and Collins kernel may differ by a
    # coordinate-independent branch phase; compare after removing it.
    phase = np.vdot(actual.field_yx, expected)
    aligned = actual.field_yx * np.exp(1j * np.angle(phase))
    np.testing.assert_allclose(aligned, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        actual.intensity_yx, np.abs(expected) ** 2, rtol=2e-5, atol=2e-6
    )
    assert actual.diagnostics.route == "scaled_fresnel_fft"


@pytest.mark.parametrize("b_sign", (-1.0, 1.0))
def test_actual_high_a_il3_regime_routes_to_scaled_fresnel_fft(b_sign):
    wavelength = 2.5079e-12
    A = 1.5e5
    B = b_sign * 8.3e-3
    C = -20.0
    D = (1.0 + B * C) / A
    spacing = 0.36e-9
    sample_x = (np.arange(129) - 64.0) * spacing
    sample_y = sample_x.copy()
    waist = 4.0e-9
    field = _normalised_gaussian(
        sample_x, sample_y, waist=(waist, waist)
    )
    detector_pixel = 55.0e-6
    detector_x = (np.arange(256) - 127.5) * detector_pixel
    detector_y = detector_x.copy()
    result = round_solenoid_grid_reference(
        field,
        sample_x,
        sample_y,
        detector_x,
        detector_y,
        radial_abcd=(A, B, C, D),
        cumulative_rotation_rad=0.17,
        detector_affine_xy=np.eye(2),
        wavelength_m=wavelength,
        options=GridWaveOptions(
            maximum_input_edge_power_fraction=1e-8,
            fft_padding_factor=4.0,
            maximum_intermediate_points=4_000_000,
        ),
    )
    assert result.diagnostics.route == "scaled_fresnel_fft"
    assert result.diagnostics.b_displacement_px == pytest.approx(0.7433752879)
    assert abs(result.diagnostics.effective_defocus_m) == pytest.approx(
        5.533333333333333e-8
    )
    assert result.diagnostics.maximum_input_kernel_phase_step_rad > 100.0
    assert result.diagnostics.fft_nyquist_shift_xy_m[0] < (
        result.diagnostics.fft_padding_margin_xy_m[0]
    )
    assert result.diagnostics.power_relative_error < 3e-6
    intensity = result.intensity_yx
    xx, yy = np.meshgrid(detector_x, detector_y, indexing="xy")
    total = float(np.sum(intensity))
    propagated_waist = waist * np.sqrt(
        A**2 + (B * wavelength / (np.pi * waist**2)) ** 2
    )
    measured_x_waist = 2.0 * np.sqrt(float(np.sum(intensity * xx**2) / total))
    measured_y_waist = 2.0 * np.sqrt(float(np.sum(intensity * yy**2) / total))
    assert measured_x_waist == pytest.approx(propagated_waist, rel=3e-4)
    assert measured_y_waist == pytest.approx(propagated_waist, rel=3e-4)


def test_auto_route_order_is_conjugate_then_scaled_fresnel_then_czt():
    wavelength = 0.05
    x = np.linspace(-1.0, 1.0, 201)
    y = np.linspace(-0.9, 0.9, 181)
    field = _normalised_gaussian(x, y, waist=(0.16, 0.14))
    common = dict(
        field_yx=field,
        x_in_m=x,
        y_in_m=y,
        detector_x_coords_m=x,
        detector_y_coords_m=y,
        cumulative_rotation_rad=0.0,
        detector_affine_xy=np.eye(2),
        wavelength_m=wavelength,
    )
    conjugate = round_solenoid_grid_reference(
        **common,
        radial_abcd=(1.0, 0.0, 0.0, 1.0),
        options=GridWaveOptions(maximum_input_edge_power_fraction=1e-8),
    )
    assert conjugate.diagnostics.route == "conjugate"

    # A large magnification makes B/A short enough for padded FFT propagation.
    A, B, C = 30.0, 0.2, 0.0
    detector = np.linspace(-25.0, 25.0, 201)
    scaled = round_solenoid_grid_reference(
        field,
        x,
        y,
        detector,
        detector,
        radial_abcd=(A, B, C, 1.0 / A),
        cumulative_rotation_rad=0.0,
        detector_affine_xy=np.eye(2),
        wavelength_m=wavelength,
        options=GridWaveOptions(
            maximum_input_edge_power_fraction=1e-8,
            fft_maximum_paraxial_angle_rad=4.0,
            fft_minimum_samples_per_detector_pixel=0.8,
        ),
    )
    assert scaled.diagnostics.route == "scaled_fresnel_fft"

    czt = round_solenoid_grid_reference(
        **common,
        radial_abcd=(1.0, 0.8, 0.0, 1.0),
        options=GridWaveOptions(
            maximum_input_edge_power_fraction=1e-8,
            czt_maximum_kernel_phase_step_rad=4.0,
            scaled_fresnel_fft_enabled=False,
            output_oversampling=1.0,
            interpolation_halo_pixels=0,
        ),
    )
    assert czt.diagnostics.route == "czt"


def test_near_b_dispatch_and_explicit_wrong_route_fail_closed():
    x = np.linspace(-1.0, 1.0, 201)
    y = np.linspace(-0.9, 0.9, 181)
    field = _normalised_gaussian(x, y, waist=(0.16, 0.14))
    common = dict(
        field_yx=field,
        x_in_m=x,
        y_in_m=y,
        detector_x_coords_m=x,
        detector_y_coords_m=y,
        cumulative_rotation_rad=0.0,
        detector_affine_xy=np.eye(2),
        wavelength_m=0.05,
        options=GridWaveOptions(maximum_input_edge_power_fraction=1e-8),
    )
    conjugate = round_solenoid_grid_reference(
        **common, radial_abcd=(1.0, 0.0, 0.0, 1.0)
    )
    assert conjugate.diagnostics.route == "conjugate"

    # B is nonzero and well beyond the sub-pixel inverse-map certificate.
    with pytest.raises(GridWaveSamplingError) as caught:
        round_solenoid_grid_reference(
            **common,
            radial_abcd=(1.0, 0.2, 0.0, 1.0),
            route="conjugate",
        )
    assert "conjugate_b_displacement" in caught.value.diagnostics.failure_reasons


def test_sampling_and_symplectic_failures_are_explicit():
    x = np.linspace(-0.5, 0.5, 41)
    field = _normalised_gaussian(x, x, waist=(0.12, 0.12))
    with pytest.raises(GridWaveSamplingError) as caught:
        round_solenoid_grid_reference(
            field,
            x,
            x,
            x,
            x,
            radial_abcd=(1.0, 0.005, 0.0, 2.0),
            cumulative_rotation_rad=0.0,
            detector_affine_xy=np.eye(2),
            wavelength_m=1e-3,
            route="czt",
            options=GridWaveOptions(
                czt_maximum_kernel_phase_step_rad=0.1,
                maximum_input_edge_power_fraction=1e-3,
            ),
        )
    failures = caught.value.diagnostics.failure_reasons
    assert "radial_abcd_not_symplectic" in failures
    assert "czt_input_kernel_aliasing" in failures


@pytest.mark.parametrize(
    (
        "sample_count",
        "sample_step_m",
        "waist_m",
        "radial_abcd",
        "rotation_rad",
        "output_oversampling",
        "expected_intermediate_shape",
        "expected_output_power_fraction",
    ),
    (
        # Exact x/y topology and round sample-to-detector map for the executed
        # single_IL2_minus imaging representative (frame 119).
        (
            1247,
            1.846446866856455e-10,
            4.6e-8,
            (
                147307.30412120692,
                8.993959050457912,
                488748.3069720605,
                29.84090493754757,
            ),
            1.3984335125762708,
            2.0,
            (610, 616),
            0.9726068931500745,
        ),
        # Exact periodic-lattice topology and native hard-edge diffraction map
        # for nominal_0 (frame 10).  Its rotated finite-support CZT envelope
        # requires four intermediate samples per detector pixel to pass the
        # unchanged 0.2-sigma absolute-intensity validation gate.
        (
            1171,
            3.1356e-11,
            7.335243886611357e-9,
            (
                -4.401768615310577e-12,
                -0.3000000000000167,
                3.333333333386406,
                3.6297595513677883,
            ),
            -1.565593446225486,
            4.0,
            (3678, 3915),
            0.9999999920055332,
        ),
    ),
)
def test_executed_neoarm_czt_representatives_are_carrier_reduced_and_certified(
    sample_count,
    sample_step_m,
    waist_m,
    radial_abcd,
    rotation_rad,
    output_oversampling,
    expected_intermediate_shape,
    expected_output_power_fraction,
):
    wavelength = 2.5079339646683074e-12  # energy2wavelength(200 kV)
    sample_x = 2e-9 + (np.arange(sample_count) - sample_count // 2) * sample_step_m
    sample_y = -1e-9 + (np.arange(sample_count) - sample_count // 2) * sample_step_m
    gaussian_x = np.exp(-((sample_x - 2e-9) / waist_m) ** 2)
    gaussian_y = np.exp(-((sample_y + 1e-9) / waist_m) ** 2)
    field = np.sqrt(2.0 / (np.pi * waist_m**2)) * np.multiply.outer(
        gaussian_y, gaussian_x
    )

    detector = (np.arange(256) - 127.5) * 55e-6
    anisotropy = 0.006
    detector_angle = np.deg2rad(-1.7)
    detector_rotation = np.asarray(
        (
            (np.cos(detector_angle), -np.sin(detector_angle)),
            (np.sin(detector_angle), np.cos(detector_angle)),
        )
    )
    detector_affine = (
        detector_rotation
        @ np.diag((1.0 + anisotropy, 1.0 - anisotropy))
        / np.sqrt(1.0 - anisotropy**2)
    )
    maximum_phase_step = 0.8 * np.pi
    result = round_solenoid_grid_reference(
        field,
        sample_x,
        sample_y,
        detector,
        detector,
        radial_abcd=radial_abcd,
        cumulative_rotation_rad=rotation_rad,
        detector_affine_xy=detector_affine,
        wavelength_m=wavelength,
        route="auto",
        options=GridWaveOptions(
            czt_maximum_kernel_phase_step_rad=maximum_phase_step,
            maximum_input_edge_power_fraction=1e-3,
            fft_padding_factor=3.0,
            fft_maximum_nyquist_shift_fraction_of_padding=0.8,
            fft_maximum_paraxial_angle_rad=0.1,
            fft_minimum_samples_per_detector_pixel=1.0,
            output_oversampling=output_oversampling,
            interpolation_halo_pixels=3,
            maximum_intermediate_points=16_000_000,
        ),
    )

    diagnostics = result.diagnostics
    assert diagnostics.route == "czt"
    assert diagnostics.passed
    assert diagnostics.intermediate_shape_yx == expected_intermediate_shape
    assert diagnostics.maximum_input_kernel_phase_step_rad <= maximum_phase_step
    assert diagnostics.maximum_output_kernel_phase_step_rad <= (
        maximum_phase_step / output_oversampling * (1.0 + 2e-12)
    )
    # The previously rejected rapid sampled chirp is retained as a diagnostic
    # but restored analytically, not interpolated.  Refining the intermediate
    # output grid reduces this per-step diagnostic without making it sampleable.
    assert diagnostics.maximum_output_lens_phase_step_rad > 1e5
    assert diagnostics.input_power == pytest.approx(1.0, rel=2e-6)
    assert diagnostics.output_power / diagnostics.input_power == pytest.approx(
        expected_output_power_fraction, rel=3e-6
    )
    assert result.field_yx.shape == (256, 256)
    assert np.all(np.isfinite(result.field_yx))
    assert np.all(np.isfinite(result.intensity_yx))
