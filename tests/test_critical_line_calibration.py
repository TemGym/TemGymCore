from pathlib import Path
import tomllib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.critical_line_calibration import (
    AxialPlaneCovarianceBlock,
    CONDENSER_PROBE_OBSERVABLE_NAMES,
    DEFLECTOR_SAMPLE_CENTRAL_DIFFERENCE_OBSERVABLE_NAMES,
    DEFLECTOR_SAMPLE_POSITION_DIFFERENCE_OBSERVABLE_NAMES,
    ENTRANCE_COVARIANCE_COORDINATE_NAMES,
    PHASE_SPACE_COVARIANCE_OBSERVABLE_NAMES,
    POSITION_COVARIANCE_OBSERVABLE_NAMES,
    DeflectorCommandKickMap,
    EntranceCovarianceChart,
    HardEdgeInternalPlaneBounds,
    HardEdgeInternalPlaneConstraint,
    audit_axial_plane_covariance_nuisance_identifiability,
    audit_condenser_identifiability,
    audit_entrance_covariance_nuisance_identifiability,
    axial_plane_covariance_observables,
    axial_plane_covariance_observables_with_entrance_covariance,
    circular_gaussian_aperture_transmission,
    circular_perforation_signal,
    condenser_aperture_sweep,
    condenser_independent_reconnaissance_response,
    condenser_parameter_indices,
    condenser_reconnaissance_frame_observables,
    condenser_reconnaissance_response,
    central_deflector_command_difference,
    critical_line_scan,
    deflector_command_phase_space_moments,
    detector_log_radius,
    greedy_condenser_probe_design,
    hard_edge_matrix_to_plane,
    hard_edge_matrix_between_planes,
    ideal_angular_kick_response_matrices,
    linear_zero_crossings,
    phase_space_covariance_observables,
    phase_space_probe_observables,
    position_moments,
    propagate_phase_space_gaussian,
    restrict_hard_edge_internal_plane_bounds,
    restrict_hard_edge_internal_planes_bounds,
    signed_wobble_response,
    sample_perforation_edge_scan,
    sample_plane_probe_observables,
    sample_plane_probe_observables_with_entrance_covariance,
    validate_hard_edge_internal_plane_bounds,
    validate_hard_edge_internal_planes_bounds,
)
from temgym_core.neoarm_calibration import hard_edge_column_from_config
from temgym_core.microscope_inversion import response_observable_vector


jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def column_and_config():
    path = (
        Path(__file__).parents[1]
        / "tests"
        / "fixtures"
        / "neoarm_synthetic_column.toml"
    )
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    return hard_edge_column_from_config(config), config


def source_distribution():
    mean = np.asarray((1.0e-6, 2.0e-5, -0.5e-6, -1.0e-5))
    covariance = np.diag((22e-6**2, 0.18e-3**2, 22e-6**2, 0.18e-3**2))
    return mean, covariance


def neoarm_double_deflector_command_map():
    """Known 0.290/0.295 m hardware planes with a synthetic calibration."""

    gain = np.zeros((2, 2, 4), dtype=float)
    # Four recorded controls can have arbitrary calibrated crosstalk into both
    # physical kicks.  These values are test fixtures, not NEOARM calibration.
    gain[0, 0] = (1.2e-4, 0.1e-4, 0.4e-4, -0.2e-4)
    gain[0, 1] = (-0.1e-4, 1.1e-4, 0.2e-4, 0.3e-4)
    gain[1, 0] = (-0.9e-4, 0.05e-4, 0.7e-4, 0.1e-4)
    gain[1, 1] = (0.08e-4, -0.85e-4, -0.1e-4, 0.65e-4)
    return DeflectorCommandKickMap(
        plane_z_m=np.asarray((0.290, 0.295)),
        plane_names=("condenser_deflector_upper", "condenser_deflector_lower"),
        command_names=("shift_x", "shift_y", "tilt_x", "tilt_y"),
        reference_command=np.asarray((0.03, -0.02, 0.01, 0.04)),
        kick_bias_xy_rad=np.asarray(((2.0e-6, -1.0e-6), (-0.5e-6, 0.8e-6))),
        gain_xy_rad_per_command=gain,
    )


def test_phase_space_moment_propagation_matches_direct_linear_algebra():
    matrix = np.asarray(
        (
            (1.1, 0.2, 0.1, 0.0),
            (0.0, 0.9, 0.0, 0.1),
            (-0.1, 0.0, 1.05, 0.18),
            (0.0, -0.1, 0.0, 0.95),
        )
    )
    mean, covariance = source_distribution()
    propagated_mean, propagated_covariance = propagate_phase_space_gaussian(
        matrix, mean, covariance
    )
    np.testing.assert_allclose(propagated_mean, matrix @ mean)
    np.testing.assert_allclose(propagated_covariance, matrix @ covariance @ matrix.T)
    affine = np.asarray(((1.02, 0.03), (-0.02, 0.98)))
    xy_mean, xy_covariance = position_moments(
        matrix, mean, covariance, affine_xy=affine
    )
    indices = np.asarray((0, 2))
    np.testing.assert_allclose(xy_mean, affine @ (matrix @ mean)[indices])
    np.testing.assert_allclose(
        xy_covariance,
        affine @ (matrix @ covariance @ matrix.T)[np.ix_(indices, indices)] @ affine.T,
    )


def test_internal_plane_matrix_matches_sample_segment(column_and_config):
    column, _ = column_and_config
    parameters = column.nominal_parameters
    dacs = np.zeros(len(column.channel_names))
    actual = hard_edge_matrix_to_plane(column, parameters, dacs, column.sample_z_m)
    expected = column.matrix(parameters, dacs, segment="source_to_sample")
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    with pytest.raises(ValueError, match="inside nominal lens"):
        hard_edge_matrix_to_plane(column, parameters, dacs, 0.2)


def test_fixed_plane_transfer_composes_and_is_jax_differentiable(
    column_and_config,
):
    column, _ = column_and_config
    parameters = jnp.asarray(column.nominal_parameters)
    dacs = jnp.zeros(len(column.channel_names))
    start_z = 0.290
    transfer = hard_edge_matrix_between_planes(
        column,
        parameters,
        dacs,
        from_z_m=start_z,
        to_z_m=column.sample_z_m,
    )
    source_to_start = hard_edge_matrix_to_plane(column, parameters, dacs, start_z)
    source_to_sample = hard_edge_matrix_to_plane(
        column, parameters, dacs, column.sample_z_m
    )
    np.testing.assert_allclose(
        transfer @ source_to_start,
        source_to_sample,
        rtol=3e-13,
        atol=3e-13,
    )

    compiled = jax.jit(
        lambda fitted: hard_edge_matrix_between_planes(
            column,
            fitted,
            dacs,
            from_z_m=start_z,
            to_z_m=column.sample_z_m,
        )
    )
    np.testing.assert_allclose(compiled(parameters), transfer, rtol=3e-13, atol=3e-13)
    derivative = np.asarray(jax.jacfwd(compiled)(parameters))
    assert derivative.shape == (4, 4, column.n_parameters)
    assert np.all(np.isfinite(derivative))
    assert np.linalg.norm(derivative) > 0.0


def test_known_double_deflector_kick_response_has_explicit_plane_state_axes(
    column_and_config,
):
    column, _ = column_and_config
    parameters = column.nominal_parameters
    dacs = np.zeros(len(column.channel_names))
    planes = np.asarray((0.290, 0.295))
    response = ideal_angular_kick_response_matrices(
        column, parameters, dacs, planes
    )
    assert response.shape == (2, 4, 2)
    angular_basis = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 1.0))
    )
    for index, plane in enumerate(planes):
        transfer = hard_edge_matrix_between_planes(
            column,
            parameters,
            dacs,
            from_z_m=float(plane),
            to_z_m=column.sample_z_m,
        )
        np.testing.assert_allclose(response[index], transfer @ angular_basis)

    single = ideal_angular_kick_response_matrices(
        column, parameters, dacs, float(planes[0])
    )
    assert single.shape == (1, 4, 2)
    np.testing.assert_allclose(single[0], response[0])

    # A kick applied after CL3 cannot contain CL1/CL3 information.  The
    # source-referenced factors must cancel, while the downstream C-mini
    # response remains optically sensitive.
    derivative = np.asarray(
        jax.jacfwd(
            lambda fitted: ideal_angular_kick_response_matrices(
                column, fitted, dacs, planes
            )
        )(jnp.asarray(parameters))
    )
    names = column.parameter_names
    upstream = tuple(
        index
        for index, name in enumerate(names)
        if name == "source_z_m" or name.startswith(("CL1.", "CL3."))
    )
    assert np.linalg.norm(derivative[..., upstream]) < 1.0e-10
    assert np.linalg.norm(derivative[..., names.index("C-mini.q_bias")]) > 1.0e-3


def test_affine_deflector_command_map_is_named_frozen_and_jax_ready():
    command_map = neoarm_double_deflector_command_map()
    assert command_map.n_planes == 2
    assert command_map.n_commands == 4
    assert command_map.n_parameters == 20
    assert len(command_map.parameter_names) == command_map.n_parameters
    assert len(set(command_map.parameter_names)) == command_map.n_parameters
    assert command_map.parameter_names[0].endswith("kick_bias_x_rad")
    assert command_map.parameter_names[-1].endswith("kick_gain_y_rad_per_tilt_y")
    assert not command_map.plane_z_m.flags.writeable
    assert not command_map.nominal_parameters.flags.writeable

    commands = jnp.asarray(
        (
            (0.03, -0.02, 0.01, 0.04),
            (0.13, -0.12, 0.06, -0.01),
        )
    )
    expected = command_map.kick_bias_xy_rad[None, :, :] + np.einsum(
        "pac,nc->npa",
        command_map.gain_xy_rad_per_command,
        np.asarray(commands) - command_map.reference_command,
    )
    compiled = jax.jit(command_map.kicks)
    actual = compiled(commands, jnp.asarray(command_map.nominal_parameters))
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    derivative = np.asarray(
        jax.jacfwd(lambda fitted: command_map.kicks(commands, fitted))(
            jnp.asarray(command_map.nominal_parameters)
        )
    )
    assert derivative.shape == (2, 2, 2, command_map.n_parameters)
    assert np.all(np.isfinite(derivative))


def test_deflector_central_pair_cancels_source_mean_covariance_and_bias(
    column_and_config,
):
    column, _ = column_and_config
    parameters = jnp.asarray(column.nominal_parameters)
    dacs = jnp.zeros(len(column.channel_names))
    command_map = neoarm_double_deflector_command_map()
    centre = jnp.asarray((0.08, -0.04, 0.03, -0.02))
    delta = jnp.asarray((0.05, -0.03, 0.02, 0.04))
    mean_a, covariance_a = source_distribution()
    mean_b = np.asarray((-8.0e-6, 7.0e-5, 3.0e-6, -4.0e-5))
    covariance_b = np.asarray(covariance_a) * np.asarray(
        (
            (1.7, 0.0, 0.0, 0.0),
            (0.0, 0.8, 0.0, 0.0),
            (0.0, 0.0, 1.3, 0.0),
            (0.0, 0.0, 0.0, 2.1),
        )
    )
    result_a = central_deflector_command_difference(
        column,
        parameters,
        dacs,
        mean_a,
        covariance_a,
        centre,
        delta,
        command_map,
        mode="full",
    )
    result_b = central_deflector_command_difference(
        column,
        parameters,
        dacs,
        mean_b,
        covariance_b,
        centre,
        delta,
        command_map,
        mode="full",
    )
    assert len(DEFLECTOR_SAMPLE_CENTRAL_DIFFERENCE_OBSERVABLE_NAMES) == 4
    assert len(DEFLECTOR_SAMPLE_POSITION_DIFFERENCE_OBSERVABLE_NAMES) == 2
    np.testing.assert_allclose(result_a.mean_half_difference, result_b.mean_half_difference)
    np.testing.assert_allclose(result_a.covariance_half_difference, 0.0, atol=0.0)
    np.testing.assert_allclose(result_b.covariance_half_difference, 0.0, atol=0.0)

    response = ideal_angular_kick_response_matrices(
        column, parameters, dacs, command_map.plane_z_m
    )
    expected_kick = np.einsum(
        "pac,c->pa", command_map.gain_xy_rad_per_command, np.asarray(delta)
    )
    expected_state = np.einsum("pia,pa->i", np.asarray(response), expected_kick)
    np.testing.assert_allclose(result_a.mean_half_difference, expected_state)

    changed_bias = command_map.nominal_parameters.copy()
    changed_bias[: 2 * command_map.n_planes] += 9.0e-5
    bias_result = central_deflector_command_difference(
        column,
        parameters,
        dacs,
        mean_a,
        covariance_a,
        centre,
        delta,
        command_map,
        calibration_parameters=changed_bias,
        mode="full",
    )
    np.testing.assert_allclose(bias_result.mean_half_difference, expected_state)


def test_deflector_command_moments_and_central_response_are_jax_differentiable(
    column_and_config,
):
    column, _ = column_and_config
    parameters = jnp.asarray(column.nominal_parameters)
    dacs = jnp.zeros(len(column.channel_names))
    mean, covariance = source_distribution()
    command_map = neoarm_double_deflector_command_map()
    commands = jnp.asarray(((0.0, 0.0, 0.0, 0.0), (0.1, -0.1, 0.05, 0.02)))
    means, covariances = deflector_command_phase_space_moments(
        column,
        parameters,
        dacs,
        mean,
        covariance,
        commands,
        command_map,
    )
    assert means.shape == (2, 4)
    assert covariances.shape == (2, 4, 4)
    np.testing.assert_allclose(covariances[0], covariances[1], rtol=0.0, atol=0.0)

    centre = jnp.zeros(4)
    delta = jnp.asarray((0.04, 0.0, -0.02, 0.01))

    @jax.jit
    def prediction(fitted_optics, fitted_deflector):
        return central_deflector_command_difference(
            column,
            fitted_optics,
            dacs,
            mean,
            covariance,
            centre,
            delta,
            command_map,
            calibration_parameters=fitted_deflector,
            mode="position",
        ).mean_half_difference

    actual = prediction(parameters, jnp.asarray(command_map.nominal_parameters))
    assert actual.shape == (2,)
    optical_jacobian, calibration_jacobian = jax.jacfwd(
        prediction, argnums=(0, 1)
    )(parameters, jnp.asarray(command_map.nominal_parameters))
    assert optical_jacobian.shape == (2, column.n_parameters)
    assert calibration_jacobian.shape == (2, command_map.n_parameters)
    assert np.all(np.isfinite(optical_jacobian))
    assert np.all(np.isfinite(calibration_jacobian))
    np.testing.assert_allclose(
        calibration_jacobian[:, : 2 * command_map.n_planes], 0.0, atol=0.0
    )
    assert np.linalg.norm(calibration_jacobian) > 0.0


def test_deflector_calibration_contract_rejects_ambiguous_or_unsafe_inputs(
    column_and_config,
):
    column, _ = column_and_config
    command_map = neoarm_double_deflector_command_map()
    with pytest.raises(ValueError, match="one or two planes"):
        DeflectorCommandKickMap(
            plane_z_m=np.asarray((0.28, 0.29, 0.30)),
            plane_names=("a", "b", "c"),
            command_names=("x",),
            reference_command=np.zeros(1),
            kick_bias_xy_rad=np.zeros((3, 2)),
            gain_xy_rad_per_command=np.zeros((3, 2, 1)),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        DeflectorCommandKickMap(
            plane_z_m=np.asarray((0.295, 0.290)),
            plane_names=("a", "b"),
            command_names=("x",),
            reference_command=np.zeros(1),
            kick_bias_xy_rad=np.zeros((2, 2)),
            gain_xy_rad_per_command=np.zeros((2, 2, 1)),
        )
    bad_gain = command_map.gain_xy_rad_per_command.copy()
    bad_gain[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        DeflectorCommandKickMap(
            plane_z_m=command_map.plane_z_m,
            plane_names=command_map.plane_names,
            command_names=command_map.command_names,
            reference_command=command_map.reference_command,
            kick_bias_xy_rad=command_map.kick_bias_xy_rad,
            gain_xy_rad_per_command=bad_gain,
        )
    with pytest.raises(ValueError, match="command_values"):
        command_map.kicks(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="inside nominal lens"):
        ideal_angular_kick_response_matrices(
            column,
            column.nominal_parameters,
            np.zeros(len(column.channel_names)),
            0.300,
        )
    mean, covariance = source_distribution()
    with pytest.raises(ValueError, match="non-zero"):
        central_deflector_command_difference(
            column,
            column.nominal_parameters,
            np.zeros(len(column.channel_names)),
            mean,
            covariance,
            np.zeros(4),
            np.zeros(4),
            command_map,
        )


def test_neoarm_aperture_plane_bounds_are_restricted_and_certified(
    column_and_config,
):
    column, config = column_and_config
    plane = float(config["beam"]["condenser_aperture"]["z_m"])
    certificate = restrict_hard_edge_internal_plane_bounds(
        column,
        column.default_bounds(),
        plane_z_m=plane,
        upstream_lens_name="CL1",
        downstream_lens_name="CL3",
    )
    assert isinstance(certificate, HardEdgeInternalPlaneBounds)
    assert certificate.upstream_lens_name == "CL1"
    assert certificate.downstream_lens_name == "CL3"
    assert certificate.worst_case_clearance_m >= -2.0e-16
    assert certificate.was_restricted
    assert not certificate.lower_bounds.flags.writeable
    assert not certificate.upper_bounds.flags.writeable
    mutable_lower, mutable_upper = certificate.bounds
    assert mutable_lower.flags.writeable and mutable_upper.flags.writeable
    validate_hard_edge_internal_plane_bounds(
        column,
        (mutable_lower, mutable_upper),
        plane_z_m=plane,
        upstream_lens_name="CL1",
        downstream_lens_name="CL3",
    )


def test_internal_plane_restriction_closes_every_adverse_box_corner(
    column_and_config,
):
    column, config = column_and_config
    plane = float(config["beam"]["condenser_aperture"]["z_m"])
    lower, upper = column.default_bounds()
    names = column.parameter_names
    up_z = names.index("CL1.z_center_m")
    up_log_l = names.index("CL1.log_length_m")
    down_z = names.index("CL3.z_center_m")
    down_log_l = names.index("CL3.log_length_m")
    upper[up_z] += 2.0e-3
    lower[down_z] -= 3.0e-3
    with pytest.raises(ValueError, match="clearance claim"):
        validate_hard_edge_internal_plane_bounds(
            column,
            (lower, upper),
            plane_z_m=plane,
            upstream_lens_name="CL1",
            downstream_lens_name="CL3",
            minimum_clearance_m=50.0e-6,
        )

    restricted = restrict_hard_edge_internal_plane_bounds(
        column,
        (lower, upper),
        plane_z_m=plane,
        upstream_lens_name="CL1",
        downstream_lens_name="CL3",
        minimum_clearance_m=50.0e-6,
    )
    assert restricted.was_restricted
    assert restricted.upstream_clearance_m >= 50.0e-6 - 2.0e-16
    assert restricted.downstream_clearance_m >= 50.0e-6 - 2.0e-16
    np.testing.assert_allclose(
        restricted.worst_case_upstream_exit_z_m,
        restricted.upper_bounds[up_z]
        + 0.5 * np.exp(restricted.upper_bounds[up_log_l]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        restricted.worst_case_downstream_entrance_z_m,
        restricted.lower_bounds[down_z]
        - 0.5 * np.exp(restricted.upper_bounds[down_log_l]),
        rtol=0.0,
        atol=0.0,
    )
    reference = column.nominal_parameters
    assert np.all(reference >= restricted.lower_bounds)
    assert np.all(reference <= restricted.upper_bounds)
    revalidated = validate_hard_edge_internal_plane_bounds(
        column,
        restricted.bounds,
        plane_z_m=plane,
        upstream_lens_name="CL1",
        downstream_lens_name="CL3",
        minimum_clearance_m=50.0e-6,
    )
    assert revalidated.worst_case_clearance_m == pytest.approx(
        restricted.worst_case_clearance_m, abs=2.0e-16
    )


def test_internal_plane_bounds_fail_for_invalid_pair_and_infeasible_reference(
    column_and_config,
):
    column, config = column_and_config
    bounds = column.default_bounds()
    plane = float(config["beam"]["condenser_aperture"]["z_m"])
    with pytest.raises(ValueError, match="adjacent and in axial order"):
        validate_hard_edge_internal_plane_bounds(
            column,
            bounds,
            plane_z_m=plane,
            upstream_lens_name="CL3",
            downstream_lens_name="CL1",
        )
    with pytest.raises(ValueError, match="nominal inter-lens gap"):
        restrict_hard_edge_internal_plane_bounds(
            column,
            bounds,
            plane_z_m=0.205,
            upstream_lens_name="CL1",
            downstream_lens_name="CL3",
        )
    with pytest.raises(ValueError, match="reference_parameters"):
        restrict_hard_edge_internal_plane_bounds(
            column,
            bounds,
            plane_z_m=plane,
            upstream_lens_name="CL1",
            downstream_lens_name="CL3",
            minimum_clearance_m=16.0e-3,
        )


def test_actual_condenser_hardware_planes_share_one_certified_parameter_box(
    column_and_config,
):
    column, _ = column_and_config
    hardware_path = (
        Path(__file__).parents[1]
        / "examples"
        / "microscope_models"
        / "microscope.toml"
    )
    with hardware_path.open("rb") as stream:
        hardware = tomllib.load(stream)
    aperture_z = float(hardware["apertures"]["C_aperture"]["z_m"])
    deflector_z = float(hardware["deflectors"]["CL12_deflector"]["z_m"])
    deflector_spacing = float(
        hardware["deflectors"]["CL12_deflector"]["spacing_m"]
    )
    plane_z = (aperture_z, deflector_z, deflector_z + deflector_spacing)
    np.testing.assert_allclose(plane_z, (0.250, 0.290, 0.295), atol=0.0, rtol=0.0)

    minimum_clearance = 0.5e-3
    constraints = tuple(
        HardEdgeInternalPlaneConstraint(
            z_m,
            "CL3",
            "C-mini",
            minimum_clearance,
        )
        for z_m in plane_z
    )
    certificates = restrict_hard_edge_internal_planes_bounds(
        column,
        column.default_bounds(),
        constraints,
    )
    assert len(certificates) == 3
    assert all(certificate.was_restricted for certificate in certificates)
    for certificate in certificates[1:]:
        np.testing.assert_array_equal(
            certificate.lower_bounds,
            certificates[0].lower_bounds,
        )
        np.testing.assert_array_equal(
            certificate.upper_bounds,
            certificates[0].upper_bounds,
        )

    # Intersecting independently restricted boxes must not depend on how the
    # physical planes happen to be listed in an acquisition manifest.
    reverse = restrict_hard_edge_internal_planes_bounds(
        column,
        column.default_bounds(),
        tuple(reversed(constraints)),
    )
    np.testing.assert_array_equal(
        certificates[0].lower_bounds,
        reverse[0].lower_bounds,
    )
    np.testing.assert_array_equal(
        certificates[0].upper_bounds,
        reverse[0].upper_bounds,
    )

    validated = validate_hard_edge_internal_planes_bounds(
        column,
        certificates[0].bounds,
        constraints,
    )
    assert tuple(certificate.plane_z_m for certificate in validated) == plane_z

    names = column.parameter_names
    up_z = names.index("CL3.z_center_m")
    up_log_l = names.index("CL3.log_length_m")
    down_z = names.index("C-mini.z_center_m")
    down_log_l = names.index("C-mini.log_length_m")
    lower, upper = certificates[0].bounds

    # These are all adverse corners relevant to static membership.  Every
    # other parameter is immaterial to whether CL3 or C-mini contains a plane.
    for corner in np.ndindex(*(2,) * 4):
        values = (
            (lower[up_z], upper[up_z])[corner[0]],
            (lower[up_log_l], upper[up_log_l])[corner[1]],
            (lower[down_z], upper[down_z])[corner[2]],
            (lower[down_log_l], upper[down_log_l])[corner[3]],
        )
        upstream_exit = values[0] + 0.5 * np.exp(values[1])
        downstream_entrance = values[2] - 0.5 * np.exp(values[3])
        for specification in constraints:
            assert upstream_exit <= (
                specification.plane_z_m
                - specification.minimum_clearance_m
                + 2.0e-16
            )
            assert downstream_entrance >= (
                specification.plane_z_m
                + specification.minimum_clearance_m
                - 2.0e-16
            )

    tight = certificates[-1]
    assert tight.plane_z_m == pytest.approx(0.295, abs=0.0)
    assert tight.downstream_clearance_m == pytest.approx(
        minimum_clearance,
        abs=2.0e-16,
    )
    assert lower[down_z] - 0.5 * np.exp(upper[down_log_l]) >= (
        0.295 + minimum_clearance - 2.0e-16
    )


def test_multi_plane_bound_validation_identifies_the_failing_plane(
    column_and_config,
):
    column, _ = column_and_config
    constraints = (
        HardEdgeInternalPlaneConstraint(0.250, "CL3", "C-mini", 0.5e-3),
        HardEdgeInternalPlaneConstraint(0.295, "CL3", "C-mini", 0.5e-3),
    )
    first_plane = restrict_hard_edge_internal_plane_bounds(
        column,
        column.default_bounds(),
        plane_z_m=constraints[0].plane_z_m,
        upstream_lens_name=constraints[0].upstream_lens_name,
        downstream_lens_name=constraints[0].downstream_lens_name,
        minimum_clearance_m=constraints[0].minimum_clearance_m,
    )
    with pytest.raises(ValueError, match=r"constraint 1 at z=0\.295 m failed"):
        validate_hard_edge_internal_planes_bounds(
            column,
            first_plane.bounds,
            constraints,
        )
    with pytest.raises(ValueError, match="at least one"):
        restrict_hard_edge_internal_planes_bounds(
            column,
            column.default_bounds(),
            (),
        )


def test_calibrated_sample_plane_response_has_exact_zero_condenser_jacobian(
    column_and_config,
):
    """A downstream response cannot identify lenses upstream of the sample."""

    column, _ = column_and_config
    parameters = jnp.asarray(column.nominal_parameters)
    dacs = jnp.zeros(len(column.channel_names))
    condenser_indices = condenser_parameter_indices(column)
    jacobian = np.asarray(
        jax.jacfwd(
            lambda fitted: response_observable_vector(
                column.matrix(fitted, dacs, segment="sample_to_detector")
            )
        )(parameters)
    )[:, condenser_indices]

    np.testing.assert_array_equal(jacobian, np.zeros_like(jacobian))


@pytest.mark.parametrize("radius_over_sigma", (0.5, 1.0, 2.0, 3.0))
def test_circular_aperture_matches_centered_isotropic_closed_form(radius_over_sigma):
    sigma = 7.0e-6
    radius = radius_over_sigma * sigma
    numerical = circular_gaussian_aperture_transmission(
        np.zeros(2),
        np.eye(2) * sigma**2,
        radius,
        radial_order=48,
        azimuthal_order=128,
    )
    expected = 1.0 - np.exp(-0.5 * radius_over_sigma**2)
    assert float(numerical) == pytest.approx(expected, rel=2e-10, abs=2e-12)


def test_aperture_probability_has_finite_jax_gradient():
    covariance = jnp.asarray(((30e-12, 4e-12), (4e-12, 20e-12)))
    derivative = jax.grad(
        lambda offset: circular_gaussian_aperture_transmission(
            jnp.asarray((offset, 0.0)), covariance, 8e-6
        )
    )(jnp.asarray(1e-6))
    assert np.isfinite(float(derivative))
    assert derivative < 0.0


def test_linear_zero_crossings_handles_multiple_critical_branches():
    x = np.linspace(-2.0, 2.0, 17)
    roots = linear_zero_crossings(x, (x + 1.25) * (x - 0.5))
    np.testing.assert_allclose(roots, (-1.25, 0.5), atol=1e-14)


def test_wobble_response_matches_autodiff_limit(column_and_config):
    column, _ = column_and_config
    parameters = column.nominal_parameters
    dacs = np.zeros(len(column.channel_names))
    mean, covariance = source_distribution()
    channel = column.channel_names.index("IL2")
    finite = signed_wobble_response(
        column,
        parameters,
        dacs,
        mean,
        covariance,
        wobble_channel_index=channel,
        wobble_step=1e-5,
    )

    def response(value):
        controls = jnp.asarray(dacs).at[channel].set(value)
        return detector_log_radius(column, parameters, controls, mean, covariance)

    automatic = jax.grad(response)(jnp.asarray(0.0))
    np.testing.assert_allclose(finite, automatic, rtol=5e-7, atol=1e-9)


def test_explicit_detector_plane_preserves_default_log_radius(column_and_config):
    column, _ = column_and_config
    parameters = column.nominal_parameters
    dacs = np.linspace(-0.1, 0.1, len(column.channel_names))
    mean, covariance = source_distribution()
    default = detector_log_radius(column, parameters, dacs, mean, covariance)
    explicit = detector_log_radius(
        column,
        parameters,
        dacs,
        mean,
        covariance,
        observation_z_m=column.detector_z_m,
    )
    np.testing.assert_allclose(explicit, default, rtol=2e-13, atol=2e-13)


def test_sample_plane_wobble_response_matches_autodiff_limit(column_and_config):
    column, _ = column_and_config
    parameters = column.nominal_parameters
    dacs = np.zeros(len(column.channel_names))
    mean, covariance = source_distribution()
    channel = column.channel_names.index("C-mini")
    finite = signed_wobble_response(
        column,
        parameters,
        dacs,
        mean,
        covariance,
        wobble_channel_index=channel,
        wobble_step=1e-5,
        observation_z_m=column.sample_z_m,
    )

    def response(value):
        controls = jnp.asarray(dacs).at[channel].set(value)
        return detector_log_radius(
            column,
            parameters,
            controls,
            mean,
            covariance,
            observation_z_m=column.sample_z_m,
        )

    automatic = jax.grad(response)(jnp.asarray(0.0))
    np.testing.assert_allclose(finite, automatic, rtol=5e-7, atol=1e-9)


def test_critical_scan_records_and_uses_sample_observation_plane(column_and_config):
    column, _ = column_and_config
    parameters = column.nominal_parameters
    reference = np.zeros(len(column.channel_names))
    mean, covariance = source_distribution()
    scan_values = np.linspace(-0.35, 0.35, 9)
    scan_channel = column.channel_names.index("CL1")
    wobble_channel = column.channel_names.index("CL3")
    sample_scan = critical_line_scan(
        column,
        parameters,
        reference,
        mean,
        covariance,
        scan_channel_index=scan_channel,
        wobble_channel_index=wobble_channel,
        scan_values=scan_values,
        wobble_step=0.01,
        observation_z_m=column.sample_z_m,
    )
    expected = []
    for value in scan_values:
        dacs = reference.copy()
        dacs[scan_channel] = value
        expected.append(
            signed_wobble_response(
                column,
                parameters,
                dacs,
                mean,
                covariance,
                wobble_channel_index=wobble_channel,
                wobble_step=0.01,
                observation_z_m=column.sample_z_m,
            )
        )
    assert sample_scan.observation_z_m == column.sample_z_m
    np.testing.assert_allclose(sample_scan.response, expected)

    detector_scan = critical_line_scan(
        column,
        parameters,
        reference,
        mean,
        covariance,
        scan_channel_index=scan_channel,
        wobble_channel_index=wobble_channel,
        scan_values=scan_values,
        wobble_step=0.01,
    )
    assert detector_scan.observation_z_m is None
    assert not np.allclose(sample_scan.response, detector_scan.response)


def test_internal_observation_plane_rejects_lens_interior(column_and_config):
    column, _ = column_and_config
    mean, covariance = source_distribution()
    with pytest.raises(ValueError, match="inside nominal lens"):
        detector_log_radius(
            column,
            column.nominal_parameters,
            np.zeros(len(column.channel_names)),
            mean,
            covariance,
            observation_z_m=0.2,
        )


def test_critical_and_aperture_scans_are_finite_in_raw_dac_coordinates(
    column_and_config,
):
    column, _ = column_and_config
    parameters = column.nominal_parameters
    reference = np.zeros(len(column.channel_names))
    mean, covariance = source_distribution()
    scan_values = np.linspace(-0.3, 0.4, 15)
    critical = critical_line_scan(
        column,
        parameters,
        reference,
        mean,
        covariance,
        scan_channel_index=column.channel_names.index("IL1"),
        wobble_channel_index=column.channel_names.index("IL2"),
        scan_values=scan_values,
        wobble_step=0.01,
    )
    assert critical.response.shape == scan_values.shape
    assert np.isfinite(critical.response).all()
    aperture = condenser_aperture_sweep(
        column,
        parameters,
        reference,
        mean,
        covariance,
        aperture_z_m=0.225,
        aperture_radius_m=20e-6,
        scan_channel_index=column.channel_names.index("CL1"),
        scan_values=scan_values,
        radial_order=20,
        azimuthal_order=48,
    )
    assert np.isfinite(aperture.transmission).all()
    assert np.all((0.0 <= aperture.transmission) & (aperture.transmission <= 1.0))
    assert np.all(aperture.incident_rms_radius_m > 0.0)


def test_phase_space_probe_observables_have_explicit_physical_contract():
    mean = jnp.asarray((2.0e-9, 3.0e-6, -4.0e-9, -5.0e-6))
    covariance = jnp.asarray(
        (
            (9.0e-18, 3.0e-15, 0.0, 1.5e-15),
            (3.0e-15, 16.0e-12, -0.5e-15, 0.0),
            (0.0, -0.5e-15, 25.0e-18, 5.0e-15),
            (1.5e-15, 0.0, 5.0e-15, 36.0e-12),
        )
    )
    actual = phase_space_probe_observables(mean, covariance)
    position_variance = 17.0e-18
    angle_variance = 26.0e-12
    normalization = np.sqrt(position_variance * angle_variance)
    expected = np.asarray(
        (
            mean[0],
            mean[2],
            mean[1],
            mean[3],
            0.5 * np.log(position_variance),
            0.5 * np.log(angle_variance),
            4.0e-15 / normalization,
            1.0e-15 / normalization,
        )
    )
    assert len(CONDENSER_PROBE_OBSERVABLE_NAMES) == 8
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    derivative = jax.jacfwd(
        lambda scale: phase_space_probe_observables(mean, scale * covariance)
    )(jnp.asarray(1.0))
    assert np.all(np.isfinite(np.asarray(derivative)))
    np.testing.assert_allclose(derivative[4:6], 0.5, atol=2e-13)
    np.testing.assert_allclose(derivative[6:], 0.0, atol=2e-13)


def test_phase_space_covariance_observables_are_unit_stable_and_complete(
    column_and_config,
):
    scales = np.asarray((3.0e-9, 4.0e-6, 5.0e-9, 6.0e-6))
    correlation = np.asarray(
        (
            (1.0, 0.12, -0.08, 0.04),
            (0.12, 1.0, 0.06, -0.03),
            (-0.08, 0.06, 1.0, 0.15),
            (0.04, -0.03, 0.15, 1.0),
        )
    )
    covariance = np.diag(scales) @ correlation @ np.diag(scales)
    full = np.asarray(phase_space_covariance_observables(covariance))
    assert len(PHASE_SPACE_COVARIANCE_OBSERVABLE_NAMES) == full.size == 10
    np.testing.assert_allclose(full[:4], np.log(scales), rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        full[4:], correlation[np.triu_indices(4, k=1)], rtol=2e-14, atol=2e-14
    )
    position = np.asarray(
        phase_space_covariance_observables(covariance, mode="position")
    )
    assert len(POSITION_COVARIANCE_OBSERVABLE_NAMES) == position.size == 3
    np.testing.assert_allclose(position, full[[0, 2, 5]], rtol=0.0, atol=0.0)

    chart = EntranceCovarianceChart(covariance)
    chart_jacobian = np.asarray(
        jax.jacfwd(
            lambda eta: phase_space_covariance_observables(
                chart.covariance(eta)
            )
        )(jnp.zeros(chart.n_coordinates))
    )
    normalized = chart_jacobian / np.linalg.norm(
        chart_jacobian, axis=0, keepdims=True
    )
    singular = np.linalg.svd(normalized, compute_uv=False)
    assert np.sum(singular > 1.0e-10 * singular[0]) == 10

    column, _ = column_and_config
    controls = np.zeros(len(column.channel_names))
    direct = axial_plane_covariance_observables(
        column,
        column.nominal_parameters,
        controls,
        covariance,
        plane_z_m=0.225,
    )
    released = jax.jit(
        lambda eta: axial_plane_covariance_observables_with_entrance_covariance(
            column,
            jnp.asarray(column.nominal_parameters),
            eta,
            controls,
            chart,
            plane_z_m=0.225,
        )
    )(jnp.zeros(chart.n_coordinates))
    assert direct.shape == released.shape == (1, 10)
    np.testing.assert_allclose(direct, released, rtol=2e-13, atol=2e-13)


def test_perforation_edge_signal_is_direct_and_differentiable(column_and_config):
    sigma = 3.0e-9
    radius = 2.0 * sigma
    vacuum, material = 0.91, 0.17
    centred = circular_perforation_signal(
        np.zeros(2),
        np.eye(2) * sigma**2,
        np.zeros(2),
        radius,
        vacuum_signal=vacuum,
        material_signal=material,
        radial_order=48,
        azimuthal_order=128,
    )
    expected_probability = 1.0 - np.exp(-2.0)
    assert float(centred) == pytest.approx(
        material + (vacuum - material) * expected_probability,
        rel=2e-10,
    )
    gradient = jax.grad(
        lambda offset: circular_perforation_signal(
            np.zeros(2),
            np.eye(2) * sigma**2,
            jnp.asarray((offset, 0.0)),
            radius,
            radial_order=24,
            azimuthal_order=64,
        )
    )(jnp.asarray(sigma))
    assert np.isfinite(float(gradient)) and gradient < 0.0

    column, _ = column_and_config
    offsets = np.asarray(((-2e-9, 0.0), (0.0, 0.0), (2e-9, 0.0)))
    values = sample_perforation_edge_scan(
        column,
        column.nominal_parameters,
        np.zeros(len(column.channel_names)),
        np.zeros(4),
        np.diag((12e-6**2, 0.1e-3**2, 12e-6**2, 0.1e-3**2)),
        stage_offsets_xy_m=offsets,
        perforation_centre_xy_m=np.zeros(2),
        perforation_radius_m=40e-9,
        radial_order=12,
        azimuthal_order=32,
    )
    assert values.shape == (3,)
    np.testing.assert_allclose(values[0], values[2], rtol=2e-12, atol=2e-12)


def _neoarm_reconnaissance_contract(column, config):
    parameters = column.nominal_parameters
    reference = np.zeros(len(column.channel_names))
    entrance = config["beam"]["entrance_calibration"]
    sample_sigma_position = np.asarray(
        entrance["reference_sample_sigma_position_xy_m"]
    )
    sample_sigma_tilt = np.asarray(entrance["reference_sample_sigma_tilt_xy_rad"])
    sample_covariance = np.diag(
        (
            sample_sigma_position[0] ** 2,
            sample_sigma_tilt[0] ** 2,
            sample_sigma_position[1] ** 2,
            sample_sigma_tilt[1] ** 2,
        )
    )
    reference_map = np.asarray(
        column.matrix(parameters, reference, segment="source_to_sample")
    )
    inverse = np.linalg.inv(reference_map)
    source_covariance = inverse @ sample_covariance @ inverse.T
    source_mean = np.asarray(entrance["mean_state_x_thetax_y_thetay"])
    recon = config["experiment"]["condenser_reconnaissance"]
    cl1 = column.channel_names.index("CL1")
    cl3 = column.channel_names.index("CL3")
    half_width = float(recon["scan_half_width_raw_dac"])
    scan = np.linspace(
        -half_width, half_width, int(recon["scan_point_count"])
    )
    wobble = float(recon["wobble_step_raw_dac"])
    pairs = []
    for value in scan:
        pair = np.repeat(reference[None, :], 2, axis=0)
        pair[:, cl1] = value
        pair[0, cl3] -= wobble
        pair[1, cl3] += wobble
        pairs.append(pair)
    scales = np.empty(column.n_parameters)
    scales[:2] = 1.0e-3
    cursor = 2
    for lens in column.lenses:
        scales[cursor : cursor + 3] = (5.0e-4, 0.08, 0.08)
        cursor += 3
        if lens.channel_index is not None:
            scales[cursor] = 0.08
            cursor += 1
    return (
        parameters,
        reference,
        source_mean,
        source_covariance,
        np.stack(pairs),
        wobble,
        scales,
    )


def test_entrance_covariance_chart_origin_is_the_calibrated_reference(
    column_and_config,
):
    column, config = column_and_config
    _, _, _, reference, _, _, _ = _neoarm_reconnaissance_contract(column, config)
    chart = EntranceCovarianceChart(reference)
    assert chart.coordinate_names == ENTRANCE_COVARIANCE_COORDINATE_NAMES
    assert chart.n_coordinates == 10
    np.testing.assert_allclose(
        chart.covariance(jnp.zeros(chart.n_coordinates)),
        reference,
        rtol=3e-15,
        atol=3e-27,
    )


def test_entrance_covariance_chart_is_spd_jittable_and_differentiable(
    column_and_config,
):
    column, config = column_and_config
    _, _, _, reference, _, _, _ = _neoarm_reconnaissance_contract(column, config)
    chart = EntranceCovarianceChart(reference)
    coordinates = jnp.linspace(-0.35, 0.4, chart.n_coordinates)
    covariance = np.asarray(jax.jit(chart.covariance)(coordinates))
    np.testing.assert_allclose(covariance, covariance.T, rtol=0.0, atol=1e-27)
    assert np.linalg.eigvalsh(covariance).min() > 0.0
    derivative = np.asarray(jax.jacfwd(chart.covariance)(coordinates))
    assert derivative.shape == (4, 4, chart.n_coordinates)
    assert np.all(np.isfinite(derivative))


def test_entrance_covariance_chart_has_all_ten_covariance_tangent_modes(
    column_and_config,
):
    column, config = column_and_config
    _, _, _, reference, _, _, _ = _neoarm_reconnaissance_contract(column, config)
    chart = EntranceCovarianceChart(reference)
    tangent = np.asarray(
        jax.jacfwd(lambda eta: chart.covariance(eta).reshape(-1))(
            jnp.zeros(chart.n_coordinates)
        )
    )
    # Normalize columns before the rank decision because x/theta entries carry
    # intentionally different SI units.  The chart itself spans Sym(4).
    tangent = tangent / np.linalg.norm(tangent, axis=0, keepdims=True)
    singular = np.linalg.svd(tangent, compute_uv=False)
    assert np.sum(singular > 1.0e-10 * singular[0]) == 10


def test_covariance_nuisance_audit_preserves_condenser_rank_and_profiles_upstream(
    column_and_config,
):
    """The four-state pilot is 12/12 condenser and 15/16 full upstream."""

    column, config = column_and_config
    (
        parameters,
        reference,
        source_mean,
        source_covariance,
        pairs,
        wobble,
        parameter_scales,
    ) = _neoarm_reconnaissance_contract(column, config)
    chart = EntranceCovarianceChart(source_covariance)
    aperture = config["beam"]["condenser_aperture"]

    def baseline_prediction(fitted_parameters, fitted_covariance):
        return condenser_independent_reconnaissance_response(
            column,
            fitted_parameters,
            pairs,
            source_mean,
            fitted_covariance,
            aperture_z_m=aperture["z_m"],
            aperture_radius_m=aperture["radius_m"],
            radial_order=16,
            azimuthal_order=32,
        )

    baseline_truth = np.asarray(
        baseline_prediction(jnp.asarray(parameters), jnp.asarray(source_covariance))
    )
    pair_count = pairs.shape[0]
    incident_electrons = float(aperture["incident_electrons_per_frame"])
    baseline_sigma = np.concatenate(
        (
            np.sqrt(
                np.maximum(
                    baseline_truth[: 2 * pair_count]
                    * incident_electrons,
                    1.0,
                )
            )
            / incident_electrons,
            np.full(2 * pair_count, 0.002),
        )
    )
    multivariate_values = np.asarray(
        (
            (-0.13526724, -0.20884910, 0.24530710),
            (-0.09839488, 0.12788477, -0.34999685),
            (0.05658233, -0.27411085, -0.17553819),
            (-0.23130277, 0.28816942, 0.27868015),
        )
    )
    probe_states = np.repeat(reference[None, :], len(multivariate_values), axis=0)
    probe_states[:, :3] = multivariate_values

    # The public released-covariance predictor is directly differentiable and
    # retains the state/observable axes needed by a nonlinear likelihood.
    probe_jacobian = jax.jacfwd(
        lambda eta: sample_plane_probe_observables_with_entrance_covariance(
            column,
            jnp.asarray(parameters),
            eta,
            probe_states,
            source_mean,
            chart,
            mode="complex_radial",
        )
    )(jnp.zeros(chart.n_coordinates))
    assert probe_jacobian.shape == (4, 3, 10)
    assert np.all(np.isfinite(np.asarray(probe_jacobian)))

    condenser_indices = condenser_parameter_indices(column)
    condenser_audit = audit_entrance_covariance_nuisance_identifiability(
        column,
        parameters,
        probe_states,
        source_mean,
        chart,
        probe_standard_deviations=0.002,
        parameter_scales=parameter_scales,
        parameter_indices=condenser_indices,
        baseline_prediction_function=baseline_prediction,
        baseline_standard_deviations=baseline_sigma,
        rank_rtol=1.0e-7,
    )
    assert condenser_audit.conditional_rank == 12
    assert condenser_audit.nuisance_profiled_rank == 12
    assert condenser_audit.nuisance_rank == 7

    upstream_indices = np.asarray(
        [
            index
            for index, name in enumerate(column.parameter_names)
            if name == "source_z_m"
            or name.startswith(("CL1.", "CL3.", "C-mini.", "Obj-prefield."))
        ]
    )
    upstream_audit = audit_entrance_covariance_nuisance_identifiability(
        column,
        parameters,
        probe_states,
        source_mean,
        chart,
        probe_standard_deviations=0.002,
        parameter_scales=parameter_scales,
        parameter_indices=upstream_indices,
        baseline_prediction_function=baseline_prediction,
        baseline_standard_deviations=baseline_sigma,
        rank_rtol=1.0e-7,
    )
    assert upstream_audit.conditional_rank == 16
    assert upstream_audit.nuisance_profiled_rank == 15
    assert upstream_audit.nuisance_rank == 7
    assert upstream_audit.diagnostics.block_names == (
        "condenser_reconnaissance",
        "sample_plane_probes",
    )
    assert not upstream_audit.parameter_indices.flags.writeable


def test_second_known_plane_closes_upstream_only_with_full_phase_space(
    column_and_config,
):
    """A screen covariance stays 15/16; two full covariance charts give 16/16."""

    column, config = column_and_config
    (
        parameters,
        reference,
        source_mean,
        source_covariance,
        pairs,
        _,
        parameter_scales,
    ) = _neoarm_reconnaissance_contract(column, config)
    chart = EntranceCovarianceChart(source_covariance)
    aperture = config["beam"]["condenser_aperture"]
    multivariate_values = np.asarray(
        (
            (-0.13526724, -0.20884910, 0.24530710),
            (-0.09839488, 0.12788477, -0.34999685),
            (0.05658233, -0.27411085, -0.17553819),
            (-0.23130277, 0.28816942, 0.27868015),
        )
    )
    probe_states = np.repeat(reference[None, :], len(multivariate_values), axis=0)
    probe_states[:, :3] = multivariate_values

    # These are algebraically non-duplicated per-frame transmissions and
    # absolute radii.  The derived signed radius difference is not counted
    # again; a real likelihood retains their same-image statistical covariance.
    def baseline_prediction(fitted_parameters, fitted_covariance):
        return condenser_independent_reconnaissance_response(
            column,
            fitted_parameters,
            pairs,
            source_mean,
            fitted_covariance,
            aperture_z_m=aperture["z_m"],
            aperture_radius_m=aperture["radius_m"],
            radial_order=16,
            azimuthal_order=32,
        )

    reconnaissance_truth = np.asarray(
        condenser_independent_reconnaissance_response(
            column,
            parameters,
            pairs,
            source_mean,
            source_covariance,
            aperture_z_m=aperture["z_m"],
            aperture_radius_m=aperture["radius_m"],
            radial_order=16,
            azimuthal_order=32,
        )
    )
    pair_count = pairs.shape[0]
    incident_electrons = float(aperture["incident_electrons_per_frame"])
    baseline_sigma = np.concatenate(
        (
            np.sqrt(
                np.maximum(
                    reconnaissance_truth[: 2 * pair_count] * incident_electrons,
                    1.0,
                )
            )
            / incident_electrons,
            np.full(2 * pair_count, 0.002),
        )
    )
    upstream_indices = np.asarray(
        [
            index
            for index, name in enumerate(column.parameter_names)
            if name == "source_z_m"
            or name.startswith(("CL1.", "CL3.", "C-mini.", "Obj-prefield."))
        ]
    )
    sample_full = AxialPlaneCovarianceBlock(
        name="sample_phase_space_covariance",
        plane_z_m=column.sample_z_m,
        dac_values=probe_states,
        standard_deviations=0.002,
        mode="full",
    )
    aperture_position = AxialPlaneCovarianceBlock(
        name="aperture_position_covariance",
        plane_z_m=aperture["z_m"],
        dac_values=probe_states,
        standard_deviations=0.002,
        mode="position",
    )
    position_only = audit_axial_plane_covariance_nuisance_identifiability(
        column,
        parameters,
        chart,
        (sample_full, aperture_position),
        parameter_scales=parameter_scales,
        parameter_indices=upstream_indices,
        baseline_prediction_function=baseline_prediction,
        baseline_standard_deviations=baseline_sigma,
        baseline_block_name="independent_reconnaissance",
        rank_rtol=1.0e-7,
    )
    assert position_only.conditional_rank == 16
    assert position_only.nuisance_profiled_rank == 15
    assert position_only.nuisance_rank == 10
    position_singular = (
        position_only.diagnostics.combined
        .column_equilibrated_profiled_optical.singular_values
    )
    assert position_singular[-1] / position_singular[0] == pytest.approx(
        1.823881e-8, rel=5e-3
    )
    assert position_singular[-2] / position_singular[0] > 1.0e-5

    aperture_full = AxialPlaneCovarianceBlock(
        name="aperture_phase_space_covariance",
        plane_z_m=aperture["z_m"],
        dac_values=probe_states,
        standard_deviations=0.002,
        mode="full",
    )
    full_phase_space = audit_axial_plane_covariance_nuisance_identifiability(
        column,
        parameters,
        chart,
        (sample_full, aperture_full),
        parameter_scales=parameter_scales,
        parameter_indices=upstream_indices,
        baseline_prediction_function=baseline_prediction,
        baseline_standard_deviations=baseline_sigma,
        baseline_block_name="independent_reconnaissance",
        rank_rtol=1.0e-7,
    )
    assert full_phase_space.conditional_rank == 16
    assert full_phase_space.nuisance_profiled_rank == 16
    assert full_phase_space.nuisance_rank == 10
    assert full_phase_space.diagnostics.block_names == (
        "independent_reconnaissance",
        "sample_phase_space_covariance",
        "aperture_phase_space_covariance",
    )
    profiled_singular = (
        full_phase_space.diagnostics.combined
        .column_equilibrated_profiled_optical.singular_values
    )
    weakest_relative = profiled_singular[-1] / profiled_singular[0]
    assert weakest_relative == pytest.approx(3.966822e-5, rel=5e-3)
    assert not sample_full.dac_values.flags.writeable
    assert not sample_full.flattened_standard_deviations().flags.writeable


def test_neoarm_condenser_rank_and_minimal_multiscale_sample_extension(
    column_and_config,
):
    """Regression for the exact current and proposed conditional ranks."""

    column, config = column_and_config
    (
        parameters,
        reference,
        source_mean,
        source_covariance,
        pairs,
        wobble,
        parameter_scales,
    ) = _neoarm_reconnaissance_contract(column, config)
    aperture = config["beam"]["condenser_aperture"]

    def current_prediction(fitted_parameters):
        return condenser_reconnaissance_response(
            column,
            fitted_parameters,
            pairs,
            source_mean,
            source_covariance,
            wobble_step=wobble,
            aperture_z_m=aperture["z_m"],
            aperture_radius_m=aperture["radius_m"],
            radial_order=16,
            azimuthal_order=32,
        )

    truth = np.asarray(current_prediction(jnp.asarray(parameters)))
    pair_count = pairs.shape[0]
    incident_electrons = float(aperture["incident_electrons_per_frame"])
    standard_deviations = np.concatenate(
        (
            np.full(pair_count, np.sqrt(2.0) * 0.002 / (2.0 * wobble)),
            np.sqrt(np.maximum(truth[pair_count:] * incident_electrons, 1.0))
            / incident_electrons,
        )
    )
    current = audit_condenser_identifiability(
        column,
        current_prediction,
        parameters,
        standard_deviations=standard_deviations,
        parameter_scales=parameter_scales,
        rank_rtol=1.0e-7,
    )
    assert current.rank == 7
    assert current.nullity == 5
    cmini_slope = current.parameter_names.index("C-mini.q_slope_per_dac")
    np.testing.assert_allclose(current.scaled_jacobian[:, cmini_slope], 0.0)

    # Four safe-range candidates per condenser.  A complex radial probe state
    # supplies only two new independent symplectic covariance combinations,
    # so three optical states are the minimum that can close a five-dimensional
    # baseline null space.
    candidates = []
    labels = []
    for channel_name in ("CL1", "CL3", "C-mini"):
        channel = column.channel_names.index(channel_name)
        for value in (-0.35, -0.175, 0.175, 0.35):
            dacs = reference.copy()
            dacs[channel] = value
            candidates.append(dacs)
            labels.append(f"{channel_name}{value:+.3f}")
    selection = greedy_condenser_probe_design(
        column,
        parameters,
        np.stack(candidates),
        source_mean,
        source_covariance,
        baseline_scaled_jacobian=current.scaled_jacobian,
        parameter_scales=parameter_scales,
        observable_standard_deviations=0.002,
        observable_mode="complex_radial",
        rank_rtol=1.0e-7,
    )
    assert selection.baseline_rank == 7
    assert selection.achieved_rank == selection.target_rank == 12
    assert len(selection.selected_candidate_indices) == 3
    assert [labels[index] for index in selection.selected_candidate_indices] == [
        "C-mini+0.350",
        "C-mini-0.350",
        "CL3+0.350",
    ]
    assert selection.reached_target
    assert selection.singular_values[-1] / selection.singular_values[0] > 1e-6
    assert not selection.selected_dac_values.flags.writeable

    # The two absolute radii were already estimated for every signed-wobble
    # exposure.  Use the algebraically non-duplicated per-frame transmissions
    # and radii in the likelihood: appending their derived signed difference
    # would count the same radius noise twice.  Their measured same-image
    # covariance may still be non-diagonal.  The absolute radii add one rank at
    # zero frame cost.  Pure rank closure then chooses two C-mini states, but an
    # explicit conditioning gate adds a third state before stopping.
    def retained_radius_prediction(fitted_parameters):
        return condenser_independent_reconnaissance_response(
            column,
            fitted_parameters,
            pairs,
            source_mean,
            source_covariance,
            aperture_z_m=aperture["z_m"],
            aperture_radius_m=aperture["radius_m"],
            radial_order=16,
            azimuthal_order=32,
        )

    retained_truth = np.asarray(retained_radius_prediction(jnp.asarray(parameters)))
    retained_standard_deviations = np.concatenate(
        (
            np.sqrt(
                np.maximum(
                    retained_truth[: 2 * pair_count] * incident_electrons,
                    1.0,
                )
            )
            / incident_electrons,
            np.full(2 * pair_count, 0.002),
        )
    )
    retained = audit_condenser_identifiability(
        column,
        retained_radius_prediction,
        parameters,
        standard_deviations=retained_standard_deviations,
        parameter_scales=parameter_scales,
        rank_rtol=1.0e-7,
    )
    assert retained.rank == 8
    rank_only = greedy_condenser_probe_design(
        column,
        parameters,
        np.stack(candidates),
        source_mean,
        source_covariance,
        baseline_scaled_jacobian=retained.scaled_jacobian,
        parameter_scales=parameter_scales,
        observable_standard_deviations=0.002,
        observable_mode="complex_radial",
        rank_rtol=1.0e-7,
    )
    assert rank_only.achieved_rank == 12
    assert len(rank_only.selected_candidate_indices) == 2
    assert [labels[index] for index in rank_only.selected_candidate_indices] == [
        "C-mini+0.350",
        "C-mini-0.350",
    ]
    assert rank_only.achieved_relative_singular_value == pytest.approx(
        2.30334e-7, rel=2e-3
    )
    assert rank_only.conditioning_target_met

    conditioned = greedy_condenser_probe_design(
        column,
        parameters,
        np.stack(candidates),
        source_mean,
        source_covariance,
        baseline_scaled_jacobian=retained.scaled_jacobian,
        parameter_scales=parameter_scales,
        observable_standard_deviations=0.002,
        observable_mode="complex_radial",
        rank_rtol=1.0e-7,
        minimum_relative_singular_value=1.0e-6,
    )
    assert conditioned.achieved_rank == 12
    assert len(conditioned.selected_candidate_indices) == 3
    assert conditioned.selected_candidate_indices[:2].tolist() == (
        rank_only.selected_candidate_indices.tolist()
    )
    assert [labels[index] for index in conditioned.selected_candidate_indices] == [
        "C-mini+0.350",
        "C-mini-0.350",
        "CL3+0.350",
    ]
    assert conditioned.achieved_relative_singular_value == pytest.approx(
        1.62109e-6, rel=2e-3
    )
    assert conditioned.achieved_relative_singular_value >= 1.0e-6
    assert conditioned.conditioning_target_met
    assert conditioned.reached_target

    # If the calibrated source plane and fixed objective-prefield parameters
    # are released too, the same current experiment is 8/16.  Four
    # multivariate probe states are sufficient for local 16/16 rank in this
    # synthetic safe box (each radial symplectic block adds at most two ranks).
    upstream_indices = np.asarray(
        [
            index
            for index, name in enumerate(column.parameter_names)
            if name == "source_z_m"
            or name.startswith(("CL1.", "CL3.", "C-mini.", "Obj-prefield."))
        ]
    )
    upstream_current = audit_condenser_identifiability(
        column,
        current_prediction,
        parameters,
        standard_deviations=standard_deviations,
        parameter_scales=parameter_scales,
        parameter_indices=upstream_indices,
        rank_rtol=1.0e-7,
    )
    assert upstream_current.rank == 8
    multivariate_values = np.asarray(
        (
            (-0.13526724, -0.20884910, 0.24530710),
            (-0.09839488, 0.12788477, -0.34999685),
            (0.05658233, -0.27411085, -0.17553819),
            (-0.23130277, 0.28816942, 0.27868015),
        )
    )
    multivariate_candidates = np.repeat(
        reference[None, :], len(multivariate_values), axis=0
    )
    multivariate_candidates[:, :3] = multivariate_values
    upstream_selection = greedy_condenser_probe_design(
        column,
        parameters,
        multivariate_candidates,
        source_mean,
        source_covariance,
        baseline_scaled_jacobian=upstream_current.scaled_jacobian,
        parameter_scales=parameter_scales,
        observable_standard_deviations=0.002,
        observable_mode="complex_radial",
        parameter_indices=upstream_indices,
        rank_rtol=1.0e-7,
    )
    assert upstream_selection.baseline_rank == 8
    assert upstream_selection.achieved_rank == 16
    assert len(upstream_selection.selected_candidate_indices) == 4
    assert upstream_selection.singular_values[-1] / upstream_selection.singular_values[0] > 2e-7


def test_reconnaissance_retains_absolute_radius_without_extra_frames(
    column_and_config,
):
    column, config = column_and_config
    (
        parameters,
        _,
        source_mean,
        source_covariance,
        pairs,
        wobble,
        _,
    ) = _neoarm_reconnaissance_contract(column, config)
    aperture = config["beam"]["condenser_aperture"]
    log_radius, transmission = condenser_reconnaissance_frame_observables(
        column,
        parameters,
        pairs,
        source_mean,
        source_covariance,
        aperture_z_m=aperture["z_m"],
        aperture_radius_m=aperture["radius_m"],
        radial_order=12,
        azimuthal_order=32,
    )
    compact = condenser_reconnaissance_response(
        column,
        parameters,
        pairs,
        source_mean,
        source_covariance,
        wobble_step=wobble,
        aperture_z_m=aperture["z_m"],
        aperture_radius_m=aperture["radius_m"],
        radial_order=12,
        azimuthal_order=32,
    )
    retained = condenser_reconnaissance_response(
        column,
        parameters,
        pairs,
        source_mean,
        source_covariance,
        wobble_step=wobble,
        aperture_z_m=aperture["z_m"],
        aperture_radius_m=aperture["radius_m"],
        retain_absolute_log_radius=True,
        radial_order=12,
        azimuthal_order=32,
    )
    pair_count = pairs.shape[0]
    independent = condenser_independent_reconnaissance_response(
        column,
        parameters,
        pairs,
        source_mean,
        source_covariance,
        aperture_z_m=aperture["z_m"],
        aperture_radius_m=aperture["radius_m"],
        radial_order=12,
        azimuthal_order=32,
    )
    assert compact.shape == (3 * pair_count,)
    assert retained.shape == (5 * pair_count,)
    assert independent.shape == (4 * pair_count,)
    np.testing.assert_allclose(compact[pair_count:], transmission.reshape(-1))
    np.testing.assert_allclose(retained[-2 * pair_count :], log_radius.reshape(-1))
    np.testing.assert_allclose(independent[: 2 * pair_count], transmission.reshape(-1))
    np.testing.assert_allclose(independent[2 * pair_count :], log_radius.reshape(-1))
    derivative = jax.jacfwd(
        lambda fitted: condenser_reconnaissance_response(
            column,
            fitted,
            pairs[:1],
            source_mean,
            source_covariance,
            wobble_step=wobble,
            aperture_z_m=aperture["z_m"],
            aperture_radius_m=aperture["radius_m"],
            radial_order=12,
            azimuthal_order=32,
        )
    )(jnp.asarray(parameters))
    assert derivative.shape == (3, column.n_parameters)
    assert np.all(np.isfinite(np.asarray(derivative)))
