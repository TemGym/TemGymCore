from pathlib import Path
import tomllib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.critical_line_calibration import (
    ApertureEdgeScanDesign,
    aperture_edge_scan_from_moments,
    condenser_aperture_edge_scan,
    hard_edge_matrix_to_plane,
    position_moments,
    three_normal_aperture_edge_design,
)
from temgym_core.neoarm_calibration import hard_edge_column_from_config


jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def column():
    path = (
        Path(__file__).parents[1]
        / "tests"
        / "fixtures"
        / "neoarm_synthetic_column.toml"
    )
    with path.open("rb") as stream:
        return hard_edge_column_from_config(tomllib.load(stream))


@pytest.fixture(scope="module")
def physical_aperture():
    # Current NEOARM microscope-model geometry: this intentionally lives in a
    # test fixture, not in the reusable implementation or synthetic config.
    return {
        "z_m": 0.250,
        "radius_m": 10.0e-6,
        "dose_electrons": 2.0e6,
        "edge_step_m": 1.25e-6,
    }


@pytest.fixture(scope="module")
def entrance_distribution():
    mean = np.asarray((0.4e-6, 8.0e-6, -0.25e-6, -5.0e-6))
    covariance = np.diag(
        (2.0e-6**2, 20.0e-6**2, 2.4e-6**2, 17.0e-6**2)
    )
    return mean, covariance


def test_three_normal_design_is_a_22_raw_frame_contract(physical_aperture):
    design = three_normal_aperture_edge_design(
        physical_aperture["radius_m"], physical_aperture["edge_step_m"]
    )

    assert design.n_normals == 3
    assert design.n_offsets == 7
    assert design.n_scan_frames == 21
    assert design.n_reference_frames == 1
    assert design.n_frames == 22
    assert design.aperture_centre_offsets_xy_m.shape == (21, 2)
    assert design.scan_normal_indices.tolist() == [
        normal for normal in range(3) for _ in range(7)
    ]
    assert design.frame_kinds == (
        ("aperture_edge",) * 21 + ("aperture_out_dose_reference",)
    )
    np.testing.assert_allclose(
        np.linalg.norm(design.scan_normals_xy, axis=1), 1.0, atol=2e-15
    )
    np.testing.assert_allclose(
        design.scan_normals_xy.sum(axis=0), 0.0, atol=5e-15
    )
    assert not design.aperture_centre_offsets_xy_m.flags.writeable


def test_design_accepts_one_or_a_bank_of_two_dimensional_normals(
    physical_aperture,
):
    one_normal = ApertureEdgeScanDesign(
        scan_normals_xy=np.asarray((4.0, 0.0)),
        signed_edge_offsets_m=np.asarray((-1.0e-6, 0.0, 1.0e-6)),
        reference_radius_m=physical_aperture["radius_m"],
        include_aperture_out_reference=False,
    )
    assert one_normal.aperture_centre_offsets_xy_m.shape == (3, 2)
    np.testing.assert_allclose(one_normal.scan_normals_xy, ((1.0, 0.0),))
    assert one_normal.n_frames == 3

    two_normals = ApertureEdgeScanDesign(
        scan_normals_xy=np.asarray(((1.0, 0.0), (1.0, 1.0))),
        signed_edge_offsets_m=np.asarray((0.0, 1.0e-6)),
        reference_radius_m=physical_aperture["radius_m"],
    )
    assert two_normals.aperture_centre_offsets_xy_m.shape == (4, 2)
    np.testing.assert_allclose(
        two_normals.scan_normals_xy[1], np.asarray((1.0, 1.0)) / np.sqrt(2.0)
    )
    assert two_normals.n_frames == 5


def test_aperture_translation_and_opposite_beam_translation_are_identical(
    physical_aperture,
):
    sigma = 3.6e-6
    offsets = np.asarray(
        ((-7.0e-6, 1.0e-6), (2.0e-6, -4.0e-6), (8.0e-6, 3.0e-6))
    )
    common = dict(
        beam_mean_xy_m=np.asarray((0.3e-6, -0.2e-6)),
        beam_covariance_xy_m2=np.asarray(
            ((sigma**2, 0.17 * sigma**2), (0.17 * sigma**2, 1.3 * sigma**2))
        ),
        aperture_centre_xy_m=np.asarray((0.1e-6, 0.25e-6)),
        aperture_radius_m=physical_aperture["radius_m"],
        incident_dose_electrons=physical_aperture["dose_electrons"],
        radial_order=32,
        azimuthal_order=96,
    )
    moved_aperture = aperture_edge_scan_from_moments(
        **common, aperture_centre_offsets_xy_m=offsets
    )
    moved_beam = aperture_edge_scan_from_moments(
        **common, beam_offsets_xy_m=-offsets
    )

    np.testing.assert_allclose(
        moved_aperture.relative_beam_mean_xy_m,
        moved_beam.relative_beam_mean_xy_m,
        rtol=0.0,
        atol=2e-21,
    )
    np.testing.assert_allclose(
        moved_aperture.transmission,
        moved_beam.transmission,
        rtol=3e-13,
        atol=3e-13,
    )


def test_batched_explicit_nuisances_give_absolute_counts_and_reference(
    physical_aperture,
):
    beam_mean = np.asarray(((0.0, 0.0), (0.6e-6, -0.4e-6)))
    covariance = np.stack(
        (
            np.diag((3.0e-6**2, 4.0e-6**2)),
            np.asarray(((4.0e-6**2, 2.0e-12), (2.0e-12, 3.5e-6**2))),
        )
    )
    centres = np.asarray(((0.0, 0.0), (0.25e-6, -0.1e-6)))
    radii = physical_aperture["radius_m"] * np.asarray((0.98, 1.03))
    doses = physical_aperture["dose_electrons"] * np.asarray((0.8, 1.25))
    state_offsets = np.asarray(
        (
            ((8.0e-6, 0.0), (10.0e-6, 0.0)),
            ((0.0, 8.0e-6), (0.0, 10.0e-6)),
        )
    )
    prediction = aperture_edge_scan_from_moments(
        beam_mean,
        covariance,
        aperture_centre_xy_m=centres,
        aperture_radius_m=radii,
        incident_dose_electrons=doses,
        aperture_centre_offsets_xy_m=state_offsets,
        radial_order=24,
        azimuthal_order=64,
    )

    assert prediction.beam_mean_xy_m.shape == (2, 2)
    assert prediction.beam_covariance_xy_m2.shape == (2, 2, 2)
    assert prediction.transmission.shape == (2, 2)
    assert prediction.expected_transmitted_electrons.shape == (2, 2)
    assert prediction.aperture_out_expected_electrons.shape == (2,)
    np.testing.assert_allclose(
        prediction.expected_transmitted_electrons,
        prediction.transmission * doses[:, None],
        rtol=2e-14,
    )
    np.testing.assert_allclose(prediction.aperture_out_expected_electrons, doses)
    assert np.all((prediction.transmission > 0.0) & (prediction.transmission < 1.0))


def test_column_scan_exposes_physical_plane_moments_and_raw_counts(
    column,
    physical_aperture,
    entrance_distribution,
):
    source_mean, source_covariance = entrance_distribution
    design = three_normal_aperture_edge_design(
        physical_aperture["radius_m"], physical_aperture["edge_step_m"]
    )
    dacs = np.zeros((2, len(column.channel_names)))
    dacs[1, column.channel_names.index("CL1")] = 0.08
    centres = np.asarray(((0.2e-6, -0.1e-6), (-0.3e-6, 0.25e-6)))
    radii = physical_aperture["radius_m"] * np.asarray((1.0, 1.015))
    doses = physical_aperture["dose_electrons"] * np.asarray((1.0, 0.9))

    prediction = condenser_aperture_edge_scan(
        column,
        column.nominal_parameters,
        dacs,
        source_mean,
        source_covariance,
        aperture_z_m=physical_aperture["z_m"],
        aperture_centre_xy_m=centres,
        aperture_radius_m=radii,
        incident_dose_electrons=doses,
        aperture_centre_offsets_xy_m=design.aperture_centre_offsets_xy_m,
        radial_order=24,
        azimuthal_order=64,
    )
    direct_matrix = hard_edge_matrix_to_plane(
        column,
        column.nominal_parameters,
        dacs[0],
        physical_aperture["z_m"],
    )
    direct_mean, direct_covariance = position_moments(
        direct_matrix, source_mean, source_covariance
    )

    np.testing.assert_allclose(prediction.beam_mean_xy_m[0], direct_mean)
    np.testing.assert_allclose(prediction.beam_covariance_xy_m2[0], direct_covariance)
    raw_counts = design.raw_expected_counts(prediction)
    assert raw_counts.shape == (2, 22)
    np.testing.assert_allclose(raw_counts[:, -1], doses)
    assert np.all(np.isfinite(raw_counts))
    assert np.all(raw_counts[:, :-1] <= doses[:, None])


def test_column_scan_is_jittable_and_differentiable_in_optics_and_nuisances(
    column,
    physical_aperture,
    entrance_distribution,
):
    source_mean, source_covariance = entrance_distribution
    design = three_normal_aperture_edge_design(
        physical_aperture["radius_m"], physical_aperture["edge_step_m"]
    )
    dacs = np.zeros(len(column.channel_names))
    dacs[column.channel_names.index("CL3")] = -0.06
    offsets = design.aperture_centre_offsets_xy_m

    def prediction_vector(parameters, nuisance):
        centre = nuisance[:2]
        radius = physical_aperture["radius_m"] * jnp.exp(nuisance[2])
        dose = physical_aperture["dose_electrons"] * jnp.exp(nuisance[3])
        prediction = condenser_aperture_edge_scan(
            column,
            parameters,
            dacs,
            source_mean,
            source_covariance,
            aperture_z_m=physical_aperture["z_m"],
            aperture_centre_xy_m=centre,
            aperture_radius_m=radius,
            incident_dose_electrons=dose,
            aperture_centre_offsets_xy_m=offsets,
            radial_order=16,
            azimuthal_order=48,
        )
        # Scaling the absolute counts preserves their information while keeping
        # the test Jacobian numerically comparable to transmission.
        return jnp.concatenate(
            (
                prediction.transmission.reshape(-1),
                prediction.expected_transmitted_electrons.reshape(-1)
                / physical_aperture["dose_electrons"],
                prediction.aperture_out_expected_electrons
                / physical_aperture["dose_electrons"],
            )
        )

    nominal_nuisance = jnp.zeros(4)
    compiled = jax.jit(prediction_vector)(
        jnp.asarray(column.nominal_parameters), nominal_nuisance
    )
    assert compiled.shape == (43,)
    assert np.all(np.isfinite(np.asarray(compiled)))

    nuisance_jacobian = np.asarray(
        jax.jacfwd(lambda nuisance: prediction_vector(
            jnp.asarray(column.nominal_parameters), nuisance
        ))(nominal_nuisance)
    )
    assert nuisance_jacobian.shape == (43, 4)
    assert np.all(np.isfinite(nuisance_jacobian))
    assert np.all(np.linalg.norm(nuisance_jacobian, axis=0) > 1e-5)

    optical_jacobian = np.asarray(
        jax.jacfwd(lambda parameters: prediction_vector(
            parameters, nominal_nuisance
        ))(jnp.asarray(column.nominal_parameters))
    )
    assert optical_jacobian.shape == (43, column.n_parameters)
    assert np.all(np.isfinite(optical_jacobian))
    assert np.linalg.norm(optical_jacobian) > 1e-6


def test_aperture_radius_remains_a_differentiable_nuisance(physical_aperture):
    sigma = 4.0e-6
    derivative = jax.grad(
        lambda log_radius: aperture_edge_scan_from_moments(
            np.zeros(2),
            np.eye(2) * sigma**2,
            aperture_centre_xy_m=np.zeros(2),
            aperture_radius_m=physical_aperture["radius_m"] * jnp.exp(log_radius),
            incident_dose_electrons=physical_aperture["dose_electrons"],
            radial_order=24,
            azimuthal_order=64,
        ).transmission[0, 0]
    )(jnp.asarray(0.0))
    assert np.isfinite(float(derivative))
    assert float(derivative) > 0.0


def test_scan_rejects_hidden_or_ambiguous_nuisance_contracts(physical_aperture):
    common = dict(
        beam_mean_xy_m=np.zeros(2),
        beam_covariance_xy_m2=np.eye(2) * (3.0e-6) ** 2,
        aperture_centre_xy_m=np.zeros(2),
        aperture_radius_m=physical_aperture["radius_m"],
        incident_dose_electrons=physical_aperture["dose_electrons"],
    )
    with pytest.raises(ValueError, match="not both"):
        aperture_edge_scan_from_moments(
            **common,
            aperture_centre_offsets_xy_m=np.zeros((2, 2)),
            beam_offsets_xy_m=np.zeros((2, 2)),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        aperture_edge_scan_from_moments(
            **(common | {"incident_dose_electrons": 0.0})
        )
    with pytest.raises(ValueError, match="strictly positive"):
        aperture_edge_scan_from_moments(
            **(common | {"aperture_radius_m": -1.0})
        )
