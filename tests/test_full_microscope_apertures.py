"""Native aperture moments include clipping, diffraction and narrow crossovers."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar

from temgym_core.calibration.apertures import make_aperture_moments
from temgym_core.calibration.condenser_model import (
    condenser_with_aperture,
    generating_parameters,
    source_moments,
    specimen_moments,
)
from temgym_core.components import SigmoidAperture
from temgym_core.run import column_matrix


def parameters():
    return jnp.r_[
        generating_parameters(), np.log(0.0025 / (0.056 - 0.0025)), 0.2, -0.15
    ]


def test_aperture_is_an_inserted_native_component_between_cl3_and_mini():
    p = parameters()
    column = condenser_with_aperture(
        p, jnp.ones(3), radius_m=2e-6, edge_width_m=150e-9, centre_m=p[20:22] * 1e-6
    )
    aperture = column[2]
    assert isinstance(aperture, SigmoidAperture)
    assert column[1].z + column[1].length < aperture.z < column[3].z
    np.testing.assert_allclose([aperture.x0, aperture.y0], p[20:22] * 1e-6)


def test_large_inserted_aperture_approaches_lossless_native_transport():
    p = parameters()
    controls = jnp.array([1.37, 0.814, 1.145])
    predict = make_aperture_moments(
        radius_m=200e-6, edge_width_m=150e-9, voltage_v=200000.0
    )
    mean, cov, flux = predict(p, controls, p[20:22] * 1e-6)
    expected_mean, expected_cov = specimen_moments(p, controls)
    units = np.sqrt(np.diag(expected_cov))
    scale = units[:, None] * units[None, :]
    np.testing.assert_allclose(mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(cov / scale, expected_cov / scale, atol=1e-7)
    np.testing.assert_allclose(flux, 1.0, atol=1e-9)


def test_aperture_integral_and_derivatives_converge():
    p = parameters()
    controls = jnp.array([1.37, 0.814, 1.145])
    centre = p[20:22] * 1e-6
    coarse = make_aperture_moments(
        radius_m=2e-6, edge_width_m=150e-9, voltage_v=200000.0
    )
    fine = make_aperture_moments(
        radius_m=2e-6,
        edge_width_m=150e-9,
        voltage_v=200000.0,
        radial_order=384,
        angular_order=128,
    )
    a, b = coarse(p, controls, centre), fine(p, controls, centre)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, atol=1e-14, rtol=1e-7)

    def observable(p):
        m, s, f = coarse(p, controls, p[20:22] * 1e-6)
        return jnp.r_[
            m / jnp.array([1e-7, 1e-7, 1e-3, 1e-3]), s[0, 0] / 1e-14, s[2, 2] / 1e-6, f
        ]

    direction = (
        jnp.zeros(22)
        .at[jnp.array([0, 5, 9, 19, 20])]
        .set(jnp.array([0.1, -0.15, 0.05, 0.2, 0.03]))
    )
    derivative = jax.jacfwd(observable)(p) @ direction
    step = 1e-5
    difference = (
        observable(p + step * direction) - observable(p - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(derivative, difference, rtol=3e-5, atol=1e-6)


def test_crossover_is_resolved_away_from_aperture_grid_centre():
    p = parameters()
    centre = jnp.array([0.2e-6, -0.15e-6])

    def aperture_variance(cl1):
        column = condenser_with_aperture(
            p,
            jnp.array([cl1, 1.0, 1.0]),
            radius_m=2e-6,
            edge_width_m=150e-9,
            centre_m=centre,
        )
        matrix = column_matrix(column[:3], source_z=-jnp.sum(jnp.exp(p[:9])))[:4, :4]
        _, cov = source_moments(p)
        return (matrix @ cov @ matrix.T)[0, 0]

    variance = jax.jit(aperture_variance)
    cl1 = minimize_scalar(
        lambda c: float(variance(c)),
        bounds=(1.3, 3.0),
        method="bounded",
        options={"xatol": 1e-12},
    ).x
    coarse = make_aperture_moments(
        radius_m=2e-6, edge_width_m=150e-9, voltage_v=200000.0
    )
    fine = make_aperture_moments(
        radius_m=2e-6,
        edge_width_m=150e-9,
        voltage_v=200000.0,
        radial_order=384,
        angular_order=128,
    )
    a, b = (
        coarse(p, jnp.array([cl1, 1.0, 1.0]), centre),
        fine(p, jnp.array([cl1, 1.0, 1.0]), centre),
    )
    assert float(a[2]) > 0.999
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, atol=1e-13, rtol=1e-6)
