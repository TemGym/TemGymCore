import numpy as np
import jax
import jax.numpy as jnp
import pytest

from temgym_core.gaussian_action2D import (
    make_gaussian,
    ABCDPropagator2D,
    Lens,
    run_to_end,
)
from temgym_core.components import Detector

from temgym_core.utils import energy2wavelength

jax.config.update("jax_enable_x64", True)


def test_free_space_paraxial_updates_q_inv():
    # Analytic solution to free space propagation of complex q_inv parameter
    # of gaussian.
    voltage = 100e3
    r_curv = -0.001
    waist = 1e-6
    wavelength = energy2wavelength(voltage)
    q_inv = 1/r_curv + 1j * (wavelength) / (jnp.pi * waist ** 2)
    Q_inv = jnp.diag(jnp.array([q_inv, q_inv], dtype=jnp.complex128))

    ray_in = make_gaussian(
        x=0.0,
        y=0.0,
        dx=0.0,
        dy=0.0,
        z=0.0,
        voltage=voltage,
        waist_x=waist,
        waist_y=waist,
        InitPhase=0.0,
        InitAmp=1.0,
        RadiusOfCurvature_x=r_curv,
        RadiusOfCurvature_y=r_curv,
    )

    dist = 1e-3
    propagator = ABCDPropagator2D.free_space(dist)
    ray_out = propagator(ray_in)

    I2 = jnp.eye(2, dtype=jnp.complex128)
    expected_Q = Q_inv @ jnp.linalg.inv(I2 + dist * Q_inv)

    np.testing.assert_allclose(
        np.asarray(ray_out.S2),
        np.asarray(expected_Q),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(ray_in.C),
        np.asarray(1.0 + 0.0j),
        rtol=0.0,
        atol=1e-12,
    )
    k = 2 * jnp.pi / wavelength
    det_term = jnp.linalg.det(I2 + dist * Q_inv)
    expected_C = ray_in.C * jnp.exp(1j * k * dist) / jnp.sqrt(det_term)

    np.testing.assert_allclose(
        np.asarray(ray_out.C),
        np.asarray(expected_C),
        rtol=1e-12,
        atol=1e-12,
    )


def test_thin_lens_updates_q_inv():
    # Analytic solution for a thin lens: q_inv_out = q_inv_in - 1/f
    voltage = 100e3
    r_curv = -0.001
    waist = 1e-6
    wavelength = energy2wavelength(voltage)
    q_inv = 1 / r_curv + 1j * (wavelength) / (jnp.pi * waist ** 2)
    Q_inv = jnp.diag(jnp.array([q_inv, q_inv], dtype=jnp.complex128))

    ray_in = make_gaussian(
        x=0.0,
        y=0.0,
        dx=0.0,
        dy=0.0,
        z=0.0,
        voltage=voltage,
        waist_x=waist,
        waist_y=waist,
        InitPhase=0.0,
        InitAmp=1.0,
        RadiusOfCurvature_x=r_curv,
        RadiusOfCurvature_y=r_curv,
    )

    f = 5e-3

    # support either ABCDPropagator2D.thin_lens or ABCDPropagator2D.lens if present
    lens_ctor = getattr(ABCDPropagator2D, "thin_lens", None) or getattr(
        ABCDPropagator2D, "lens", None
    )
    if lens_ctor is None:
        pytest.skip("No thin_lens or lens constructor on ABCDPropagator2D")

    propagator = lens_ctor(f)
    ray_out = propagator(ray_in)

    I2 = jnp.eye(2, dtype=jnp.complex128)
    expected_Q = Q_inv - (1.0 / f) * I2

    np.testing.assert_allclose(
        np.asarray(ray_out.S2),
        np.asarray(expected_Q),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(ray_in.C),
        np.asarray(1.0 + 0.0j),
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(ray_out.C),
        np.asarray(ray_in.C),
        rtol=1e-12,
        atol=1e-12,
    )


def test_fourier_transform_ABCD_matrix_updates_q_inv():
    # Test general ABCD transform: A=0, B=f, C=-1/f, D=0
    voltage = 100000
    r_curv = jnp.inf
    waist = 3e-8
    wavelength = energy2wavelength(voltage)
    q_inv = 1 / r_curv + 1j * (wavelength) / (jnp.pi * waist ** 2)
    Q_inv = jnp.diag(jnp.array([q_inv, q_inv], dtype=jnp.complex128))

    ray_in = make_gaussian(
        x=0.0e-6,
        y=0.2e-6,
        dx=0.0,
        dy=0.0,
        z=0.0,
        voltage=voltage,
        waist_x=waist,
        waist_y=waist,
        InitPhase=0.0,
        InitAmp=1.0,
        RadiusOfCurvature_x=r_curv,
        RadiusOfCurvature_y=r_curv,
    )

    f = 1e-2
    A, B, C, D = 0.0, f, -1.0 / f, 0.0

    propagator = ABCDPropagator2D.fourier_transform(f)

    ray_out = propagator(ray_in)

    I2 = jnp.eye(2, dtype=jnp.complex128)
    expected_Q = (C * I2 + D * Q_inv) @ jnp.linalg.inv(A * I2 + B * Q_inv)

    np.testing.assert_allclose(
        np.asarray(ray_out.S2),
        np.asarray(expected_Q),
        rtol=1e-12,
        atol=1e-12,
    )

    # ---- Prefactor checks ----
    # Theory: C_out / C_in = det(A + B Q_in)^(-1/2) * exp(i k (dS0 + L))
    # For a centered beam with zero slopes, dS0 = 0.
    ABQ = (A * I2 + B * Q_inv)
    det_ABQ = jnp.linalg.det(ABQ)

    C_in = ray_in.C
    C_out = ray_out.C

    # Magnitude: |C_out/C_in| = 1 / sqrt(|det(A + B Q_in)|)
    exp_mag = 1.0 / jnp.sqrt(jnp.abs(det_ABQ))
    got_mag = jnp.abs(C_out) / jnp.abs(C_in)
    np.testing.assert_allclose(np.asarray(got_mag),
                               np.asarray(exp_mag),
                               rtol=1e-12, atol=1e-12)


def test_fourier_transform_ABCD_matrix_updates_against_stepwise():
    # Test general ABCD transform: A=0, B=f, C=-1/f, D=0
    voltage = 100000
    r_curv = jnp.inf
    waist = 3e-8

    ray_in = make_gaussian(
        x=0.0e-6,
        y=0.2e-6,
        dx=0.0,
        dy=0.0,
        z=0.0,
        voltage=voltage,
        waist_x=waist,
        waist_y=waist,
        InitPhase=0.0,
        InitAmp=1.0,
        RadiusOfCurvature_x=r_curv,
        RadiusOfCurvature_y=r_curv,
    )
    f = 1e-2
    ray_out_abcd = ABCDPropagator2D.fourier_transform(f)(ray_in)

    lens = Lens(z=f, focal_length=f)
    output_probe_grid = Detector(z=2 * f, pixel_size=(1, 1), shape=(1, 1))

    components = (lens, output_probe_grid)
    ray_out_step_wise = run_to_end(ray_in, components)

    propagator = ABCDPropagator2D.free_space(f)
    A = jnp.eye(2)
    B = jnp.zeros((2, 2))
    C = -1.0 / f * jnp.eye(2)
    D = jnp.eye(2)
    lens = ABCDPropagator2D(A=A, B=B, C=C, D=D)
    ray_out_first = propagator(ray_in)
    ray_out_lens = lens(ray_out_first)
    ray_out_step_wise_abcd = propagator(ray_out_lens)

    np.testing.assert_allclose(
        np.asarray(ray_out_abcd.S2),
        np.asarray(ray_out_step_wise_abcd.S2),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(ray_out_abcd.C),
        np.asarray(ray_out_step_wise_abcd.C),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(ray_out_abcd.S2),
        np.asarray(ray_out_step_wise.S2),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        np.asarray(ray_out_abcd.C),
        np.asarray(ray_out_step_wise.C),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        np.asarray(ray_out_step_wise_abcd.S2),
        np.asarray(ray_out_step_wise.S2),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        np.asarray(ray_out_step_wise_abcd.C),
        np.asarray(ray_out_step_wise.C),
        rtol=1e-7,
        atol=1e-7,
    )
