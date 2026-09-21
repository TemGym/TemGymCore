"""Quadratic aberrations must survive the polar singularity at the lens axis."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from temgym_core.aberrations import KrivanekCoeffs
from temgym_core.components import KrivanekLens
from temgym_core.source import gaussian_waist


@pytest.mark.parametrize('angle', [0., .4, np.pi/2])
def test_on_axis_astigmatism_matches_cartesian_pupil_hessian(angle):
    f=.02
    p=KrivanekCoeffs(C10=2e-6,C12=5e-6,phi12=angle,C21=1e-5,C30=1e-3)
    lens=KrivanekLens(z=0.,focal_length=f,coeffs=p)
    expected=-np.eye(2)/f-(p.C10*np.eye(2)+p.C12*np.array([
        [np.cos(2*angle),np.sin(2*angle)],
        [np.sin(2*angle),-np.cos(2*angle)]]))/f**2
    actual=jax.hessian(lens.phase_shift)(jnp.zeros(2))
    np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)
    beam=gaussian_waist(z=0.,voltage=30e3,rms_size=1e-5)
    result=lens(beam)
    np.testing.assert_allclose(result.Q_inv,beam.Q_inv+expected,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(result.r_xy,0.,atol=1e-20)
    np.testing.assert_allclose(result.d_xy,0.,atol=1e-20)
