from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_last_lens_tomography import (
    J2, J4, mechanical_shear, signed_position_response, known_q_coefficients,
    geometry_from_coefficients, predict_response_parameters, geometry_coordinates,
)

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('kappa,angle', [(0.,0.), (800.,.31), (-470.,-.72)])
def test_full_volume_recovers_geometry_despite_unknown_sample_field_and_rotation(kappa, angle):
    upstream = jnp.diag(jnp.array([8.,5.,1/8,1/5]))@mechanical_shear(kappa)
    q = jnp.linspace(-3.05,-.08,19)
    h = signed_position_response(q,.02,.315,upstream,frame_rotation_rad=angle)
    w = known_q_coefficients(q,h)
    result = geometry_from_coefficients(w)
    assert result['length_m'] == pytest.approx(.02,rel=1e-10)
    assert result['detector_gap_m'] == pytest.approx(.315,rel=1e-10)
    assert result['inferred_kappa_per_m'] == pytest.approx(kappa,rel=1e-10,abs=1e-10)
    np.testing.assert_allclose(jnp.linalg.det(upstream),1.,rtol=1e-13)
    target = np.block([[2*kappa*np.asarray(J2),np.eye(2)],[-np.eye(2),np.zeros((2,2))]])
    np.testing.assert_allclose(result['sample_pullback_twoform'],target,rtol=1e-10,atol=1e-10)


def test_unknown_excitation_parameter_model_preserves_physical_input_units_and_jacobian():
    inputs = jnp.array([5e-9,5e-9,1e-3,1e-3]); pixel=55e-6
    f = jnp.diag(jnp.array([8.,5.,1/8,1/5]))@mechanical_shear(800.)
    s=.315/.02; w=jnp.stack((-s*f[:2]*inputs/pixel,.02*f[2:]*inputs/pixel))
    p=jnp.r_[jnp.log(s),-1.4,-.6,w.ravel()];t=jnp.linspace(-1.,1.,11)
    expected=signed_position_response(p[1]+p[2]*t,.02,.315,f)*inputs/pixel
    np.testing.assert_allclose(predict_response_parameters(p,t),expected,rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(jnp.exp(geometry_coordinates(p,inputs,pixel)[:2]),[.02,.315],rtol=1e-12)
    assert np.all(np.isfinite(jax.jacfwd(predict_response_parameters)(p,t)))
    assert np.all(np.isfinite(jax.jacfwd(geometry_coordinates)(p,inputs,pixel)))


def test_rank_deficient_excitation_or_nonphysical_volume_rejected():
    with pytest.raises(ValueError,match='rank deficient'):
        known_q_coefficients(jnp.array([-.5,.5,-.5]),jnp.zeros((3,2,4)))
    f=jnp.diag(jnp.array([-1.,1.,1.,1.]))
    w=jnp.stack((f[:2]+.315*f[2:],-(.315/.02)*f[:2],.02*f[2:]))
    with pytest.raises(ValueError,match='volume must be positive'):
        geometry_from_coefficients(w)


def test_two_pl1_states_reconstruct_the_upstream_map_when_radial_rows_are_independent():
    q=jnp.array([-.4,-2.6]);L=.02;d=.315
    f=jnp.diag(jnp.array([8.,5.,1/8,1/5]))@mechanical_shear(800.)
    h=signed_position_response(q,L,d,f)
    from neoarm_last_lens_tomography import rotation
    derotated=jnp.einsum('nij,njk->nik',rotation(q),h)
    coefficients=jnp.stack((jnp.cos(q)-(d/L)*q*jnp.sin(q),L*jnp.sinc(q/jnp.pi)+d*jnp.cos(q)),axis=-1)
    recovered=jnp.linalg.solve(coefficients,derotated.reshape(2,8)).reshape(4,4)
    np.testing.assert_allclose(recovered,f,rtol=1e-10,atol=1e-10)
