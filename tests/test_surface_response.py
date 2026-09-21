"""Amplitude/flux conventions and angular response of the Bragg component."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase import units
from temgym_core.components import BraggSample, Detector
from temgym_core.scattering import PotentialStepResponse, copper_scattering_factor
from temgym_core.source import gaussian_waist
from temgym_core.evaluate import gaussian_image


def test_potential_step_matches_boundary_conditions_and_passivity():
    response = PotentialStepResponse(inner_potential_v=21.2)
    theta = np.deg2rad(np.array([.5, 1., 2., 4., 6.]))
    energy = 30_000.
    rest = units._me*units._c**2/units._e
    p0 = np.sqrt(energy*(1+energy/(2*rest)))
    p1_squared = (energy+21.2)*(1+(energy+21.2)/(2*rest))
    normal0 = p0*np.sin(theta)
    normal1 = np.sqrt(p1_squared-(p0*np.cos(theta))**2)
    expected = (normal0-normal1)/(normal0+normal1)
    np.testing.assert_allclose(response.amplitude(theta,energy),expected,rtol=1e-10)
    r = np.asarray(response.reflectivity(theta,energy))
    assert np.all(np.diff(r)<0) and np.all((r>0)&(r<1))
    transmission = 1+expected
    np.testing.assert_allclose(r+normal1/normal0*abs(transmission)**2,1.,atol=1e-13)
    assert response.reflectivity(0.,energy)==1.
    vacuum = PotentialStepResponse(inner_potential_v=0.)
    np.testing.assert_array_equal(vacuum.amplitude(jnp.array([0.,.1]),energy),0.)
    absorbing = PotentialStepResponse(inner_potential_v=21.2,absorption_ev=2.)
    values = absorbing.reflectivity(theta,energy)
    assert np.all((values>=0)&(values<=1))
    assert absorbing.reflectivity(-.1,energy)==0.


def test_complex_action_jit_and_angle_derivative():
    response = PotentialStepResponse(inner_potential_v=21.2,absorption_ev=1.)
    theta,energy,k=.03,30_000.,9e11
    action=response.complex_action(theta,energy,k)
    np.testing.assert_allclose(jnp.exp(1j*k*action),response.amplitude(theta,energy),rtol=1e-13)
    assert action.imag>0
    fn=jax.jit(lambda r,t:r.reflectivity(t,energy))
    derivative=jax.grad(lambda t:fn(response,t))(theta)
    finite=(fn(response,theta+1e-6)-fn(response,theta-1e-6))/2e-6
    np.testing.assert_allclose(derivative,finite,rtol=2e-8)


def test_bragg_response_preserves_absolute_flux_and_exact_zero_orders():
    response=PotentialStepResponse(inner_potential_v=21.2)
    theta=np.deg2rad(2.)
    sample=BraggSample(q_xy=jnp.zeros((2,2)),coefficients=jnp.array([.8+0j,0j]),
                      angular_response=response,surface_normal_xyz=(np.cos(theta),0.,-np.sin(theta)))
    beam=gaussian_waist(z=0.,voltage=30e3,rms_size=3e-9)
    output=jax.jit(lambda s,b:s(b))(sample,beam)
    expected=.8*response.amplitude(theta,beam.voltage)
    np.testing.assert_allclose(output.amplitude[0]/beam.amplitude,expected,rtol=1e-12)
    assert output.amplitude[1]==0
    detector=Detector(z=0.,shape=(128,128),pixel_size=(.4e-9,.4e-9))
    flux=float(gaussian_image(output,detector,method='scan',samples_per_pixel=2).sum())
    np.testing.assert_allclose(flux,.8**2*response.reflectivity(theta,beam.voltage),rtol=1e-9)
    actions=sample.order_action(theta,beam.voltage,beam.k)
    np.testing.assert_allclose(jnp.exp(1j*beam.k*actions[0]),expected,rtol=1e-12)
    coefficients=sample.coefficients_at(jnp.array([theta,2*theta]),30e3)
    assert coefficients.shape==(2,2)
    truncated=replace(sample,q_xy=sample.q_xy[:1],coefficients=sample.coefficients[:1])
    np.testing.assert_array_equal(truncated.coefficients_at(theta,30e3),coefficients[0,:1])


def test_copper_factor_frequency_convention():
    # Independent forward-factor value from 2*sum(a_i) and large-q asymptote.
    np.testing.assert_allclose(copper_scattering_factor(0.),5.601000000000,rtol=1e-4)
    f=np.asarray(copper_scattering_factor(jnp.array([0.,1.,2.,4.,8.])))
    assert np.all(f>0) and np.all(np.diff(f)<0)
    np.testing.assert_allclose(copper_scattering_factor(-1.),copper_scattering_factor(1.))


@pytest.mark.parametrize('kwargs',[{'inner_potential_v':-1.},{'inner_potential_v':np.nan},
                                  {'inner_potential_v':21.2,'absorption_ev':-1.}])
def test_invalid_potential_is_rejected(kwargs):
    with pytest.raises(ValueError,match='nonnegative'):
        PotentialStepResponse(**kwargs)


def test_bragg_response_uses_each_incident_direction_and_field_evaluation():
    response=PotentialStepResponse(inner_potential_v=21.2)
    angle=np.deg2rad(3.)
    sample=BraggSample(q_xy=jnp.array([[0.,0.],[2e9,0.]]),coefficients=jnp.array([.6,.2j]),
        angular_response=response,surface_normal_xyz=(np.cos(angle),0.,-np.sin(angle)))
    beam=gaussian_waist(z=0.,voltage=30e3,rms_size=3e-9).to_vector()
    bundle=jax.tree.map(lambda x:jnp.repeat(x,2,axis=0),beam).derive(dx=jnp.array([0.,.01]))
    output=sample(bundle)
    angles=angle-np.arctan([0.,.01])
    expected=sample.coefficients_at(angles,30e3)
    np.testing.assert_allclose(output.amplitude.reshape(2,2)/bundle.amplitude[:,None],expected,rtol=1e-12)
    with pytest.raises(ValueError,match='grazing_angle_rad'):
        sample.evaluate_complex(jnp.zeros((1,2)))
    np.testing.assert_allclose(sample.evaluate_complex(jnp.zeros((2,2)),grazing_angle_rad=angles,voltage=30e3),
                               expected.sum(axis=-1))
