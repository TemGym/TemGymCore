import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.axial_field_column import FixedAxialFieldMesh
from temgym_core.axial_field_profile import axial_field_prefix,prefix_map_to_plane,sample_conditioned_basis
from temgym_core.immersed_objective import SharedObjectiveField,mechanical_to_canonical

jax.config.update('jax_enable_x64',True)


@pytest.fixture
def data():
    field = SharedObjectiveField.from_window(-.05,0.,.05,n_slices_per_side=128)
    boundaries = jnp.array([-.3,-.2,-.18,0.,.2,.22,.35])
    mesh = FixedAxialFieldMesh.from_boundaries(field,boundaries,(-1,0,-1,-1,1,-1),3,steps_per_segment=32)
    parameters = jnp.array([-2.7,-.0005,np.log(.002),.15])
    rates = jnp.array([35.,-40.])
    prefix = jax.jit(lambda p,b,r:axial_field_prefix(mesh,p,b,r))(parameters,boundaries,rates)
    return mesh,parameters,boundaries,rates,prefix


def test_prefix_edges_and_partial_planes_match_independent_full_tracing(data):
    mesh,p,b,r,prefix = data
    full = mesh.split_maps(p,b,r)
    np.testing.assert_allclose(prefix.canonical_maps[-1],full.full,rtol=3e-12,atol=1e-9)
    planes = jnp.array([-.3,-.19,-.01,0.,.07,.205,.35])
    cached = jax.jit(jax.vmap(lambda z:prefix_map_to_plane(mesh,p,prefix,z)[0]))(planes)
    direct = jax.jit(jax.vmap(lambda z:mesh.canonical_map_to_plane(p,b,r,z)))(planes)
    np.testing.assert_allclose(cached,direct,rtol=3e-12,atol=1e-9)
    assert np.isnan(np.asarray(prefix_map_to_plane(mesh,p,prefix,-.31)[0])).all()


def test_canonical_gaussian_two_ray_radius_matches_covariance(data):
    mesh,p,b,r,prefix = data
    width,angle = 60e-9,1.3e-5
    basis = jnp.zeros((4,2)).at[0,0].set(width).at[1,1].set(angle)
    sample_map,_ = prefix_map_to_plane(mesh,p,prefix,0.)
    sigma = jnp.diag(jnp.array([width,angle,width,angle])**2/4)
    for plane in (-.25,-.005,0.,.05,.31):
        radial,states = sample_conditioned_basis(mesh,p,prefix,0.,basis,plane)
        to_plane,_ = prefix_map_to_plane(mesh,p,prefix,plane)
        transfer = to_plane@jnp.linalg.inv(sample_map)
        covariance = transfer@sigma@transfer.T
        expected = 2*np.sqrt(float(covariance[0,0]))
        assert float(jnp.linalg.norm(radial[0])) == pytest.approx(expected,rel=2e-11)
        np.testing.assert_allclose(radial[1],0.,atol=2e-13)
    at_sample,_ = sample_conditioned_basis(mesh,p,prefix,0.,basis,0.)
    np.testing.assert_allclose(at_sample,[[width,0.],[0.,0.]],atol=1e-18)


def test_mechanical_height_input_keeps_orthogonal_magnetic_response(data):
    mesh,p,b,r,prefix = data
    k = mesh.split_maps(p,b,r).sample_k_m_inv
    h,alpha = 5e-9,1e-3
    basis = mechanical_to_canonical(k)@jnp.zeros((4,2)).at[0,0].set(h).at[1,1].set(alpha)
    values,_ = sample_conditioned_basis(mesh,p,prefix,0.,basis,.12)
    # In co-rotating coordinates, y_height=k B h while x_tilt=B alpha.
    assert float(values[1,0]) == pytest.approx(float(k*h/alpha*values[0,1]),rel=1e-11)
    assert abs(float(values[1,0]))>0
    np.testing.assert_allclose(values[1,1],0.,atol=1e-14)


def test_prefix_parameter_and_plane_derivatives_are_finite(data):
    mesh,p,b,r,_ = data
    def field_response(par):
        prefix = axial_field_prefix(mesh,par,b,r)
        return prefix_map_to_plane(mesh,par,prefix,.13)[0]
    assert np.all(np.isfinite(jax.jit(jax.jacfwd(field_response))(p)))
    prefix = axial_field_prefix(mesh,p,b,r)
    f = lambda z:prefix_map_to_plane(mesh,p,prefix,z)[0]
    assert np.all(np.isfinite(jax.jacfwd(jax.jacfwd(f))(.13)))


def test_zero_objective_larmor_angle_integrates_controlled_field_exactly(data):
    mesh,p,b,r,_ = data
    p = p.at[0].set(0.)
    prefix = axial_field_prefix(mesh,p,b,r)
    expected = -(35.*.02-40.*.02)
    assert float(prefix.larmor_radians[-1]) == pytest.approx(expected,abs=2e-15)
