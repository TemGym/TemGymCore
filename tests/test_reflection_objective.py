"""Integration checks for the notebook's aberrated objective and surface adapter."""
import json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope='module')
def model():
    notebook=Path(__file__).resolve().parents[1]/'examples/sem_reflection/sem_reflection.ipynb'
    cells=json.loads(notebook.read_text())['cells']
    namespace={}
    for index in (1,3,6):
        exec(compile(''.join(cells[index]['source']),f'{notebook}:cell{index}','exec'),namespace)
    return namespace


def test_displaced_specular_packet_matches_mirror_symmetry(model):
    m=model
    previous=m['out_frame']
    reflection=jnp.diag(jnp.array([-1.,1.,1.]))
    m['out_frame']=replace(previous,u_xyz=-reflection@m['column_frame'].u_xyz,
                          v_xyz=reflection@m['column_frame'].v_xyz)
    try:
        beam=m['probe'].derive(x=3e-7,y=-2e-7).to_vector()
        result,_=m['reflect_packets'](beam,np.array([[0,0]]),np.array([1.+0j]),apply_response=False)
        flip=jnp.diag(jnp.array([-1.,1.]))
        np.testing.assert_allclose(result.r_xy,beam.r_xy@flip,rtol=1e-9,atol=1e-15)
        np.testing.assert_allclose(result.d_xy,beam.d_xy@flip,rtol=1e-12,atol=1e-15)
        np.testing.assert_allclose(result.Q_inv,flip@beam.Q_inv@flip,rtol=1e-8,atol=1e-7)
        np.testing.assert_allclose(result.amplitude,beam.amplitude,rtol=1e-8)
        np.testing.assert_allclose(result.pathlength,beam.pathlength,rtol=0,atol=1e-17)
    finally:
        m['out_frame']=previous


def test_lateral_translation_keeps_the_lattice_phase(model):
    m=model
    beam=m['probe'].derive(x=1e-8,y=2e-9).to_vector()
    displacement=.37*m['lateral_period']
    orders=np.array([[0,1]])
    a,_=m['reflect_packets'](beam,orders,np.array([1.+0j]),apply_response=False)
    b,_=m['reflect_packets'](beam.derive(y=beam.y+displacement),orders,np.array([1.+0j]),apply_response=False)
    np.testing.assert_allclose(b.r_xy-a.r_xy,[[0.,displacement]],rtol=1e-8,atol=1e-17)
    expected=2*np.pi*displacement/m['lateral_period']
    actual=np.asarray((b.pathlength-a.pathlength)*beam.k)
    np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=2e-6)
    np.testing.assert_allclose(b.Q_inv,a.Q_inv,rtol=1e-12)


def test_aberrated_packet_field_matches_exact_objective_phase(model):
    m=model
    coeff=m['KrivanekCoeffs'](C30=1e-3,C21=1e-5,C12=5e-7,phi21=.4,phi12=.7)
    kicks,_=m['solve_pivot'](4.,coeffs=coeff)
    column=m['column'](kicks,coeff)
    np.testing.assert_allclose(m['hit_and_angles'](kicks,coeff),[0,0,np.deg2rad(4),0],atol=1e-11)
    pupil=m['run_to_end'](m['source'],(*column[:-2],m['Plane'](z=m['P']['objective_z'])),propagator=m['fs'])
    width=np.sqrt(2/(float(pupil.k)*np.linalg.eigvalsh(np.asarray(pupil.Q_inv.imag)).min()))
    axis=np.linspace(-2,2,21)*width
    xx,yy=np.meshgrid(axis+float(pupil.x),axis+float(pupil.y))
    grid=SimpleNamespace(shape=xx.shape,coords=jnp.asarray(np.column_stack((xx.ravel(),yy.ravel()))))
    before=np.asarray(m['evaluate_gaussians'](pupil,grid,method='scan'))
    action=jax.vmap(column[-2].phase_shift)(grid.coords)
    exact=before*np.exp(1j*float(pupil.k)*np.asarray(action).reshape(xx.shape))
    errors=[]
    for n in (64,128):
        bundle=m['decompose_gaussian_angular'](m['source'],num_across=n,through=m['column'](kicks),
            broadening=m['P']['broadening'],extent_sigma=m['P']['extent_sigma'])
        output=m['run_to_end'](bundle,column[:-1],propagator=m['fs'])
        field=np.asarray(m['evaluate_gaussians'](output,grid,method='scan'))
        errors.append(np.linalg.norm(field-exact)/np.linalg.norm(exact))
    assert errors[1]<errors[0]
    assert errors[1]<2e-4
