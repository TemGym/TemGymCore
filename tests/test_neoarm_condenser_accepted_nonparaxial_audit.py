import sys
from pathlib import Path
import jax.numpy as jnp
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_accepted_nonparaxial_audit import AcceptedCondenserAdapter,aperture_points
from neoarm_condenser_nonparaxial_benchmark import gaussian_phase_space_rule
from neoarm_condenser_stigmator_contract import drift,GROUPED
from temgym_core.immersed_objective import uniform_field_canonical_matrix


def test_aperture_forward_stops_after_CL3_and_closes_paraxial_map():
    class Geometry:
        source_z_m=-.65
        objective_entrance_m=-.05
        aperture_z_m=-.4
    optical=np.array([[-.5,np.log(.02),.2,.55],[-.43,np.log(.015),.15,.65],[-.09,np.log(.008),.1,.45]])
    source=np.diag(np.array([5e-9,6e-9,50e-6,45e-6])**2)
    adapter=AcceptedCondenserAdapter(Geometry(),optical,source,jnp.tile(jnp.eye(4),(2,1,1)))
    points,_=gaussian_phase_space_rule(source,3);control=jnp.array([.2,.3,1.5])
    cursor=adapter.source_z_m;m=jnp.eye(4)
    for i in range(2):
        z,loglength,bias,slope=optical[i];length=np.exp(loglength)
        m=drift(z-length/2-cursor)@m
        native=uniform_field_canonical_matrix((bias+slope*control[i])/length,length)
        m=native[jnp.ix_(GROUPED,GROUPED)]@m;cursor=z+length/2
    m=drift(adapter.aperture_z_m-cursor)@m
    np.testing.assert_allclose(aperture_points(adapter,control,points),points@m[:2].T,atol=1e-18,rtol=1e-12)
    np.testing.assert_array_equal(aperture_points(adapter,control,points),aperture_points(adapter,control.at[2].set(3.),points))
