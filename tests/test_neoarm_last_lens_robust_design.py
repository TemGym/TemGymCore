from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_last_lens_robust_design import resized_pure_covariance,fixed_selection_gate
from neoarm_mixed_gaussian_validation import MixedGaussianSource,physical_emittances
jax.config.update('jax_enable_x64',True)


def test_resizing_changes_position_and_angle_reciprocally_preserving_quantum_emittance():
    source=MixedGaussianSource(np.diag([2e-9,1.5e-9])**2,np.zeros((2,2)),np.zeros((4,4)),np.zeros(4))
    for factor in (.9,1.,1.1):
        c=np.asarray(resized_pure_covariance(source.total_covariance,factor))
        np.testing.assert_allclose(np.diag(c)[[0,2]],np.diag(source.total_covariance)[[0,2]]*factor**2)
        np.testing.assert_allclose(np.diag(c)[[1,3]],np.diag(source.total_covariance)[[1,3]]/factor**2)
        np.testing.assert_allclose(physical_emittances(c,source.wavelength_m),source.wavelength_m/(4*np.pi),rtol=1e-12)


def test_fixed_selection_cannot_drop_a_failed_perturbed_state():
    rms=jnp.array([[.4,.5],[.05,.07],[.27,.4]])
    capture=jnp.ones((3,9));selected=jnp.array([True,False,True])
    assert fixed_selection_gate(rms,capture,selected)
    assert not fixed_selection_gate(rms.at[2,0].set(.249),capture,selected)
    assert not fixed_selection_gate(rms,capture.at[0,3].set(.98),selected)
