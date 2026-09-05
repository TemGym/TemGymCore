from pathlib import Path
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_actual_analyser_contract import compose_analyser_maps, gaussian_rectangle_capture

jax.config.update('jax_enable_x64', True)


def test_canonical_mechanical_composition_and_grouping_have_no_extra_shear():
    pre = np.array([[1., .2, .1, .01], [.5, 1.1, .03, -.2],
                    [-.1, -.01, .9, .3], [-.03, .2, .6, 1.3]])
    downstream = np.stack((np.eye(4), np.diag([2., .5, 3., 1/3])))
    model = SimpleNamespace(prefield=pre, downstream=downstream, objective_entrance_m=-.04)
    d = np.eye(4); d[0, 1] = d[2, 3] = .01
    group = np.array([0, 2, 1, 3])
    expected = (downstream@pre@d)[:, group[:, None], group[None, :]]
    np.testing.assert_allclose(compose_analyser_maps(model, -.05), expected, atol=1e-15)


def test_gaussian_rectangle_capture_against_separable_integral_and_refinement():
    covariance = jnp.array([[[400., 0.], [0., 1600.]], [[1e6, 0.], [0., 4e6]]])
    expected = jnp.prod(jax.scipy.special.erf(128/jnp.sqrt(2*jnp.diagonal(covariance, axis1=-2, axis2=-1))), axis=-1)
    result = jax.jit(lambda c: gaussian_rectangle_capture(c, order=64))(covariance)
    np.testing.assert_allclose(result, expected, atol=2e-13, rtol=0)
    refined = gaussian_rectangle_capture(covariance[1:], order=16)
    np.testing.assert_allclose(refined, result[1:], atol=2e-14, rtol=0)
