"""Pure-JAX tangent and adjoint maps on the packed (N, 12) array (the CPU implementation).

The maps are compared with ``jax.jvp`` / ``jax.vjp`` of an independent scan renderer at
*identical packed inputs* (so absolute-phase rounding cannot enter), and with each other
through a dot-product (adjoint) test under JAX's real pairing Re(sum ct * E).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from render_bundles import field_scan, random_packed, rel, scattered_coords, unit_grid

from temgym_core.render.coefficients import moments_to_packed_cotangent, tangent_coefficients
from temgym_core.render.kernels import reference as ref
from temgym_core.render.predicate import support_limit

jax.config.update("jax_enable_x64", True)

CASES = {
    "n200_grid64": dict(n=200, coords="grid", shape=(64, 64), width=0.25),
    "n40_grid13x11": dict(n=40, coords="grid", shape=(13, 11), width=0.4),
    "n17_scattered": dict(n=17, coords="scatter", shape=(150,), width=0.5),
    "n60_narrow": dict(n=60, coords="grid", shape=(32, 32), width=0.08),
}


def _problem(name, seed=0):
    case = CASES[name]
    rng = np.random.default_rng(seed)
    packed = jnp.asarray(random_packed(rng, case["n"], width=case["width"]))
    if case["coords"] == "grid":
        coords = jnp.asarray(unit_grid(*case["shape"]))
    else:
        coords = jnp.asarray(scattered_coords(rng, case["shape"][0]))
    return rng, packed, coords


def tangent_reference(packed, coords, tdot, **kwargs):
    """``T tdot = K(C(tdot))``, shape ``(nt, P)``."""
    return ref.polynomial_sum(packed, coords, tangent_coefficients(packed, tdot), **kwargs)


def adjoint_reference(packed, coords, ct, **kwargs):
    """``T^T ct = C^T(K^T(ct))`` from the closed-form formulas, shape ``(nt, N, 12)``."""
    return moments_to_packed_cotangent(packed, ref.polynomial_moments(packed, coords, ct, **kwargs))


def adjoint_via_transpose(packed, coords, ct):
    """Same map with ``C^T`` from ``jax.linear_transpose``."""
    moments = ref.polynomial_moments(packed, coords, ct)
    zeros = jnp.zeros((ct.shape[0],) + packed.shape, jnp.float64)
    transpose = jax.linear_transpose(lambda t: tangent_coefficients(packed, t), zeros)
    return transpose(moments)[0]


def _jvp_scan(packed, coords, tdot, limit=None):
    f = lambda pk: field_scan(pk, coords, limit)
    return jnp.stack([jax.jvp(f, (packed,), (t,))[1] for t in tdot])


def _vjp_scan(packed, coords, ct, limit=None):
    _, pullback = jax.vjp(lambda pk: field_scan(pk, coords, limit), packed)
    return jnp.stack([pullback(c)[0] for c in ct])


@pytest.mark.parametrize("name", CASES)
def test_forward_matches_scan(name):
    _, packed, coords = _problem(name)
    assert rel(ref.field(packed, coords), field_scan(packed, coords)) <= 1e-12


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("nt", [1, 7])
def test_tangent_matches_jvp_of_scan(name, nt):
    rng, packed, coords = _problem(name)
    tdot = jnp.asarray(rng.normal(size=(nt,) + packed.shape))
    assert rel(tangent_reference(packed, coords, tdot), _jvp_scan(packed, coords, tdot)) <= 1e-10


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("nt", [1, 3])
def test_adjoint_matches_vjp_of_scan(name, nt):
    rng, packed, coords = _problem(name)
    n_pix = coords.shape[0]
    ct = jnp.asarray(rng.normal(size=(nt, n_pix)) + 1j * rng.normal(size=(nt, n_pix)))
    assert rel(adjoint_reference(packed, coords, ct), _vjp_scan(packed, coords, ct)) <= 1e-10


@pytest.mark.parametrize("name", ["n200_grid64", "n17_scattered"])
def test_hand_derived_formulas_equal_autodiff_transpose(name):
    """The closed-form moment-to-gradient formulas agree with ``jax.linear_transpose`` of C."""
    rng, packed, coords = _problem(name)
    ct = jnp.asarray(rng.normal(size=(2, coords.shape[0]))
                     + 1j * rng.normal(size=(2, coords.shape[0])))
    assert rel(adjoint_reference(packed, coords, ct), adjoint_via_transpose(packed, coords, ct)) <= 1e-12


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("limit", [None, support_limit(1e-4)])
def test_dot_product_adjoint(name, limit):
    """|<T tdot, ct> - <tdot, T^T ct>| / (|T tdot| |ct|) <= 1e-13 (real pairing)."""
    rng, packed, coords = _problem(name)
    n_pix = coords.shape[0]
    tdot = jnp.asarray(rng.normal(size=(1,) + packed.shape))
    ct = jnp.asarray(rng.normal(size=(1, n_pix)) + 1j * rng.normal(size=(1, n_pix)))
    tangent = tangent_reference(packed, coords, tdot, limit=limit)
    adjoint = adjoint_reference(packed, coords, ct, limit=limit)
    lhs = jnp.real(jnp.sum(ct * tangent))
    rhs = jnp.sum(tdot * adjoint)
    scale = jnp.linalg.norm(tangent) * jnp.linalg.norm(ct)
    assert float(jnp.abs(lhs - rhs) / scale) <= 1e-13


def test_dot_product_of_the_kernel_pair_alone():
    """K and K^T (the two kernels) are exact transposes: same test in moment space."""
    rng, packed, coords = _problem("n40_grid13x11")
    n_pix, n = coords.shape[0], packed.shape[0]
    w = jnp.asarray(rng.normal(size=(2, n, 12)))
    ct = jnp.asarray(rng.normal(size=(2, n_pix)) + 1j * rng.normal(size=(2, n_pix)))
    lhs = jnp.real(jnp.sum(ct * ref.polynomial_sum(packed, coords, w), axis=-1))
    rhs = jnp.sum(w * ref.polynomial_moments(packed, coords, ct), axis=(1, 2))
    scale = jnp.linalg.norm(ref.polynomial_sum(packed, coords, w)) * jnp.linalg.norm(ct)
    assert float(jnp.max(jnp.abs(lhs - rhs)) / scale) <= 1e-13


@pytest.mark.parametrize("limit", [None, support_limit(1e-3)])
def test_truncation_error_bound_and_frozen_mask_derivatives(limit):
    """The truncated sum is within cutoff * sum|A| of the exact one; its derivatives are
    those of the truncated scan (mask frozen)."""
    rng, packed, coords = _problem("n40_grid13x11")
    exact = ref.field(packed, coords)
    truncated = ref.field(packed, coords, limit=limit)
    if limit is not None:
        amplitude = jnp.hypot(packed[:, 10], packed[:, 11]).sum()
        assert float(jnp.max(jnp.abs(truncated - exact))) <= np.exp(-limit) * float(amplitude)
    tdot = jnp.asarray(rng.normal(size=(2,) + packed.shape))
    assert rel(tangent_reference(packed, coords, tdot, limit=limit),
               _jvp_scan(packed, coords, tdot, limit)) <= 1e-10


def test_chunking_is_exact():
    rng, packed, coords = _problem("n40_grid13x11")
    tdot = jnp.asarray(rng.normal(size=(2,) + packed.shape))
    ct = jnp.asarray(rng.normal(size=(2, coords.shape[0])) + 0j)
    a = tangent_reference(packed, coords, tdot, chunk=16)
    b = tangent_reference(packed, coords, tdot, chunk=4096)
    assert rel(a, b) <= 1e-14
    assert rel(adjoint_reference(packed, coords, ct, chunk=16),
               adjoint_reference(packed, coords, ct, chunk=4096)) <= 1e-14


def test_tangent_of_each_column_is_the_partial_derivative():
    """Unit tangents e_j on one beam reproduce dE/dcolumn_j from the scan reference."""
    _, packed, coords = _problem("n17_scattered")
    tdot = np.zeros((12,) + packed.shape)
    for j in range(12):
        tdot[j, 5, j] = 1.0
    tdot = jnp.asarray(tdot)
    assert rel(tangent_reference(packed, coords, tdot), _jvp_scan(packed, coords, tdot)) <= 1e-10


def test_empty_inputs():
    _, packed, coords = _problem("n17_scattered")
    assert ref.field(packed[:0], coords).shape == (coords.shape[0],)
    assert ref.field(packed, coords[:0]).shape == (0,)
    assert ref.polynomial_sum(packed, coords, jnp.zeros((0,) + packed.shape)).shape == (0, 150)
    assert ref.polynomial_moments(packed[:0], coords, jnp.zeros((2, 150), complex)).shape == (2, 0, 12)
