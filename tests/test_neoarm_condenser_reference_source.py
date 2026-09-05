"""Coordinate-only tests: no synthetic truth or acquisition files required."""

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples/lens_inversion"))
from neoarm_condenser_reference_source import ReferenceSourceChart
from neoarm_condenser_covariance_fit import BASIS, INVISIBLE, SOURCE_SCALE
from neoarm_condenser_image_recovery import OMEGA


def make_chart():
    def matrix(optical, control):
        a, b, c = jnp.exp(optical[0]+control[0]), optical[1], optical[2]
        radial = jnp.array([[a, b], [c, (1+b*c)/a]])
        return jnp.kron(jnp.eye(2), radial)
    quotient = SimpleNamespace(
        model=SimpleNamespace(condenser_matrix=matrix),
        decode=lambda p: (p[:12],),
        admissible=lambda p: bool(np.isfinite(p).all() and p[12] > 0),
    )
    return ReferenceSourceChart(quotient, np.zeros(7), 1e-5)


def parameters():
    return jnp.asarray(np.r_[[.2, .01, -.5], np.zeros(9),
                            [2., 3., 4., 5., .1, .2, .3, .4, .5], .2, .01])


def test_round_transport_preserves_invisible_covariance_and_symplectic_form():
    chart = make_chart()
    matrix = np.asarray(chart.balanced_map(parameters()))
    np.testing.assert_allclose(matrix@INVISIBLE@matrix.T, INVISIBLE, atol=1e-12)
    np.testing.assert_allclose(matrix@OMEGA@matrix.T, OMEGA, atol=1e-12)


def test_reference_quotient_roundtrip_jit_and_derivative():
    chart = make_chart()
    p = parameters()
    roundtrip = jax.jit(lambda p: chart.decode(chart.encode(p)))
    np.testing.assert_allclose(roundtrip(p), p, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(jax.jacfwd(roundtrip)(p), np.eye(23), rtol=1e-9, atol=1e-9)


def test_same_transported_covariance_and_free_parameters():
    chart = make_chart()
    p = parameters()
    recovered = chart.decode(chart.encode(p))
    source = lambda p: jnp.einsum("k,kij->ij", p[12:21], jnp.asarray(BASIS))*jnp.outer(SOURCE_SCALE, SOURCE_SCALE)
    matrix = chart.quotient.model.condenser_matrix(p[:12], jnp.zeros(7))
    np.testing.assert_allclose(matrix@source(p)@matrix.T, matrix@source(recovered)@matrix.T, rtol=1e-10, atol=1e-25)
    assert chart.admissible(chart.encode(p))
    np.testing.assert_array_equal(recovered[:12], p[:12])
    np.testing.assert_array_equal(recovered[21:], p[21:])


def test_invalid_reference_contract_rejected():
    with pytest.raises(ValueError):
        ReferenceSourceChart(make_chart().quotient, np.zeros(6), 1e-5)
    with pytest.raises(ValueError):
        ReferenceSourceChart(make_chart().quotient, np.zeros(7), 0.)
