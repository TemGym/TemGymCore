import sys
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_cold_joint_endpoint import select_cold_seed, joint_reference_coordinates
from neoarm_condenser_endpoint_recovery import ORDER


def benchmark():
    rows = []
    for sign in (-1, 0, 1):
        rows.append(dict(sign=sign, label=f'start{sign}', initial_optical_parameters=[sign]*12,
            source_initialization=dict(training_only=True), moment_chi_square=10-sign,
            final_exact_fit=dict(accepted=True, optimizer_success=True,
                original_parameters=np.full(23, sign).tolist(), training_chi_square=3+sign,
                numerical_full_rank=True, all23_parameters_free=True,
                original_normalized_gradient=1e-8, original_information_whitened_gradient=1e-8)))
    return dict(rows=rows)


def test_selection_uses_original_likelihood_not_guide_or_truth():
    report = benchmark()
    p, selection = select_cold_seed(report)
    np.testing.assert_array_equal(p, -np.ones(23))
    assert selection['sign'] == -1
    assert selection['prescribed_start_count'] == 3
    assert selection['accepted_count'] == 3


def test_failed_lower_cost_cannot_seed_and_no_accepted_is_error():
    report = benchmark()
    report['rows'][0]['final_exact_fit']['accepted'] = False
    p, selection = select_cold_seed(report)
    assert selection['sign'] == 0
    np.testing.assert_array_equal(p, np.zeros(23))
    for row in report['rows']:
        row['final_exact_fit'] = None
    with pytest.raises(ValueError, match='No accepted'):
        select_cold_seed(report)


@pytest.mark.parametrize('field,value', [('original_normalized_gradient', .01),
    ('original_information_whitened_gradient', None), ('numerical_full_rank', False),
    ('all23_parameters_free', False), ('optimizer_success', False), ('training_chi_square', np.nan)])
def test_inconsistent_acceptance_is_rejected(field, value):
    report = benchmark()
    report['rows'][0]['final_exact_fit'][field] = value
    with pytest.raises(ValueError, match='stationarity'):
        select_cold_seed(report)


def test_only_prescribed_three_starts_are_admitted():
    report = benchmark()
    report['rows'].append(deepcopy(report['rows'][0]))
    with pytest.raises(ValueError, match='three prescribed'):
        select_cold_seed(report)


def test_joint_chart_roundtrip_derivatives_and_endpoint_passthrough():
    class InvertibleChart:
        def encode(self, p): return p.at[12:21].set(2*p[12:21]+p[0])
        def decode(self, p): return p.at[12:21].set((p[12:21]-p[0])/2)
    encode, decode = joint_reference_coordinates(InvertibleChart())
    p = jnp.linspace(-.2, .3, 29)
    q = jax.jit(encode)(p)
    np.testing.assert_allclose(jax.jit(decode)(q), p, atol=1e-15)
    np.testing.assert_allclose(q[-6:], p[-6:])
    np.testing.assert_allclose(jax.jacfwd(lambda x: decode(encode(x)))(p), np.eye(29), atol=1e-15)
    assert len(ORDER[:6]) == 6 and len(ORDER[6:]) == 23
