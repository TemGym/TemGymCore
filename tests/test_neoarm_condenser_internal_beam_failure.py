from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_internal_beam_failure import guard_diagnostic, balanced_diagnostic


def test_guard_diagnostic_preserves_acceptance_and_never_repairs_matrix():
    symplectic = np.array([[1., 2., 0., 0.], [0., 1., 0., 0.],
                          [0., 0., 1., 2.], [0., 0., 0., 1.]])
    assert guard_diagnostic(symplectic)['accepted']
    broken = symplectic.copy(); broken[0, 0] += 1e-5
    original = broken.copy()
    record = guard_diagnostic(broken)
    assert not record['accepted'] and record['maximum_guard_ratio'] > 1.
    np.testing.assert_array_equal(broken, original)
    with pytest.raises(ValueError):
        guard_diagnostic(np.full((4, 4), np.nan))


def test_balanced_guard_uses_symplectic_units_not_independent_coordinate_whitening():
    matrix = np.diag([1e5, 1e-5, 2e5, .5e-5])
    source = np.diag([25e-18, 1e-8, 36e-18, 2e-8])
    result = balanced_diagnostic(matrix, source, left=matrix, right=np.eye(4))
    assert result['maximum_absolute_symplectic_residual'] < 1e-12
    for key in ('source_scale', 'detector_scale'):
        scale = np.array(result[key])
        np.testing.assert_allclose(scale[[0, 2]]*scale[[1, 3]],
                                   result['common_conjugate_scale_m'], rtol=1e-15)
