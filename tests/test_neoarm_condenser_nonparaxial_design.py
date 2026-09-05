import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_nonparaxial_design import symplectic_source_squeeze,symplectic_emittances


@pytest.mark.parametrize('s',[.5,2.,8.,32.])
def test_source_squeeze_preserves_both_quantum_emittances(s):
    c=np.diag(np.array([5e-9,6e-9,100e-6,90e-6])**2)
    actual=symplectic_source_squeeze(c,s)
    np.testing.assert_allclose(symplectic_emittances(actual),symplectic_emittances(c),rtol=1e-12)
    np.testing.assert_allclose(np.linalg.det(actual),np.linalg.det(c),rtol=1e-12)
    np.testing.assert_allclose(np.sqrt(np.diag(actual))/np.sqrt(np.diag(c)),[s,s,1/s,1/s])


def test_source_squeeze_rejects_unphysical_scaling():
    for s in [0.,-1.,np.nan]:
        with pytest.raises(ValueError):symplectic_source_squeeze(np.eye(4),s)
