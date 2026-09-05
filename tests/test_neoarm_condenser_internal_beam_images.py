from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_internal_beam_images import roles


def test_internal_Q_acquisition_roles_are_explicit_boolean_and_disjoint():
    train, held = roles([True, True, False], [False, False, True], 3)
    assert train.sum() == 2 and held.sum() == 1
    with pytest.raises(ValueError):
        roles([1, 1, 0], [0, 0, 1], 3)
    with pytest.raises(ValueError):
        roles([True, True], [True, False], 2)
