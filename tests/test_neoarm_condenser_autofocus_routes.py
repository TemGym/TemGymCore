from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_autofocus_routes import independent_roles


def test_autofocus_roles_are_disjoint_and_do_not_change_training_selection():
    controls = np.arange(30).reshape(10, 3)/30
    selected, train = independent_roles(controls, [0, 3, 8], 2)
    np.testing.assert_array_equal(selected[:3], [0, 3, 8])
    np.testing.assert_array_equal(train, [True, True, True, False, False])
    assert len(np.unique(selected)) == 5
    with pytest.raises(ValueError):
        independent_roles(controls[:3], [0, 1, 2], 1)
