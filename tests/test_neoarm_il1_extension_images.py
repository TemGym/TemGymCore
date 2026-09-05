from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_il1_extension_images import public_arrays


def test_il1_public_export_omits_generating_fields():
    a=dict(downstream_controls=np.zeros((2,3)),il1_controls=np.arange(4.),response=np.ones((4,2,2,4)),
        expected_counts=np.ones(3),variance_counts=np.ones(3),source_covariance=np.eye(4),true_q1=np.arange(4.))
    assert set(public_arrays(a))=={'downstream_controls','il1_controls','response'}
