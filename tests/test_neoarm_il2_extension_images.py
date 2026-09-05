from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_il2_extension_images import public_arrays


def test_inference_export_allowlists_new_controls_without_generating_fields():
    arrays=dict(downstream_controls=np.ones((2,2)),il2_controls=np.arange(4.),response=np.zeros((4,2,2,4)),
        expected_counts=np.ones((4,2,9,2,2)),variance_counts=np.ones((4,2,9,2,2)),
        source_covariance=np.eye(4),true_IL2_q=np.arange(4.))
    public=public_arrays(arrays)
    assert set(public)=={'downstream_controls','il2_controls','response'}
