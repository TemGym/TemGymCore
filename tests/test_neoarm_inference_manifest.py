from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_inference_manifest import inference_arrays


def test_inference_allowlist_excludes_truth_and_unknown_future_fields():
    arrays={key:np.ones(2) for key in ('response','response_covariance','pl1_controls','noisy_counts',
        'expected_counts','candidate_q','il3_q_offsets','source_covariance','future_truth_field')}
    assert set(inference_arrays(arrays))=={'response','response_covariance','pl1_controls','noisy_counts'}
