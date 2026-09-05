from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_pl1_il3_images import assemble_response_covariance,training_heldout_masks


def test_joint_covariance_retains_full_blocks_without_reusing_calibration_noise():
    c=np.tile(np.eye(8),(5,2,1,1));c[:,:,0,4]=.2;c[:,:,4,0]=.2
    paired,joint=assemble_response_covariance(c)
    assert paired.shape==(5,16,16) and joint.shape==(80,80)
    for i in range(5):
        np.testing.assert_array_equal(paired[i,:8,:8],c[i,0])
        np.testing.assert_array_equal(paired[i,8:,8:],c[i,1])
        np.testing.assert_array_equal(joint[16*i:16*(i+1),16*i:16*(i+1)],paired[i])
    assert np.count_nonzero(joint[:16,16:])==0
    assert np.linalg.eigvalsh(joint).min()>0


def test_endpoints_are_frozen_holdouts():
    train,held=training_heldout_masks([-.1,-.05,0,.05,.1])
    np.testing.assert_array_equal(train,[False,True,True,True,False]);np.testing.assert_array_equal(held,~train)
    with pytest.raises(ValueError,match='precisely five'):training_heldout_masks([-.05,0,.05])
