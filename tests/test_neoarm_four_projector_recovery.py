from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_four_projector_recovery import encode_physical,physical_coordinates,selected_data


def test_four_positive_spans_roundtrip():
    p=np.r_[np.log(.02),np.log(.315),-1.57,-1.52,np.log(.018),np.log(.033),.96,.4,
        np.log(.018),np.log(.0445),2.7,.45,np.log(.02),np.log(.058),2.3,.5,2.,.3,.02,.7,.005]
    encoded=encode_physical(p);np.testing.assert_allclose(physical_coordinates(encoded),p,rtol=2e-14,atol=2e-14)
    assert np.exp(encoded[0])==pytest.approx(.5265)


def test_training_excludes_fresh_poisoned_heldout():
    response=np.ones((4,2,2,4));response[2]=np.nan
    arrays=dict(response=response,joint_response_covariance=np.eye(64),downstream_controls=np.array([[.8,.75,-.75],[.8,.75,.75]]),
        il1_controls=np.array([-.75,0.,.375,.75]),training_mask=np.array([True,True,False,True]),heldout_mask=np.array([False,False,True,False]))
    controls,data,cov=selected_data(arrays,True)
    np.testing.assert_array_equal(controls[:,3],[-.75,-.75,0.,0.,.75,.75]);assert data.shape==(6,2,4) and cov.shape==(48,48)
    with pytest.raises(ValueError):selected_data(arrays,False)
