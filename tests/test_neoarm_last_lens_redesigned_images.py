from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_last_lens_redesigned_images import normalized_excitation,assert_separate_acquisitions,truth_response_likelihood


def test_heldout_controls_are_separate_and_excitation_conversion_is_correct():
    q=np.linspace(-.05,-np.pi+.05,25);t=np.asarray(normalized_excitation(q))
    np.testing.assert_allclose(t,np.linspace(-1,1,25),atol=1e-14)
    assert_separate_acquisitions(t,np.asarray(normalized_excitation([-.37,-2.10,-2.75])))
    with pytest.raises(ValueError,match='overlap'):assert_separate_acquisitions(t,t[:1])


def test_truth_likelihood_whitens_full_covariance_without_fitting_parameters():
    generator=np.random.default_rng(83);factor=generator.normal(size=(8,8));c=factor@factor.T+np.eye(8)
    z=generator.normal(size=(5,8));error=(z@np.linalg.cholesky(c).T).reshape(5,2,4)
    result=truth_response_likelihood(error,np.zeros_like(error),np.tile(c,(5,1,1)))
    assert result['degrees_of_freedom']==40
    assert result['chi_square']==pytest.approx(float(np.sum(z*z)))
    assert result['optical_parameters_fitted'] is False
