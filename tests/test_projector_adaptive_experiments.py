"""Check prospective design, control conventions and accumulated measurements."""
import sys
from pathlib import Path

import jax
import numpy as np

jax.config.update('jax_enable_x64',True)
sys.path.insert(0,str(Path(__file__).parents[1]/'examples/microscope_models'))
from finite_projector_pipeline import initial_guess,public_specification,prepare_column_fit,predict_observables
from projector_adaptive_experiments import CalibrationMeasurements,five_controls,select_informative_settings


def test_accumulation_keeps_original_observations_once_and_round_trips(tmp_path):
    controls=five_controls(np.array([[.9,.8,.7,.9]]))
    pairs=dict(lens_controls=controls,angle_changes_rad=np.array([[.006,0.]]),
               displacements_pixels=np.array([[1.,2.]]),covariance_pixels=np.eye(2)[None]*.01,
               pair_labels=['independent pair'])
    shifts=dict(pairs,position_changes_m=np.array([[20e-9,0.]]),angle_changes_rad=np.zeros((1,2)))
    first=CalibrationMeasurements(controls,np.ones((1,3)),np.eye(3)[None],pairs,shifts)
    second=CalibrationMeasurements(controls*1.01,np.ones((1,3))*2,np.eye(3)[None]*2,pairs,shifts)
    combined=first.append(second)
    combined.save(tmp_path/'measurements.npz')
    restored=CalibrationMeasurements.load(tmp_path/'measurements.npz')
    np.testing.assert_array_equal(restored.values,[[1.,1.,1.],[2.,2.,2.]])
    np.testing.assert_array_equal(first.controls[:,0],[1.])
    assert restored.tilts['pair_labels']==['independent pair']*2
    assert restored.shifts['position_changes_m'].shape==(2,2)


def test_information_design_respects_allowed_settings_without_generating_truth():
    p=initial_guess();spec=public_specification()
    controls=five_controls(np.array([[.9,.8,.7,.9],[1.,.8,.7,.9]]))
    problem=prepare_column_fit(lens_controls=controls,
       measured_values=np.array([predict_observables(p,c,spec) for c in controls]),
       measurement_covariances=np.tile(np.eye(3)*1e-4,(2,1,1)),specimen_and_detector=spec,prior_parameters=p)
    candidates=dict(lens_controls=np.array([[1.,.9,.8,.7,.9],[1.001,.9,.8,.7,.9],[1.,1.,.7,.8,.9]]),
                    labels=np.array(['projector1','objective','projector2']))
    selected=select_informative_settings(candidates=candidates,current_parameters=p,current_problem=problem,
          specimen_and_detector=spec,number_of_settings=2,allowed=np.array([True,False,True]))
    assert set(selected['indices'])=={0,2}
    assert np.all(np.asarray(selected['information_gain'])>0)
    assert selected['information_gain'][1] <= selected['information_gain'][0]
    np.testing.assert_array_equal(candidates['lens_controls'][:,0],[1.,1.001,1.])


def test_round_lens_strengths_can_be_estimated_from_shift_rotation_without_geometry():
    from finite_projector_pipeline import predict_beam_shift_displacements
    from projector_notebook_helpers import parameters_to_model
    from projector_adaptive_experiments import estimate_strengths_from_beam_rotation
    p=initial_guess();p[11]=-.45;p[12:16]=np.log([.56,.67,.44,.74])
    rng=np.random.default_rng(57)
    controls=np.c_[1+rng.uniform(-.002,.002,40),rng.uniform(.5,1.4,(40,4))]
    positions=np.tile([20e-9,0.],(40,1));angles=np.zeros((40,2))
    shifts=np.asarray(predict_beam_shift_displacements(p,controls,positions,angles,110e-6))
    measured=dict(lens_controls=controls,position_changes_m=positions,angle_changes_rad=angles,
                  displacements_pixels=shifts,covariance_pixels=np.tile(np.eye(2)*1e-6,(40,1,1)))
    starting=parameters_to_model(initial_guess())
    result=estimate_strengths_from_beam_rotation(beam_shift_measurements=measured,starting_model=starting)
    np.testing.assert_allclose(result['objective_strength'],p[11],atol=1e-9)
    np.testing.assert_allclose(list(result['excitation_gains'].values()),np.exp(p[12:16]),atol=1e-9)
    assert starting['objective_strength']==-.4
