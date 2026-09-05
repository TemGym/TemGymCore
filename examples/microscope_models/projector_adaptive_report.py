"""Small, readable reports for the adaptive microscope-calibration experiment."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from IPython.display import HTML, display
import html

from projector_notebook_helpers import _table
from projector_adaptive_experiments import model_to_parameters, score_parameters


def show_round_results(*, baseline, calibration_rounds):
    rows=[]
    initial=score_parameters(parameters=model_to_parameters(baseline.model),truth_directory=baseline.source_experiment.output_directory)
    rows.append(['Starting calibration','82','—','—',f"{100*max(abs(v['relative_error']) for v in initial.values()):.3f}%"])
    labels=['Mixed projector settings','Also vary the objective','Wider compensated settings']
    for label,state in zip(labels,calibration_rounds):
        f=state.feedback
        rows.append([label,str(f['image_count']),f"{f['before_prediction_rms']:.2f}",
                     f"{f['after_refit_rms']:.2f}",f"{100*f['maximum_relative_parameter_error']:.3f}%"])
    _table(['Round','Images used in fit','New-data error before fit / noise','After refit / noise','Largest parameter error'],rows)
    return rows


def show_final_parameters(*, calibration):
    errors=score_parameters(parameters=model_to_parameters(calibration.model),truth_directory=calibration.source_experiment.output_directory)
    labels={
        'lens_positions_mm':'Position (mm)','lens_lengths_mm':'Length (mm)',
        'detector_position_mm':'Detector position (mm)','objective_strength':'Objective q',
        'excitation_gains':'Excitation gain','beam_curvature_per_m':'Beam curvature (1/m)',
        'angular_spread_mrad':'Angular spread (mrad)','sample_rotation_deg':'Specimen rotation (degrees)',
    }
    rows=[]
    for key,v in errors.items():
        prefix,_,name=key.partition('.')
        label=f'{name}: {labels[prefix]}' if name else labels[prefix]
        rows.append([label,f"{v['truth']:.6g}",f"{v['fitted']:.6g}",f"{100*v['relative_error']:+.4f}%"])
    body='<table><thead><tr>'+''.join(f'<th>{s}</th>' for s in ['Parameter','Truth','Fit','Relative error'])+'</tr></thead><tbody>'
    body+=''.join('<tr>'+''.join(f'<td>{html.escape(v)}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    display(HTML('<details><summary>All 19 fitted parameters</summary>'+body+'</details>'))
    return errors


def show_selected_controls(*, calibration, reference_controls):
    design=json.loads((calibration.output_directory/'design.json').read_text())
    base=np.asarray(reference_controls)
    if len(base)==4:base=np.r_[1.,base]
    changes=100*(np.asarray(design['lens_controls'])/base-1)
    rows=[[str(i+1),*[f'{v:+.3f}%' for v in row]] for i,row in enumerate(changes)]
    _table(['Setting','Objective','IL1','IL2','IL3','PL1'],rows)


def show_recovery_checks(*, output_directory, true_detector_position_mm=None):
    folder=Path(output_directory)
    prediction=json.loads((folder/'fresh_predictions/prediction_check.json').read_text())
    rows=[['Fresh images, without refitting',f"{prediction['image_count']} images; error {prediction['prediction_rms']:.3f} noise units"]]
    for i in [1,2]:
        errors=json.loads((folder/f'noise_repeat_{i}/parameter_errors.json').read_text())
        rows.append([f'Independent noise repeat {i}',f"All 19 errors below {100*max(abs(v['relative_error']) for v in errors.values()):.3f}%"])
    information=json.loads((folder/'identifiability_checks.json').read_text())
    no_prior=information['without_soft_priors']
    rows.append(['Remove soft priors',f"Detector changes by {no_prior['detector_change_mm']:+.4f} mm"])
    if (folder/'ab_only_geometry.json').exists():
        ab=json.loads((folder/'ab_only_geometry.json').read_text())[0]
        rows.append(['Use calibrated displacements alone',f"All 16 optical errors below {100*ab['maximum_relative_optical_error']:.3f}%; no common illumination parameters needed"])
    starts=json.loads((folder/'final_landscape/starts.json').read_text())
    valid=[v for v in starts if v['success']]
    best=min(v['total_squared_residual'] for v in valid)
    close=sum(v['total_squared_residual']<=best+1 for v in valid)
    rows.append(['Direct fits from 20 dispersed starts',f'{close}/20 reached the good solution; others have much larger errors or did not converge'])
    staged=json.loads((folder/'ab_initialised_starts/summary.json').read_text())
    accepted=sum(v.get('joint_success',False) and v.get('joint_rms',np.inf)<1.5 for v in staged)
    rows.append(['Fit calibrated displacements first, then all data',f'{accepted}/{len(staged)} starts reached the same good solution'])
    _table(['Check','Result'],rows)
    profiles=json.loads((folder/'final_landscape/detector_profile.json').read_text())
    z=np.array([v['fixed_detector_position_mm'] for v in profiles])
    cost=np.array([v['total_squared_residual']-best if v['success'] else np.nan for v in profiles])
    # A profiled optimizer may beat the unrestricted fit by floating-point
    # tolerance. Keep meaningful negative differences visible for diagnosis.
    if np.nanmin(cost)>-1e-3:
        cost=np.maximum(cost,0.)
    fitted=min(valid,key=lambda v:v['total_squared_residual'])['model']['detector_position_mm']
    fig,axes=plt.subplots(1,2,figsize=(10,3.2),constrained_layout=True)
    for ax,mask in [(axes[0],abs(z-fitted)<1.1),(axes[1],np.ones(len(z),bool))]:
        ax.plot(z[mask],cost[mask],'o-')
        ax.axvline(fitted,color='tab:orange',linestyle='--',label='Best fit')
        if true_detector_position_mm is not None:
            ax.axvline(true_detector_position_mm,color='black',linestyle=':',label='Generating position')
        ax.set(xlabel='Fixed detector position (mm)',ylabel='Squared error above best fit')
        ax.xaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_title('Fine detector-position scan')
    axes[1].set_title('Wider scan, same 18 parameters refitted')
    axes[1].set_yscale('symlog',linthresh=1.)
    if np.nanmin(cost)>=0:
        for ax in axes:ax.set_ylim(bottom=0)
    axes[0].legend()
    plt.show()
    _table(['Data','Smallest sensitivity value'],[
        [label,f"{information[key]['singular_values'][-1]:.6g}"]
        for key,label in [('baseline','Starting calibration'),('projector_combinations','Mixed projectors'),
                          ('objective_combinations','Also objective'),('wider_combinations','Wider settings')]])
    print('Sensitivity values use the same parameter units and measured noise. '
          'The smallest value measures the weakest local parameter combination; it is not a global uniqueness proof.')
    return rows
