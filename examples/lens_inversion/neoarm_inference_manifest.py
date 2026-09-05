"""Export explicit, truth-free fitting inputs from immutable simulation audits.

The fitter receives measured responses, their covariance, known input steps
and control coordinates. No expected/noiseless images, generating q labels,
source covariance, optical truth or truth-based information enters this NPZ.
The generating truth file is referenced separately for post-fit checks only.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

PUBLIC_ARRAY_KEYS=frozenset({
    'noisy_counts','heldout_noisy_counts','fitted_counts','heldout_fitted_counts',
    'image_parameters','heldout_image_parameters','image_parameter_covariances','heldout_image_parameter_covariances',
    'response','heldout_response','response_covariance','heldout_response_covariance',
    'paired_response_covariance','joint_response_covariance','accepted_normalized_dac','heldout_normalized_dac',
    'pl1_controls','il3_controls','training_mask','heldout_mask','raw_dac_values','heldout_raw_dac_values',
    'input_steps_grouped','input_states_grouped','detector_pixel_m',
})


def inference_arrays(arrays):
    """Allowlist, not a truth-name blacklist; future fields are excluded."""
    return {key:np.asarray(arrays[key]) for key in sorted(PUBLIC_ARRAY_KEYS.intersection(arrays.keys()))}


def export(audit_path,output,*,calibration_manifest=None):
    root=Path(__file__).resolve().parents[2];audit_path=Path(audit_path).resolve();output=Path(output).resolve()
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();audit=json.loads(audit_path.read_text())
    for name,value in audit['source_hashes'].items():
        if digest(root/name)!=value:raise ValueError(f'Stale simulation audit source:{name}')
    input_path=root/audit['arrays_path']
    if digest(input_path)!=audit['arrays_sha256']:raise ValueError('Stale simulation array hash.')
    if not all(row['passed'] for row in audit['independent_gaussian_checks']):raise ValueError('Independent Gaussian admission failed.')
    with np.load(input_path,allow_pickle=False) as arrays:public=inference_arrays(arrays)
    output.parent.mkdir(parents=True,exist_ok=True);arrays_path=output.with_suffix('.npz')
    np.savez_compressed(arrays_path,**public)
    sources=dict(audit['source_hashes']);sources.update({str(p.relative_to(root)):digest(p) for p in (Path(__file__),audit_path,input_path)})
    report=dict(schema='neoarm-optical-inference-v1',backend=audit['backend'],source_hashes=sources,
        simulation_audit_path=str(audit_path.relative_to(root)),simulation_audit_sha256=digest(audit_path),
        arrays_path=str(arrays_path.relative_to(root)),arrays_sha256=digest(arrays_path),
        public_array_keys=sorted(public),input_steps_grouped=audit['input_steps_grouped'],detector_pixel_m=audit['detector_pixel_m'],
        truth_path=audit['truth_path'],truth_sha256=audit['truth_sha256'],truth_use='post-optimization diagnostics only',
        independent_gaussian_checks=[{'passed':True,'scope':'independent physical Gaussian propagation and pixel quadrature; details pinned in simulation audit'}],
        generated_frame_count=audit['generated_frame_count'],incident_electrons_per_frame=audit['incident_electrons_per_frame'],
        optical_parameters_fitted=False,generating_truth_fields_in_public_arrays=False,
        limitations=['Only allowlisted noisy image measurements, fitted image summaries, controls and known calibration inputs are numerical fitting inputs.',
            'Simulation/validation source hashes establish provenance; they are not additional observations or parameter priors.',
            'Noise covariance is conditionally estimated from the synthetic image likelihood; shared input calibration and other instrument uncertainties remain.'])
    for name in ('paired_data_admitted','image_generation_admitted','training_frame_count','heldout_frame_count',
                 'training_group_indices','heldout_group_indices','pl1_controls','il3_controls','control_definitions',
                 'calibration_response_noise_independent','shared_input_calibration_uncertainty_still_required'):
        if name in audit:report[name]=audit[name]
    if calibration_manifest is not None:
        path=Path(calibration_manifest).resolve();report['independent_calibration_report']=str(path.relative_to(root))
        report['independent_calibration_report_sha256']=digest(path);sources[str(path.relative_to(root))]=digest(path)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--calibration-manifest',type=Path)
    args=parser.parse_args();export(args.audit,args.output,calibration_manifest=args.calibration_manifest)
