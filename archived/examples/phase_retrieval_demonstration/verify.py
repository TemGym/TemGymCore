"""Check saved results and predict an unused IL1 setting with native propagation."""
import json
from dataclasses import replace
import numpy as np
import jax.numpy as jnp
from demonstration import HERE, microscope, optics, grid, make_gaussian, run_to_end, evaluate_gaussians, DETECTOR_SIZE, DETECTOR_PITCH_M, Ray, Plane


def main():
    d=np.load(HERE/'dataset.npz')
    report=json.loads((HERE/'calibration.json').read_text())
    model=microscope()
    # Native ray response at reference focus and acquired IL1 currents.
    _,base=optics(model,0.)
    np.testing.assert_allclose(base['magnification'],40000.,rtol=1e-10)
    assert abs(base['effective_defocus_m'])<1e-12
    for saved in report['settings']:
        column,current=optics(model,saved['delta'])
        np.testing.assert_allclose(current['matrix'],saved['matrix'],atol=1e-12)
        # Independent ray check: each free-drift conjugate eliminates dependence
        # on the launch angle of a fixed specimen point, including extrapolations.
        rays=Ray(x=jnp.full(3,35e-9), y=jnp.zeros(3),
                 dx=jnp.array([-20e-6,0,20e-6]), dy=jnp.array([10e-6,0,-10e-6]),
                 z=float(model.components['sample'].z), voltage=model.voltage,pathlength=0.)
        for i,p in enumerate(saved['image_planes']):
            out=run_to_end(rays,(*column[:i+1],Plane(z=p['image_z_m'])))
            np.testing.assert_allclose(np.asarray(out.x),float(out.x[0]),atol=1e-12)
            np.testing.assert_allclose(np.asarray(out.y),float(out.y[0]),atol=1e-12)
    assert d['intensities'].shape == (4, DETECTOR_SIZE, DETECTOR_SIZE)
    assert d['vacuum_intensities'].shape == d['intensities'].shape
    assert report['detector_shape_yx'] == [DETECTOR_SIZE, DETECTOR_SIZE]
    np.testing.assert_allclose(d['detector_pitch_m'], DETECTOR_PITCH_M)
    assert np.isfinite(d['intensities']).all() and d['intensities'].min()>=0
    assert report['metrics']['phase_rmse_rad']<.04
    # No inverse fit at this fifth setting; propagate recovered coefficients.
    delta=-.15
    column,setting=optics(model,delta)
    xy=d['gaussian_centres_m']
    w=float(d['gaussian_waist_m'])
    basis=make_gaussian(x=xy[:,0],y=xy[:,1],z=model.components['sample'].z,
                        voltage=model.voltage,waist_x=w,waist_y=w,amp=1/np.pi)
    camera=grid(DETECTOR_SIZE,DETECTOR_PITCH_M)
    def render(coefficients):
        beam=replace(basis,amplitude=basis.amplitude*jnp.asarray(coefficients))
        out=run_to_end(beam,column)
        # Remove common path consistently, to keep optical exp(i k L) accurate.
        out=replace(out,pathlength=out.pathlength-out.pathlength[0])
        return np.abs(np.asarray(evaluate_gaussians(out,camera,method='scan')))**2
    truth=render(d['truth_coefficients'])
    prediction=render(d['retrieved_coefficients'])
    error=float(np.linalg.norm(prediction-truth)/np.linalg.norm(truth))
    assert error<.005,error
    result={'held_out_IL1_delta':delta,'held_out_IL1_current_A':setting['current_A'],
            'held_out_intensity_relative_error':error,
            'detector_shape_yx':list(camera.shape),
            'checks':'PASS: reference focus, saved calibration, 256×256 intensities, lens conjugate rays, finite intensities, phase error, unused-setting prediction'}
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
