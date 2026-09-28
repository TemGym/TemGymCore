"""Transport-of-intensity reconstruction from native Gaussian IL1 images.

Three images reconstruct phase; a second, half-step pair checks the derivative.
The existing modified-smiley wave is held fixed for comparison with L-BFGS.
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import json
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
from scipy.ndimage import map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from demonstration import (HERE, microscope, column_matrix, make_gaussian, run_to_end,
                           evaluate_gaussians, grid, save)
from temgym_core.constants import energy2wavelength
from temgym_core.gaussian import FreeSpacePropagator


def solve_tie(intensity, derivative, wavelength, pitch):
    """Solve -div(I grad(phi)) = k*dI/dz by conservative finite volumes.

    Arithmetic face intensities, homogeneous Neumann boundary conditions;
    remove the incompatible constant RHS and pin one phase value for the gauge.
    No truth, beam coefficients or phase-shaped prior enters this solve.
    """
    intensity = np.asarray(intensity, float)
    derivative = np.asarray(derivative, float)
    if intensity.shape != derivative.shape or intensity.min() <= 0:
        raise ValueError('Use matching arrays and positive intensity on the domain.')
    ny, nx = intensity.shape
    ids = np.arange(ny*nx).reshape(ny,nx)
    a = np.r_[ids[:,:-1].ravel(), ids[:-1,:].ravel()]
    b = np.r_[ids[:,1:].ravel(), ids[1:,:].ravel()]
    w = np.r_[((intensity[:,:-1]+intensity[:,1:])/2).ravel(),
              ((intensity[:-1,:]+intensity[1:,:])/2).ravel()]
    L = coo_matrix((np.r_[w,w,-w,-w],
                   (np.r_[a,b,a,b],np.r_[a,b,b,a])),shape=(ny*nx,ny*nx)).tocsr()
    rhs = (2*np.pi/wavelength)*pitch**2*derivative.ravel()
    correction = float(rhs.mean())
    projected = rhs-correction
    phase = np.zeros(ny*nx)
    phase[1:] = spsolve(L[1:,1:],projected[1:])
    phase -= phase.mean()
    residual = float(np.linalg.norm(L@phase-projected)/np.linalg.norm(projected))
    return phase.reshape(ny,nx), dict(linear_residual=residual,
        removed_rhs_mean=correction,
        compatibility_correction_fraction=float(abs(correction)*np.sqrt(rhs.size)/np.linalg.norm(rhs)))


def main():
    d = np.load(HERE/'dataset.npz')
    model = microscope()
    zs = float(model.components['sample'].z)
    i0 = float(model.components['IL1'].current)
    wavelength = float(energy2wavelength(model.voltage))
    def column(delta):
        m = model.with_currents(IL1=i0*(1+delta))
        return tuple(c for c in m.column if float(c.z)>zs)
    transfer = jax.jit(lambda delta: column_matrix(column(delta),source_z=zs,voltage=model.voltage))
    def defocus(delta):
        T = np.asarray(transfer(delta))
        return float(np.trace(np.linalg.solve(T[:2,:2],T[:2,2:4]))/2)
    # Work on a common 176 nm central field at the reference detector sampling.
    camera = grid(256,55e-6)
    n, pitch = 128,55e-6/40000
    object_grid = grid(n,pitch)
    xy = d['gaussian_centres_m']; waist = float(d['gaussian_waist_m'])
    beam = make_gaussian(x=xy[:,0],y=xy[:,1],z=zs,voltage=model.voltage,
                         waist_x=waist,waist_y=waist,amp=d['truth_coefficients']/np.pi)
    # Native renderer is independent of the pixel-based TIE inverse.
    render = lambda b,g: np.asarray(evaluate_gaussians(b,g,method='scan'))
    targets = [-200e-9,-100e-9,0.,100e-9,200e-9]
    raw, registered, records = [],[],[]
    for z in targets:
        delta = 0. if z==0 else brentq(lambda v:defocus(v)-z,-.08,.08,xtol=1e-13)
        col = column(delta)
        T = np.asarray(transfer(delta)); A,B = T[:2,:2],T[:2,2:4]
        zeff = float(np.trace(np.linalg.solve(A,B))/2)
        np.testing.assert_allclose(np.linalg.solve(A,B),zeff*np.eye(2),atol=1e-13)
        # Omit the unobservable axial carrier throughout the column, rather
        # than subtracting it after tiny relative actions have lost precision.
        out = jax.jit(lambda b:run_to_end(b,col,
            propagator=FreeSpacePropagator(include_axial_path=False)))(beam)
        out = replace(out,pathlength=out.pathlength-out.pathlength[0])
        image = np.abs(render(out,camera))**2
        # x_detector = A*x_object. Multiplication by det(A) restores the
        # specimen-area irradiance before computing a derivative in B/A.
        detector_xy = np.asarray(object_grid.coords)@A.T
        indices = detector_xy/55e-6+(256-1)/2
        common = map_coordinates(image,[indices[:,1],indices[:,0]],order=5,
                                  mode='constant',cval=np.nan).reshape(n,n)*abs(np.linalg.det(A))
        assert np.isfinite(common).all() and common.min()>0
        raw.append(image);registered.append(common)
        records.append(dict(IL1_delta=delta,current_A=i0*(1+delta),effective_defocus_m=zeff,
                            magnification=float(np.sqrt(np.linalg.det(A))),
                            rotation_deg=float(np.degrees(np.arctan2(A[1,0],A[0,0]))),matrix=T.tolist()))
        print('Rendered',records[-1]['effective_defocus_m']*1e9,'nm, IL1',i0*(1+delta),'A',flush=True)
    raw, registered = np.array(raw),np.array(registered)
    focus = registered[2]
    truth = render(beam,object_grid)
    true_phase = np.angle(truth)
    mask = (np.linalg.norm(np.asarray(object_grid.coords),axis=1)<70e-9).reshape(n,n)
    phases=[];derivatives=[];metrics=[]
    for lo,hi in [(0,4),(1,3)]:
        step = (records[hi]['effective_defocus_m']-records[lo]['effective_defocus_m'])/2
        derivative = (registered[hi]-registered[lo])/(2*step)
        phase,check = solve_tie(focus,derivative,wavelength,pitch)
        # Truth enters only here for piston alignment and validation.
        phase -= np.mean((phase-true_phase)[mask])
        error = phase-true_phase
        check.update(step_nm=step*1e9,phase_rmse_rad=float(np.sqrt(np.mean(error[mask]**2))),
                     phase_correlation=float(np.corrcoef(phase[mask],true_phase[mask])[0,1]))
        phases.append(phase);derivatives.append(derivative);metrics.append(check)
    print(json.dumps(metrics,indent=2),flush=True)
    # Save before acceptance checks so failed runs remain inspectable.
    np.savez_compressed(HERE/'tie_dataset.npz',raw_intensities=raw,
        registered_intensities=registered,derivatives=np.array(derivatives),
        reconstructed_phases=np.array(phases),true_phase=true_phase,
        true_wave=truth,mask=mask,specimen_pitch_m=pitch,detector_pitch_m=55e-6,
        effective_defocus_m=[r['effective_defocus_m'] for r in records])
    report=dict(method='Variable-intensity finite-volume TIE',wavelength_m=wavelength,
        detector_shape=[256,256],reconstruction_shape=[n,n],specimen_pitch_m=pitch,
        acquisition_count=3,validation_extra_images=2,
        boundary_condition='Zero normal phase-current flux at reconstruction boundary',
        sample='Existing phase-remapped smiley represented by 2304 Gaussian beamlets',
        settings=records,metrics=metrics)
    (HERE/'tie_results.json').write_text(json.dumps(report,indent=2)+'\n')
    make_figures(registered,phases,true_phase,mask,pitch,metrics)
    assert all(m['linear_residual']<1e-8 for m in metrics)
    assert metrics[0]['phase_rmse_rad']<.03,metrics
    assert metrics[0]['phase_correlation']>.995,metrics
    assert abs(metrics[0]['phase_rmse_rad']-metrics[1]['phase_rmse_rad'])<.005,metrics


def make_figures(images,phases,truth,mask,pitch,metrics):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none'})
    extent=np.array([-1,1,-1,1])*truth.shape[0]*pitch*1e9/2
    fig,axs=plt.subplots(1,3,figsize=(13,4.7),layout='constrained')
    for ax,values,title in zip(axs,[truth,phases[0],phases[0]-truth],
                             ['True phase','Recovered phase (TIE)','TIE − true phase']):
        err=ax is axs[2]
        im=ax.imshow(values,origin='lower',extent=extent,cmap='RdBu_r' if err else 'viridis',
                     vmin=-.03 if err else 0,vmax=.03 if err else 1.2)
        ax.set(title=title,xlabel='x (nm)',ylabel='y (nm)')
        fig.colorbar(im,ax=ax,shrink=.8,label='rad')
    fig.suptitle('TIE from three calibrated IL1 images: −200 nm, focus, +200 nm\n'
                 f"Phase RMS {metrics[0]['phase_rmse_rad']:.4f} rad · noiseless Gaussian simulation",fontsize=16)
    save(fig,'06_TIE_phase_retrieval')
    fig,axs=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for ax,i,title in zip(axs,[0,2,4],['−200 nm','Focus','+200 nm']):
        im=ax.imshow(images[i],origin='lower',extent=extent,cmap='gray',vmin=0,vmax=1.15)
        ax.set(title=title,xlabel='Registered specimen x (nm)',ylabel='y (nm)')
    fig.colorbar(im,ax=axs,label='Intensity in specimen-area units',shrink=.8)
    fig.suptitle('Three IL1 exposures after calibrated scale and rotation correction',fontsize=15)
    save(fig,'06b_TIE_registered_images')


if __name__=='__main__':
    main()
