"""IL1 focal diversity with native TemGym Gaussian beamlets.

Run with the repository environment: python demonstration.py.
All outputs are confined to this example directory.
"""
from pathlib import Path
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MPLBACKEND', 'Agg')
import json
from dataclasses import replace
from types import SimpleNamespace
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from temgym_core.microscope import Microscope, find_imaging_condition
from temgym_core.gaussian import make_gaussian
from temgym_core.components import Plane
from temgym_core.run import run_to_end, column_matrix
from temgym_core.evaluate import evaluate_gaussians, _beam_field, _prepare_gaussian_params
from temgym_core.plotting import plot_model, legacy_beam_plot_params
from temgym_core.ray import Ray
from temgym_core.source import _gaussian_cardinal_coefficients
from temgym_core.plotting import _compute_cumulative_rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / 'figures'
COLORS = ['#31688e', '#35a779', '#de9b24', '#b54b72']
DETECTOR_SIZE = 256
DETECTOR_PITCH_M = 55e-6
LENS_NAMES = ['Objective', 'IL1', 'IL2', 'IL3', 'PL1']


def grid(n, pitch):
    v = (np.arange(n) - (n-1)/2)*pitch
    x, y = np.meshgrid(v, v)
    return SimpleNamespace(shape=(n,n), coords=jnp.asarray(np.c_[x.ravel(),y.ravel()]))


def microscope():
    model = Microscope.from_toml(REPO / 'examples/microscope_models/microscope.toml')
    initial = model.with_settings(magnification=40000.)
    return find_imaging_condition(initial, magnification=40000., vary=['IL1','IL2','IL3']).microscope


def optics(model, delta):
    current = float(model.components['IL1'].current)*(1+delta)
    changed = model.with_currents(IL1=current)
    zs = float(model.components['sample'].z)
    col = tuple(c for c in changed.column if float(c.z)>zs)
    T = np.asarray(column_matrix(col, source_z=zs, voltage=model.voltage))
    A, B = T[:2,:2], T[:2,2:4]
    effective = np.linalg.solve(A,B)
    z_eff = np.trace(effective)/2
    np.testing.assert_allclose(effective, z_eff*np.eye(2), atol=1e-13)
    col = (*col[:-1], replace(col[-1], shape=(DETECTOR_SIZE, DETECTOR_SIZE),
                             pixel_size=(DETECTOR_PITCH_M, DETECTOR_PITCH_M)))
    return col, dict(delta=float(delta), current_A=current,
                    effective_defocus_m=float(z_eff),
                    magnification=float(np.sqrt(np.linalg.det(A))),
                    rotation_deg=float(np.degrees(np.arctan2(A[1,0],A[0,0]))),
                    matrix=T.tolist(), image_planes=lens_image_planes(col, zs, model.voltage))


def lens_image_planes(column, source_z, voltage):
    """Specimen conjugate after each lens, assuming free drift after its exit.

    B + distance*D = 0 images the specimen independently of launch angle.
    A downstream lens can intercept this extrapolation; a negative distance
    denotes a virtual image. Neither should be labelled an actual ray crossing.
    """
    planes = []
    for i, (name, component) in enumerate(zip(LENS_NAMES, column[:-1])):
        T = np.asarray(column_matrix(column[:i+1], source_z=source_z, voltage=voltage))
        B, D = T[:2, 2:4], T[2:4, 2:4]
        distance = float(-np.sum(B*D)/np.sum(D*D))
        np.testing.assert_allclose(B + distance*D, 0., atol=1e-12)
        exit_z = float(component.z + getattr(component, 'length', 0.))
        image_z = exit_z + distance
        kind = ('virtual' if distance < 0 else
                'intercepted by next lens' if i < len(column)-2 and image_z > float(column[i+1].z)
                else 'real')
        planes.append(dict(lens=name, lens_exit_z_m=exit_z, image_z_m=image_z,
                           distance_from_lens_m=distance, kind=kind))
    return planes


def response(beam, coords):
    """Native per-beam fields, cached as a linear complex-wave operator.

    Optics and phases evaluated in float64; storage and inverse matvecs use
    complex64. This is the same field kernel used by evaluate_gaussians.
    """
    params = _prepare_gaussian_params(beam)
    # Remove an unobservable common path before exp(i k L), retaining relative L.
    params = (*params[:3], params[3]-params[3][0], *params[4:])
    f = jax.jit(jax.vmap(lambda r,d,a,l,q,k: _beam_field(r,d,a,l,q,k,coords)))
    return (f(*params).T*40000).astype(jnp.complex64)


def retrieve(operators, intensities, initial, iterations=3000):
    """Fit unconstrained complex coefficients from intensity alone.

    No target phase, target amplitude, or smiley geometry is passed here.
    The Gaussian basis/support and illumination used for initialization are known.
    """
    n = initial.size
    targets = jnp.sqrt(jnp.asarray(intensities.reshape(len(operators),-1))+1e-12)
    H = jnp.stack(operators)

    def objective(v, H, targets):
        c = v[:n]+1j*v[n:]
        u = H @ c.astype(jnp.complex64)
        residual = jnp.sqrt(jnp.abs(u)**2+1e-12)-targets
        return jnp.sum(residual**2)/jnp.sum(targets**2)
    vg = jax.jit(jax.value_and_grad(objective))
    history=[]
    def fun(v):
        val, grad = vg(jnp.asarray(v), H, targets)
        return float(val), np.asarray(grad, dtype=float)
    def callback(v):
        history.append(fun(v)[0])
        if len(history)%50==0:
            print(f'  iteration {len(history)}: amplitude loss {history[-1]:.3g}',flush=True)
    x0=np.r_[initial.real,initial.imag].astype(float)
    result=minimize(fun,x0,jac=True,method='L-BFGS-B',callback=callback,
                    options=dict(maxiter=iterations,ftol=1e-13,gtol=1e-9,maxcor=40))
    coeff=result.x[:n]+1j*result.x[n:]
    return coeff, np.array(history), dict(loss=float(result.fun), iterations=int(result.nit),
                                        converged=bool(result.success), message=str(result.message))


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for suffix in ['png','svg','pdf']:
        fig.savefig(OUT / f'{name}.{suffix}',dpi=220,facecolor='white',bbox_inches='tight')
    plt.close(fig)


def main():
    print('Devices:',jax.devices(),flush=True)
    model=microscope()
    zs=float(model.components['sample'].z)
    n, spacing=48,4e-9
    object_grid=grid(n,spacing)
    xy=np.asarray(object_grid.coords)
    illumination=np.exp(-np.sum(xy**2,axis=1)/(90e-9)**2)
    # The existing smiley recipe is retained in smiley.py. Encode its dark
    # features as phase; flip vertically so the face is upright with origin=lower.
    from smiley import smiley
    y, x = np.ogrid[-128:128, -128:128]
    signature = np.exp(1j*(3*y+7*x)*2*np.pi/256)
    raw=np.flipud(smiley(256)-0.3*signature+0.3)
    phase_map=gaussian_filter(1-np.clip(np.abs(raw),0,1),1.8)*1.2
    sample_width=155e-9
    indices=(xy/sample_width+0.5)*256-0.5
    phase=map_coordinates(phase_map,[indices[:,1],indices[:,0]],order=3,mode='constant',cval=0)
    # Native cardinal interpolation reproduces the designed complex wave at
    # lattice nodes instead of blurring phase gradients into strong absorption.
    coefficients=(np.pi*_gaussian_cardinal_coefficients(
        (illumination*np.exp(1j*phase)).reshape(n,n),width_over_step=1.)).ravel()
    initial=(np.pi*_gaussian_cardinal_coefficients(
        illumination.reshape(n,n),width_over_step=1.)).ravel().astype(complex)
    basis=make_gaussian(x=xy[:,0],y=xy[:,1],z=zs,voltage=model.voltage,
                        waist_x=spacing,waist_y=spacing,amp=1/np.pi)
    image_grid=grid(DETECTOR_SIZE,DETECTOR_PITCH_M)
    print("Detector image shape:", image_grid.shape, flush=True)
    deltas=np.array([-.28,-.20,-.10,.08])
    operators=[]; records=[]; columns=[]
    for d in deltas:
        col,rec=optics(model,d)
        propagated=jax.jit(lambda b:run_to_end(b,col))(basis)
        operators.append(response(propagated,image_grid.coords))
        records.append(rec); columns.append(col)
        print(rec,flush=True)
    # Measurements are coherent Gaussian intensities on the same physical camera.
    data=np.array([np.abs(np.asarray(H@jnp.asarray(coefficients,dtype=jnp.complex64)))**2
                   for H in operators]).reshape(4,*image_grid.shape)
    vacuum=np.array([np.abs(np.asarray(H@jnp.asarray(initial,dtype=jnp.complex64)))**2
                     for H in operators]).reshape(data.shape)
    # Independently check the cached operator against the public native renderer.
    full=replace(basis,amplitude=basis.amplitude*jnp.asarray(coefficients))
    native=run_to_end(full,columns[0])
    check_grid=SimpleNamespace(shape=(1,101),coords=image_grid.coords[::367][:101])
    reference=np.abs(np.asarray(evaluate_gaussians(native,check_grid,method='scan'))).ravel()**2*40000**2
    actual=data[0].ravel()[::367][:101]
    native_error=float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))
    assert native_error<2e-4, native_error
    print('Native renderer relative intensity error:',native_error,flush=True)
    recovered,history,fit=retrieve(operators,data,initial)
    print('Calibrated fit',fit,flush=True)
    # A controlled error in IL1 calibration: assume only 80% of each current step.
    bad=[]
    for d in deltas:
        col,_=optics(model,0.8*d)
        bad.append(response(jax.jit(lambda b:run_to_end(b,col))(basis),image_grid.coords))
    wrong,wrong_history,wrong_fit=retrieve(bad,data,initial,iterations=3000)
    print('Miscalibrated fit',wrong_fit,flush=True)
    sample_grid=grid(192,1e-9)
    E=response(basis,sample_grid.coords)/40000
    incident=np.asarray(E@jnp.asarray(initial,dtype=jnp.complex64))
    truth=np.asarray(E@jnp.asarray(coefficients,dtype=jnp.complex64))
    recon=np.asarray(E@jnp.asarray(recovered,dtype=jnp.complex64))
    wrong_wave=np.asarray(E@jnp.asarray(wrong,dtype=jnp.complex64))
    mask=np.linalg.norm(np.asarray(sample_grid.coords),axis=1)<77e-9
    def align(w):
        return w*np.exp(-1j*np.angle(np.vdot(truth[mask],w[mask])))
    recon,wrong_wave=align(recon),align(wrong_wave)
    phase_error=np.angle(recon*np.conj(truth))
    bad_error=np.angle(wrong_wave*np.conj(truth))
    metrics=dict(native_renderer_relative_error=native_error,
                 phase_rmse_rad=float(np.sqrt(np.mean(phase_error[mask]**2))),
                 wrong_calibration_phase_rmse_rad=float(np.sqrt(np.mean(bad_error[mask]**2))),
                 complex_wave_relative_error=float(np.linalg.norm(recon[mask]-truth[mask])/np.linalg.norm(truth[mask])),
                 calibrated_fit=fit,wrong_calibration_fit=wrong_fit)
    assert metrics['phase_rmse_rad']<0.04,metrics
    assert fit['loss']<wrong_fit['loss'],metrics
    print(json.dumps(metrics,indent=2),flush=True)
    np.savez_compressed(HERE/'dataset.npz',intensities=data,vacuum_intensities=vacuum,delta_IL1=deltas,
                        detector_pitch_m=DETECTOR_PITCH_M,gaussian_centres_m=xy,gaussian_waist_m=spacing,
                        initial_coefficients=initial,truth_coefficients=coefficients,
                        retrieved_coefficients=recovered,wrong_coefficients=wrong,
                        specimen_truth=truth.reshape(sample_grid.shape),
                        specimen_retrieved=recon.reshape(sample_grid.shape),
                        specimen_wrong=wrong_wave.reshape(sample_grid.shape),
                        specimen_incident=incident.reshape(sample_grid.shape),
                        specimen_pitch_m=1e-9,mask=mask.reshape(sample_grid.shape),
                        history=history,wrong_history=wrong_history)
    (HERE/'calibration.json').write_text(json.dumps(dict(voltage_V=model.voltage,
        baseline_IL1_A=float(model.components['IL1'].current),
        detector_shape_yx=list(image_grid.shape), detector_pitch_m=DETECTOR_PITCH_M,
        settings=records,metrics=metrics),indent=2)+'\n')
    make_figures(model,columns,records,data,vacuum,truth,recon,wrong_wave,phase_error,mask,history,wrong_history,metrics)


def make_figures(model,columns,records,data,vacuum,truth,recon,wrong,err,mask,history,wrong_history,metrics):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none'})
    height, width = data.shape[-2:]
    extent=np.array([-width,width,-height,height])*DETECTOR_PITCH_M/2*1e3
    specimen_extent=[-96,96,-96,96]
    scaled=np.array([data[j]*(r['magnification']/40000)**2 for j,r in enumerate(records)])
    vmax=np.quantile(scaled,.999)
    contrast=np.where(vacuum>vacuum.max(axis=(1,2),keepdims=True)*.03,
                      data/np.maximum(vacuum,1e-12)-1,np.nan)
    def camera(ax,j,normalized=False):
        r=records[j]
        values=contrast[j] if normalized else scaled[j]
        im=ax.imshow(values,origin='lower',extent=extent,
                     cmap='RdBu_r' if normalized else 'gray',
                     vmin=-.25 if normalized else 0,vmax=.25 if normalized else vmax,
                     interpolation='nearest')
        ax.set_title(f"IL1 {r['current_A']:.3f} A ({r['delta']:+.0%})\n"+
                     f"$z_{{eff}}$ = {r['effective_defocus_m']*1e6:+.2f} µm",color=COLORS[j],fontsize=12)
        half=min(min(width,height)*DETECTOR_PITCH_M/2*1e3,110e-9*r['magnification']*1e3)
        ax.set(xlabel='Detector x (mm)',ylabel='y (mm)',xlim=(-half,half),ylim=(-half,half))
        return im
    fig,axs=plt.subplots(1,4,figsize=(15,4),layout='constrained')
    for j,ax in enumerate(axs): im=camera(ax,j)
    fig.colorbar(im,ax=axs,label='Intensity × calibrated M² (a.u.)',shrink=.65)
    save(fig,'01_detector_defocus_series')
    fig,axs=plt.subplots(1,4,figsize=(15,4),layout='constrained')
    for j,ax in enumerate(axs): im=camera(ax,j,normalized=True)
    fig.colorbar(im,ax=axs,label='Contrast I / I(vacuum) − 1',shrink=.65)
    save(fig,'01b_detector_phase_contrast')
    fig,axs=plt.subplots(1,4,figsize=(15,4.2),layout='constrained')
    ph=np.angle(truth).reshape(192,192); pr=np.angle(recon).reshape(192,192)
    pb=np.angle(wrong).reshape(192,192)
    for ax,p,title in zip(axs[:2],[ph,pr],['True exit-wave phase','Retrieved · calibrated IL1']):
        im=ax.imshow(p,extent=specimen_extent,origin='lower',cmap='viridis',vmin=0,vmax=1.2)
        ax.set(title=title,xlabel='Specimen x (nm)',ylabel='y (nm)')
    fig.colorbar(im,ax=list(axs[:2]),label='Phase (rad)',shrink=.72)
    im=axs[2].imshow(pb,extent=specimen_extent,origin='lower',cmap='twilight',vmin=-np.pi,vmax=np.pi)
    axs[2].set(title='Retrieved · 20% step error',xlabel='Specimen x (nm)',ylabel='y (nm)')
    fig.colorbar(im,ax=axs[2],label='Phase (rad)',shrink=.72,ticks=[-np.pi,0,np.pi])
    im=axs[3].imshow(err.reshape(192,192),extent=specimen_extent,origin='lower',cmap='RdBu_r',vmin=-.05,vmax=.05)
    axs[3].set(title=f"Calibrated error\nRMS {metrics['phase_rmse_rad']:.4f} rad",xlabel='Specimen x (nm)',ylabel='y (nm)')
    fig.colorbar(im,ax=axs[3],label='Phase error (rad)',shrink=.72)
    save(fig,'02_phase_retrieval')
    make_ray_figures(model, columns, records)
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for j,r in enumerate(records):
        axs[0].scatter(r['current_A'],r['effective_defocus_m']*1e6,color=COLORS[j],s=65)
        axs[1].scatter(r['magnification']/1000,r['rotation_deg'],color=COLORS[j],s=65)
    axs[0].set(xlabel='IL1 current (A)',ylabel='Effective defocus B/A (µm)',title='Calibrated defocus diversity')
    axs[1].set(xlabel='Magnification (× 1,000)',ylabel='Image rotation (degrees)',title='Scale and rotation also change')
    axs[2].semilogy(history,label='Calibrated',color='#31688e')
    axs[2].semilogy(wrong_history,label='20% step error',color='#b54b72')
    axs[2].set(xlabel='Iteration',ylabel='Relative squared amplitude residual',title='One wave must fit all four images')
    axs[2].legend()
    save(fig,'04_calibration_and_convergence')
    # 16:9 overview designed to be inserted as one slide figure.
    fig=plt.figure(figsize=(16,9),facecolor='white')
    gs=fig.add_gridspec(2,4,left=.055,right=.94,bottom=.09,top=.80,hspace=.45,wspace=.30)
    fig.text(.055,.945,'Calibrated IL1 control makes phase measurable',fontsize=25,weight='bold')
    fig.text(.055,.895,'Four IL1 settings → calibrated Gaussian propagation → one recovered exit wave',fontsize=15,color='#475569')
    fig.text(.055,.855,'Detector crops: normalized contrast I / I(vacuum) − 1; shared range −0.25 to +0.25',fontsize=11,color='#475569')
    for j in range(4):
        ax=fig.add_subplot(gs[0,j]); camera(ax,j,normalized=True); ax.set_ylabel('y (mm)' if j==0 else '')
    titles=['True phase','Retrieved phase','Phase error']
    for j,(p,title) in enumerate(zip([ph,pr,err.reshape(192,192)],titles)):
        ax=fig.add_subplot(gs[1,j])
        iserr=j==2
        im=ax.imshow(p,extent=specimen_extent,origin='lower',cmap='RdBu_r' if iserr else 'viridis',
                     vmin=-.05 if iserr else 0,vmax=.05 if iserr else 1.2)
        ax.set(title=title,xlabel='Specimen x (nm)',ylabel='y (nm)' if j==0 else '')
        fig.colorbar(im,ax=ax,shrink=.75,label='rad')
    ax=fig.add_subplot(gs[1,3]);ax.axis('off')
    ax.text(0,.95,'Known for each exposure',weight='bold',fontsize=15)
    ax.text(0,.70,'IL1 current → full optical model\nDefocus + magnification + rotation',fontsize=12,linespacing=1.6)
    ax.text(0,.42,f"Phase RMS error\n{metrics['phase_rmse_rad']:.4f} rad",fontsize=19,color='#31688e',weight='bold')
    ax.text(0,.06,'200 kV · coherent Gaussian beamlets\nIdeal lenses · noiseless synthetic data\nComplex wave fitted from intensity only',fontsize=11,linespacing=1.6)
    save(fig,'00_slide_overview')

def make_ray_figures(model, columns, records):
    """Native rays with physical conjugate planes and detector defocus."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from matplotlib.lines import Line2D
    zs = float(model.components['sample'].z)
    rays = Ray(x=np.repeat([-65e-9,0,65e-9],3), y=np.zeros(9),
               dx=np.tile([-1e-3,0,1e-3],3), dy=np.zeros(9), z=np.full(9,zs),
               voltage=np.full(9,model.voltage), pathlength=np.zeros(9))
    def derotate(ray, prefix):
        angle = _compute_cumulative_rotation(prefix, voltage=model.voltage,
                                             include_input_rays=False)[-1]
        c,s = np.cos(angle),np.sin(angle)
        return replace(ray, x=c*ray.x+s*ray.y, y=-s*ray.x+c*ray.y,
                       dx=c*ray.dx+s*ray.dy, dy=-s*ray.dx+c*ray.dy)
    def plane(ax, z, label, color='#7c3aed', ypos=None):
        ax.axhline(z, color=color, ls='--', lw=1.3, zorder=20)
        ax.annotate(label, xy=(.02,z), xycoords=('axes fraction','data'),
                    xytext=(.02,z if ypos is None else ypos),
                    textcoords=('axes fraction','data'), fontsize=8, color=color,
                    va='bottom', bbox=dict(fc='white', ec='none', alpha=.9, pad=1.5),
                    arrowprops=None if ypos is None else dict(arrowstyle='-',color=color))
    fig, axs = plt.subplots(3,4,figsize=(17,13),layout='constrained')
    for j, (col, rec) in enumerate(zip(columns, records)):
        planes = rec['image_planes']
        objective_z = planes[0]['image_z_m']
        pp = legacy_beam_plot_params(fill_color=COLORS[j], fill_alpha=.15,
                ray_color=COLORS[j], solid_beam=False, ray_alpha=.65, interior_ray_alpha=.65,
                label_fontsize=9, fixed_xmax=7e-6, component_lw=1.5,
                label_right_pad_frac=.35, show_side_guides=False)
        # The objective image is unchanged by downstream IL1 changes.
        ax = axs[0,j]
        plot_model((col[0],Plane(z=float(col[1].z))), rays=rays,
                   plot_params=pp,ax=ax,ray_coordinate='x_rot')
        for t,label in zip(ax.texts,['Objective','']): t.set_text(label)
        plane(ax, objective_z, f'Objective image\n{(objective_z-zs)*1e3:.2f} mm')
        ax.set_ylim(float(col[1].z),zs-.018)
        ax.set_title(f"IL1 {rec['delta']:+.0%} · {rec['current_A']:.3f} A\n"
                     f"Specimen-equivalent defocus {rec['effective_defocus_m']*1e6:+.2f} µm",
                     color=COLORS[j],fontsize=11)
        ax = axs[1,j]
        at_objective = derotate(run_to_end(rays,col[:1]),col[:1])
        plot_model(col[1:-1],rays=at_objective,plot_params=replace(pp,fixed_xmax=25e-6),
                   ax=ax,ray_coordinate='x_rot')
        for t,label in zip(ax.texts,LENS_NAMES[1:]): t.set_text(label)
        ax.set_ylim(float(col[-2].z)+.012,objective_z-.012)
        plane(ax,objective_z,'Objective image')
        for p in planes[1:-1]:
            z = p['image_z_m']
            if p['kind']=='real':
                plane(ax,z,f"{p['lens']} image · {(z-zs)*1e3:.2f} mm",color=COLORS[j])
        details = [f"{p['lens']}: {(p['image_z_m']-zs)*1e3:.1f} mm ({p['kind']})"
                   for p in planes[1:-1] if p['kind']!='real']
        ax.text(.02,.03,'\n'.join(details),transform=ax.transAxes,fontsize=7.5,
                bbox=dict(fc='white',ec='none',alpha=.92),va='bottom')
        ax = axs[2,j]
        at_projector = derotate(run_to_end(rays,col[:-1]),col[:-1])
        plot_model(col[-1:],rays=at_projector,
                   plot_params=replace(pp,fixed_xmax=DETECTOR_SIZE*DETECTOR_PITCH_M/2),
                   ax=ax,ray_coordinate='x_rot')
        for t in ax.texts: t.set_text('Detector')
        image_z, detector_z = planes[-1]['image_z_m'],float(col[-1].z)
        ax.set_ylim(detector_z+.025,float(col[-2].z)-.012)
        plane(ax,image_z,f'Final image (PL1)\n{(image_z-zs)*1e3:.3f} mm',color=COLORS[j],
              ypos=image_z+.04)
        ax.axhline(detector_z,color='0.25',lw=1.1)
        ax.annotate('',xy=(.84,image_z),xytext=(.84,detector_z),
                    xycoords=('axes fraction','data'),textcoords=('axes fraction','data'),
                    arrowprops=dict(arrowstyle='<->',color=COLORS[j],lw=1.5))
        ax.text(.80,(image_z+detector_z)/2,
                f'Detector − image = {(detector_z-image_z)*1e3:+.2f} mm',
                transform=ax.get_yaxis_transform(),rotation=90,ha='right',va='center',
                color=COLORS[j],fontsize=9,bbox=dict(fc='white',ec='none',alpha=.8))
        for row in range(3):
            ax = axs[row,j]
            factor, unit = (1e3,'mm') if row==2 else (1e6,'µm')
            ax.set(xlabel=f'x ({unit})',
                   ylabel=['Objective · z from specimen (mm)',
                           'Intermediate · z from specimen (mm)',
                           'Detector · z from specimen (mm)'][row] if j==0 else '')
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos,f=factor:f'{v*f:g}'))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{(v-zs)*1e3:.0f}'))
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(False)
    fig.suptitle('Ray diagrams with objective, intermediate and final image planes\n'
                 'Dashed lines: real specimen images · fixed 256 × 256 detector',fontsize=15)
    save(fig,'03_native_ray_diagrams')

    # All lens conjugates, including virtual images and those intercepted by
    # later lenses. Keep these explicitly separate from actual ray crossings.
    _, baseline = optics(model,0.)
    fig, axs = plt.subplots(1,5,figsize=(17,5.5),layout='constrained')
    markers = {'real':'o','virtual':'v','intercepted by next lens':'x'}
    for i, (ax, name) in enumerate(zip(axs,LENS_NAMES)):
        ref = baseline['image_planes'][i]['image_z_m']
        ax.axhline((ref-zs)*1e3,color='0.4',ls='--',lw=1,label='Focused reference image')
        lens_z = records[0]['image_planes'][i]['lens_exit_z_m']
        ax.axhline((lens_z-zs)*1e3,color='0.2',lw=1.3)
        ax.text(.02,(lens_z-zs)*1e3,name+' lens',transform=ax.get_yaxis_transform(),
                va='bottom',fontsize=8,bbox=dict(fc='white',ec='none',alpha=.8))
        if i<4:
            next_z = records[0]['image_planes'][i+1]['lens_exit_z_m']
            ax.axhline((next_z-zs)*1e3,color='0.6',ls=':',lw=1)
            ax.text(.02,(next_z-zs)*1e3,LENS_NAMES[i+1]+' lens',
                    transform=ax.get_yaxis_transform(),va='bottom',fontsize=8)
        else:
            ax.axhline((float(columns[0][-1].z)-zs)*1e3,color='0.6',ls=':',lw=1)
            ax.text(.02,(float(columns[0][-1].z)-zs)*1e3,'Fixed detector',
                    transform=ax.get_yaxis_transform(),va='bottom',fontsize=8)
        for j, rec in enumerate(records):
            p = rec['image_planes'][i]
            z = p['image_z_m']
            x = rec['delta']*100
            ax.plot([x,x],[(ref-zs)*1e3,(z-zs)*1e3],color=COLORS[j],alpha=.45)
            ax.scatter(x,(z-zs)*1e3,color=COLORS[j],marker=markers[p['kind']],s=65,zorder=3)
        ax.set(title=name+' image plane',xlabel='IL1 change (%)',
               ylabel='z from specimen (mm)' if i==0 else '',xticks=[-28,-20,-10,8],xlim=(-33,13))
        ax.invert_yaxis()
        ax.margins(y=.18)
    handles = [Line2D([],[],color='0.4',ls='--',label='Image at focused reference')]
    handles += [Line2D([],[],color='0.3',ls='',marker=m,label=k.capitalize())
                for k,m in markers.items()]
    fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('Image-plane shifts after each lens (B + distance × D = 0)\n'
                 'Virtual / intercepted planes are free-drift extrapolations, not actual crossings',fontsize=14)
    save(fig,'03b_lens_image_planes')
    make_objective_image_figure(model, columns[0])


def make_objective_image_figure(model, column):
    """Show the objective's real specimen conjugate, before IL1."""
    from matplotlib.ticker import FuncFormatter
    zs = float(model.components['sample'].z)
    objective, il1 = column[:2]
    image_z = lens_image_planes(column, zs, model.voltage)[0]['image_z_m']
    image_column = (objective, Plane(z=image_z))
    T = np.asarray(column_matrix(image_column, source_z=zs, voltage=model.voltage))
    np.testing.assert_allclose(T[:2, 2:4], 0., atol=1e-12)
    magnification = float(np.sqrt(np.linalg.det(T[:2, :2])))
    angle = float(_compute_cumulative_rotation((objective,), voltage=model.voltage,
                   include_input_rays=False)[-1])
    rays = Ray(x=np.repeat([-65e-9, 0., 65e-9], 3), y=np.zeros(9),
               dx=np.tile([-1e-3, 0., 1e-3], 3), dy=np.zeros(9),
               z=np.full(9, zs), voltage=np.full(9, model.voltage),
               pathlength=np.zeros(9))
    just_after = run_to_end(rays, (objective,))
    def positions(z):
        out = run_to_end(just_after, (Plane(z=z),))
        return np.asarray(out.x)*np.cos(angle)+np.asarray(out.y)*np.sin(angle)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.8), layout='constrained')
    colors = ['#31688e', '#d39424', '#b54b72']
    # Three launch angles for each of three distinct specimen positions.
    for ax, zoom in zip(axes, [False, True]):
        if zoom:
            zz = np.array([image_z-.025, image_z, float(il1.z)])
            xx = np.stack([positions(z) for z in zz])
        else:
            zz = np.array([zs, float(objective.z), image_z, float(il1.z)])
            xx = np.vstack([np.asarray(rays.x), positions(float(objective.z)),
                            positions(image_z), positions(float(il1.z))])
        for k, (x0, color) in enumerate(zip([-65, 0, 65], colors)):
            ax.plot(xx[:, 3*k:3*k+3]*1e6, (zz-zs)*1e3,
                    color=color, lw=1.4, alpha=.8)
            ax.scatter(positions(image_z)[3*k]*1e6, (image_z-zs)*1e3,
                       color=color, s=38, zorder=6, label=f'Specimen x = {x0:+d} nm')
        ax.axhline((image_z-zs)*1e3, color='#7c3aed', ls='--', lw=1.5)
        ax.text(.02, (image_z-zs)*1e3-1.8, 'Objective image plane',
                transform=ax.get_yaxis_transform(), color='#7c3aed', fontsize=11,
                bbox=dict(fc='white', ec='none', alpha=.85), va='bottom')
        ax.axhline((float(il1.z)-zs)*1e3, color='0.4', lw=1.3)
        ax.text(.02, (float(il1.z)-zs)*1e3-1, 'IL1', transform=ax.get_yaxis_transform(),
                fontsize=11, va='bottom')
        ax.set(xlabel='x (µm)', ylabel='z from specimen (mm)', xlim=(-8, 8))
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim((float(il1.z)-zs)*1e3+4, (image_z-zs)*1e3-25 if zoom else -12)
        if not zoom:
            ax.axhline(0, color='0.6', ls=':', lw=1)
            ax.axhline((float(objective.z)-zs)*1e3, color='0.3', lw=2)
            ax.annotate('Specimen (object plane)', xy=(0, 0), xytext=(-7.5, -6),
                        fontsize=10, arrowprops=dict(arrowstyle='-', color='0.5'))
            ax.annotate('Objective lens', xy=(0, (float(objective.z)-zs)*1e3),
                        xytext=(-7.5, 17), fontsize=10,
                        arrowprops=dict(arrowstyle='-', color='0.5'))
    axes[0].set_title('Specimen → objective → first image')
    axes[1].set_title('Three rays from each point meet at its image')
    axes[1].legend(loc='upper right', fontsize=9, framealpha=.95)
    fig.suptitle(f'Objective image: {(image_z-zs)*1e3:.3f} mm after the specimen\n'
                 f'{(float(il1.z)-image_z)*1e3:.3f} mm before IL1 · magnification {magnification:.2f}×',
                 fontsize=15)
    save(fig, '03c_objective_image_plane')
    (HERE/'objective_image_plane.json').write_text(json.dumps(dict(
        specimen_z_m=zs, objective_z_m=float(objective.z), image_z_m=image_z,
        image_distance_from_specimen_m=image_z-zs,
        image_distance_from_objective_m=image_z-float(objective.z),
        distance_before_IL1_m=float(il1.z)-image_z, magnification=magnification,
        max_abs_B_m=float(np.abs(T[:2, 2:4]).max()),
        note='Ideal paraxial specimen conjugate: B=0. Downstream IL1 does not move this plane.'
    ), indent=2)+'\n')


def figures_from_saved():
    """Re-export figure layouts without repeating the inverse calculation."""
    d = np.load(HERE/'dataset.npz')
    c = json.loads((HERE/'calibration.json').read_text())
    m = microscope()
    computed = [optics(m,r['delta']) for r in c['settings']]
    columns = [item[0] for item in computed]
    c['settings'] = [item[1] for item in computed]
    truth, recon = d['specimen_truth'].ravel(), d['specimen_retrieved'].ravel()
    make_figures(m,columns,c['settings'],d['intensities'],d['vacuum_intensities'],truth,recon,
                 d['specimen_wrong'].ravel(),np.angle(recon*np.conj(truth)),
                 d['mask'].ravel(),d['history'],d['wrong_history'],c['metrics'])


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures-only', action='store_true')
    args = parser.parse_args()
    figures_from_saved() if args.figures_only else main()
