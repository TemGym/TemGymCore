"""Geometry and coherent basis checks for the reflection example."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.coordinate_transforms import PlaneFrame
from temgym_core.source import gaussian_waist, decompose_gaussian_angular
from temgym_core.components import Lens, Plane
from temgym_core.gaussian import FreeSpacePropagator
from temgym_core.run import run_to_end
from temgym_core.evaluate import evaluate_gaussians


def test_tilted_plane_intersection_roundtrip_and_derivative():
    angle = .07
    frame = PlaneFrame(origin_xyz=jnp.array([.01, -.02, .3]),
        u_xyz=jnp.array([-np.sin(angle), 0., np.cos(angle)]),
        v_xyz=jnp.array([0., 1., 0.]))
    points = jnp.array([[.01, .02, .03], [-.003, .001, 0.]])
    np.testing.assert_allclose(frame.to_local(frame.to_world(points)), points, atol=1e-16)
    def intersection(tilt):
        direction = jnp.array([tilt, .002, 1.])
        return frame.intersect_line(jnp.zeros(3), direction)[0]
    hit = intersection(.001)
    assert abs(float(frame.to_local(hit)[2])) < 1e-15
    derivative = jax.jacfwd(intersection)(.001)
    finite = (intersection(.001+1e-7)-intersection(.001-1e-7))/2e-7
    np.testing.assert_allclose(derivative, finite, rtol=1e-8)
    parallel, distance = frame.intersect_line(jnp.zeros(3), frame.u_xyz)
    assert np.isnan(parallel).all() and np.isnan(distance)


def test_angular_decomposition_preserves_complex_field_through_lenses():
    source = gaussian_waist(z=0., voltage=30e3, rms_size=1.5e-9/np.sqrt(2*np.log(2))/2)
    column = (Lens(z=.08, focal_length=.04), Lens(z=.22, focal_length=.03),
              Lens(z=.31, focal_length=.012), Plane(z=.33))
    fs = FreeSpacePropagator(include_axial_path=False)
    bundle = decompose_gaussian_angular(source, num_across=64, through=column)
    for components in ((), column):
        exact = run_to_end(source, components, propagator=fs) if components else source
        approx = run_to_end(bundle, components, propagator=fs) if components else bundle
        width = np.sqrt(2/(float(exact.k)*float(exact.Q_inv[0,0].imag)))
        x = np.linspace(-2, 2, 51)*width
        grid = SimpleNamespace(shape=x.shape, coords=jnp.array(np.column_stack((x,x*.3))))
        field = np.asarray(evaluate_gaussians(exact, grid, method='scan'))
        actual = np.asarray(evaluate_gaussians(approx, grid, method='scan'))
        assert np.linalg.norm(actual-field)/np.linalg.norm(field) < 1e-5
    coarse = decompose_gaussian_angular(source, num_across=16)
    fine = decompose_gaussian_angular(source, num_across=32)
    np.testing.assert_allclose(coarse.Q_inv[0], fine.Q_inv[0])
    with pytest.raises(ValueError):
        decompose_gaussian_angular(source, num_across=4)
    with pytest.raises(ValueError):
        decompose_gaussian_angular(source, num_across=16,broadening=1.)


def test_reflection_plot_uses_supplied_physical_paths():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from temgym_core.plotting import trace_gaussian_basis, plot_reflection_column
    column = (Lens(z=.1,focal_length=.05),Plane(z=.2))
    trace = trace_gaussian_basis(column,1e-6,voltage=30e3)
    paths=np.array([[[0.,.2],[-.02,.5]],[[0.,.2],[-.01,.5]]])
    fig, ax = plot_reflection_column(column,trace,sample_z=.2,
        outgoing_paths_xz=paths,detector_segment_xz=np.array([[-.025,.5],[-.005,.5]]),
        surface_angle_rad=-.03)
    assert any(np.allclose(line.get_xdata(),[200,500]) and
               np.allclose(line.get_ydata(),[0,-20])
               for line in ax.lines if len(line.get_xdata())==2)
    plt.close(fig)


def test_flat_sample_with_inclined_column_and_specular_reflection():
    """Physical laboratory pose leaves the specimen flat at every incidence."""
    for angle in np.deg2rad([1.5, 2., 3.]):
        axis = jnp.array([np.sin(angle),0.,np.cos(angle)])
        column = PlaneFrame(origin_xyz=-.33*axis,
            u_xyz=jnp.array([np.cos(angle),0.,-np.sin(angle)]),
            v_xyz=jnp.array([0.,1.,0.]))
        surface = PlaneFrame(origin_xyz=jnp.zeros(3),u_xyz=jnp.array([0.,0.,1.]),
            v_xyz=jnp.array([0.,1.,0.]))
        hit, distance = surface.intersect_line(column.origin_xyz,column.normal_xyz)
        np.testing.assert_allclose(hit,0.,atol=1e-15)
        np.testing.assert_allclose(distance,.33,atol=1e-15)
        reflected = axis-2*jnp.dot(axis,surface.normal_xyz)*surface.normal_xyz
        np.testing.assert_allclose(reflected,[-np.sin(angle),0.,np.cos(angle)],atol=1e-15)
        np.testing.assert_allclose(surface.to_world(jnp.array([[.1,0.,0.],[-.1,0.,0.]]))[:,0],0.)


def test_reflection_symbols_stay_square_and_perpendicular_after_resizing():
    """Check visible geometry, including changed axes scaling and export DPI."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from temgym_core.components import Deflector
    from temgym_core.plotting import trace_gaussian_basis, plot_reflection_column
    angle = np.deg2rad(2.)
    axis = np.array([np.sin(angle), 0., np.cos(angle)])
    frame = PlaneFrame(origin_xyz=-.2*axis,
        u_xyz=np.array([np.cos(angle),0.,-np.sin(angle)]),v_xyz=np.array([0.,1.,0.]))
    components = (Lens(z=.1,focal_length=.05),Deflector(z=.15,def_x=0.,def_y=0.),Plane(z=.2))
    trace = trace_gaussian_basis(components,1e-6,voltage=30e3)
    fig,ax = plot_reflection_column(components,trace,sample_z=.2,column_frame=frame,
        outgoing_paths_xz=np.array([[[0.,0.],[-.01,.3]]]),
        detector_segment_xz=np.array([[-.017,.3],[-.003,.3]]))
    coils = [p for p in ax.patches if p.get_gid() == 'reflection_lens_coil']
    deflectors = [line for line in ax.lines if line.get_gid() == 'reflection_deflector']
    assert len(coils)==2 and len(deflectors)==3
    for size,dpi in [((14,5),100),((9,8),200)]:
        fig.set_size_inches(*size); fig.set_dpi(dpi)
        ax.set_ylim(-30,20)
        fig.canvas.draw()
        axis_display = ax.transData.transform([[0.,0.],[axis[2]*1e3,axis[0]*1e3]])
        direction = np.diff(axis_display,axis=0)[0]
        direction /= np.linalg.norm(direction)
        centres=[]
        for coil in coils:
            corners = coil.get_transform().transform([[0,0],[1,0],[0,1],[.5,.5]])
            u,v = corners[1]-corners[0],corners[2]-corners[0]
            np.testing.assert_allclose(np.linalg.norm(u),8*dpi/72,rtol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(u),np.linalg.norm(v),rtol=1e-12)
            assert abs(np.dot(u,v))/(np.linalg.norm(u)*np.linalg.norm(v)) < 1e-12
            assert abs(np.dot(u/np.linalg.norm(u),direction)) < 1e-12
            centres.append(corners[3])
        paired = centres[1]-centres[0]
        assert abs(np.dot(paired/np.linalg.norm(paired),direction)) < 1e-12
        for line in deflectors:
            endpoints=line.get_transform().transform(np.column_stack((line.get_xdata(),line.get_ydata())))
            vector=endpoints[-1]-endpoints[0]
            assert abs(np.dot(vector/np.linalg.norm(vector),direction)) < 1e-12
    plt.close(fig)


def test_notebook_camera_pixels_match_world_pose():
    """A lab-Z/height move must not translate the image along lateral Y."""
    import ast
    import json
    from dataclasses import replace
    from pathlib import Path
    from temgym_core.components import Detector

    notebook = json.loads((Path(__file__).resolve().parents[1] /
        'examples/sem_reflection/sem_reflection.ipynb').read_text())
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if 'def camera_geometry(' in ''.join(cell['source']))
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'camera_geometry')
    angle = np.deg2rad(2.)
    frame = PlaneFrame(origin_xyz=jnp.zeros(3),
        u_xyz=jnp.array([np.cos(angle),0.,np.sin(angle)]),v_xyz=jnp.array([0.,1.,0.]))
    env = dict(np=np,jnp=jnp,replace=replace,pivot=jnp.zeros(3),out_frame=frame,
        detector=Detector(z=.4,shape=(16,20),pixel_size=(55e-6,55e-6)),
        representative_directions=jnp.array([frame.normal_xyz]))
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<camera_geometry>','exec'),env)
    z0,h0 = .4*np.cos(angle)*1e3,-.4*np.sin(angle)*1e3
    for dz,dh in [(0.,0.),(50.,0.),(0.,2.),(50.,2.)]:
        geometry = env['camera_geometry'](z0+dz,h0+dh)
        detector = geometry['detector']
        xy = np.asarray(detector.coords)
        pixel_world = frame.to_world(jnp.column_stack((xy,np.full(len(xy),detector.z))))
        np.testing.assert_allclose(np.asarray(pixel_world).mean(axis=0),
            [(h0+dh)*1e-3,0.,(z0+dz)*1e-3],atol=1e-14)
        # The specular ray always has transverse u=v=0. Its displacement
        # from the actual chip centre is vertical, with no lateral shift.
        spot_uv = -xy.mean(axis=0)
        np.testing.assert_allclose(spot_uv,
            [-(dh*np.cos(angle)+dz*np.sin(angle))*1e-3,0.],atol=1e-14)
