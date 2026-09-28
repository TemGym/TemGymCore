"""Explore and fit a microscope. Native geometry and saved synthetic data."""
from make_calibration_workflow_figure import (
    DATA, OUT, INK, MUTED, TEAL, OCHRE, BLUE, image_stack, arrow,
)
import ast
from pathlib import Path
import contextlib
import io
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse
from matplotlib.text import Annotation
from matplotlib.transforms import Affine2D
from temgym_core import reflection_calibration as calibration
from temgym_core.plotting import export_ray_figure, plot_reflection_column, trace_gaussian_basis
from temgym_core.components import Deflector
from temgym_core.coordinate_transforms import PlaneFrame
import jax.numpy as jnp

PURPLE = "#776294"
# B1 portrait is 707 mm wide; leave ~25 mm poster margins either side.
POSTER_WIDTH_MM = 660
POSTER_MIN_FONT_PT = 18


def use_arial():
    """Register installed Arial files; matplotlib's cache may predate them."""
    from matplotlib import font_manager
    for path in font_manager.findSystemFonts():
        if Path(path).stem.lower().startswith('arial'):
            font_manager.fontManager.addfont(path)
    font_manager.findfont('Arial', fallback_to_default=False)


def scale_for_poster(fig, factor, min_font):
    """Enlarge every point-sized quantity with the canvas, then floor text size.

    Layout is in figure fractions, so geometry follows the canvas; fonts,
    strokes, markers, tick marks and point offsets are scaled here to keep
    the manuscript proportions.
    """
    # Ticks are regenerated from axis parameters, so rescale those parameters.
    tick_settings = []
    for ax in fig.axes:
        for name, axis in (('x', ax.xaxis), ('y', ax.yaxis)):
            ticks = axis.get_major_ticks()
            if ticks:
                tick = ticks[0]
                tick_settings.append((ax, name, dict(
                    length=tick.tick1line.get_markersize() * factor,
                    width=tick.tick1line.get_markeredgewidth() * factor,
                    pad=tick.get_pad() * factor,
                    labelsize=max(tick.label1.get_fontsize() * factor, min_font))))
            axis.labelpad *= factor
        ax._set_title_offset_trans(ax.titleOffsetTrans._t[1] * 72. * factor)
    for artist in set(fig.findobj()):
        if artist.get_gid() in ('reflection_lens_yoke', 'reflection_lens_coil', 'reflection_deflector'):
            # Optical symbols are drawn in typographic points about their anchor.
            artist.set_transform(Affine2D().scale(factor) + artist.get_transform()
                                 if isinstance(artist, plt.Line2D)
                                 else Affine2D().scale(factor) + artist.get_data_transform())
        if isinstance(artist, Annotation):
            if artist.anncoords == 'offset points':
                artist.xyann = tuple(v * factor for v in artist.xyann)
            if artist.arrow_patch is not None:
                artist.arrow_patch.set_linewidth(artist.arrow_patch.get_linewidth() * factor)
                artist.arrow_patch.shrinkA *= factor
                artist.arrow_patch.shrinkB *= factor
        if isinstance(artist, plt.Text):
            artist.set_fontsize(max(artist.get_fontsize() * factor, min_font))
        elif isinstance(artist, plt.Line2D):
            artist.set_linewidth(artist.get_linewidth() * factor)
            artist.set_markersize(artist.get_markersize() * factor)
            artist.set_markeredgewidth(artist.get_markeredgewidth() * factor)
        elif isinstance(artist, FancyArrowPatch):
            artist.set_linewidth(artist.get_linewidth() * factor)
            artist.set_mutation_scale(artist.get_mutation_scale() * factor)
        elif isinstance(artist, plt.matplotlib.patches.Patch):
            artist.set_linewidth(artist.get_linewidth() * factor)
        elif isinstance(artist, plt.matplotlib.collections.Collection):
            artist.set_linewidths(np.asarray(artist.get_linewidths()) * factor)
            sizes = artist.get_sizes()
            if len(sizes):
                artist.set_sizes(np.asarray(sizes) * factor**2)
    for ax, name, settings in tick_settings:
        ax.tick_params(axis=name, **settings)


def draw_microscope(ax, components, trace, *, symbol_scale=.62, **kwargs):
    """Shared native renderer and optical-symbol styling for every panel."""
    plot_reflection_column(components, trace, title='', show_section_labels=False,
                           ax=ax, **kwargs)
    for text in list(ax.texts):
        text.remove()
    from matplotlib.transforms import Affine2D
    for patch in ax.patches:
        if patch.get_gid() in ('reflection_lens_yoke', 'reflection_lens_coil'):
            patch.set_transform(Affine2D().scale(symbol_scale) + patch.get_data_transform())
            patch.set_linewidth(.6)
    for line in ax.lines:
        if line.get_gid() == 'reflection_deflector':
            # Artist transforms scale the endpoints, but not linewidths.
            # Scale both so the native red/blue symbol retains its proportions.
            deflector_scale = symbol_scale * .65
            line.set_transform(Affine2D().scale(deflector_scale) + line.get_transform())
            line.set_linewidth(line.get_linewidth() * deflector_scale)
            line.set_solid_capstyle('butt')


def clip_to_specimen(vertices):
    """Clip a laboratory (horizontal, height) polygon to vacuum, height <= 0."""
    result = []
    previous = vertices[-1]
    for current in vertices:
        previous_inside, inside = previous[1] <= 0., current[1] <= 0.
        if previous_inside != inside:
            fraction = -previous[1] / (current[1] - previous[1])
            crossing = previous + fraction * (current - previous)
            crossing[1] = 0.
            result.append(crossing)
        if inside:
            result.append(current)
        previous = current
    return np.asarray(result)


def finish_envelope_at_surface(ax, trace, *, incidence_deg, objective_z):
    """Clip the Gaussian display to the specimen without shearing its geometry.

    Enlarge transverse widths upstream, smoothly returning to physical scale
    through the final drift. Propagate beyond the transverse specimen plane
    before clipping, so there is no end cap or artificial surface footprint.
    """
    alpha = np.deg2rad(incidence_deg)
    sine, cosine = np.sin(alpha), np.cos(alpha)
    z = np.asarray(trace.z)
    previous = np.flatnonzero(z < z[-1])[-1]
    def extend(values, planes):
        values = np.asarray(values)
        result = np.interp(planes, z, values)
        mask = planes > z[-1]
        slope = (values[-1]-values[previous]) / (z[-1]-z[previous])
        result[mask] = values[-1] + slope*(planes[mask]-z[-1])
        return result
    # Continue the real final free-space drift until the entire unmagnified
    # envelope crosses the plane; this avoids any artificial axial end cap.
    end = .001
    for _ in range(16):
        planes = np.array([end])
        centre = extend(trace.centroid, planes)[0]
        radius = np.hypot(extend(trace.position, planes), extend(trace.angle, planes))[0]
        if end*sine + (centre-radius)*cosine > 0.:
            break
        end *= 2.
    else:
        raise ValueError('Incident envelope does not fully intersect the specimen.')
    planes = np.unique(np.r_[z, np.linspace(z[0], end, 4000)])
    centre = extend(trace.centroid, planes)
    radius = np.hypot(extend(trace.position, planes), extend(trace.angle, planes))
    bands = [band for band in ax.collections if band.get_zorder() == 0]
    progress = np.clip((planes-objective_z)/(-objective_z), 0., 1.)
    # A scalar transverse display magnification; axial positions and physical
    # pose stay untouched. The smoothstep has zero derivative at both ends.
    blend = progress**2*(3.-2.*progress)
    display_radius = radius*(1. + 39.*(1.-blend))
    for band, fraction in zip(bands, np.linspace(1., 0., len(bands), endpoint=False)):
        local_x = np.r_[centre-fraction*display_radius, (centre+fraction*display_radius)[::-1]]
        local_z = np.r_[planes, planes[::-1]]
        world = np.column_stack((local_z*cosine-local_x*sine,
                                 local_z*sine+local_x*cosine))
        polygon = clip_to_specimen(world)
        assert np.isfinite(polygon).all() and np.all(polygon[:, 1] <= 0.)
        band.set_verts([polygon*1000.])
        band.set_transform(ax.transData)
        band.set_gid('surface_clipped_incident_envelope')


def mini_microscope(fig, box, parameters, spec, deflector_positions, *, muted=False):
    ax = fig.add_axes(box)
    # Fixed, unpowered notebook deflectors are identity elements in this
    # on-axis calibration. Their positions and kicks are not inferred.
    components = tuple(sorted(
        (*calibration.column(parameters, np.ones(3)),
         *(Deflector(z=z, def_x=0., def_y=0.) for z in deflector_positions)),
        key=lambda component: float(component.z)))
    trace = trace_gaussian_basis(components, 2 * parameters[7] * 1e-9,
                                voltage=spec['voltage_v'], z=parameters[0] * 1e-3)
    incoming, outgoing = calibration.frames(spec)
    directions, _, _ = calibration.order_geometry(spec)
    centre = parameters[8] * 1e-3 * np.asarray(outgoing.normal_xyz)
    screen = PlaneFrame(origin_xyz=jnp.asarray(centre), u_xyz=outgoing.u_xyz, v_xyz=outgoing.v_xyz)
    hits = np.asarray(screen.intersect_line(jnp.zeros(3), jnp.asarray(directions))[0])
    paths = np.stack((np.zeros_like(hits), hits), axis=1)[..., [0, 2]]
    half_height = spec['detector_shape'][0] * spec['pixel_pitch_m'] / 2
    ends = centre + np.array([-1., 1.])[:, None] * half_height * np.asarray(outgoing.u_xyz)
    draw_microscope(ax, components, trace, symbol_scale=.38, sample_z=0.,
                    outgoing_paths_xz=paths, detector_segment_xz=ends[:, [0, 2]],
                    column_frame=incoming, specimen_label='Cu(001)', envelope_scale=40.)
    finish_envelope_at_surface(ax, trace, incidence_deg=spec['incidence_deg'],
                               objective_z=parameters[3] * 1e-3)
    ax.set(xlim=(-465, 195), ylim=(-17, 5))
    ax.set_axis_off()
    for line in ax.lines:
        if line.get_gid() != 'reflection_deflector' and line.get_linewidth() >= 4:
            line.set_linewidth(2.5)
        if line.get_marker() == 'o':
            line.set_markersize(2)
    if muted:
        for band in ax.collections:
            if band.get_zorder() == 0:
                band.set_facecolor('#87949B')
        for patch in ax.patches:
            if patch.get_gid() == 'reflection_lens_coil':
                patch.set_facecolor('#AAB4BA')
                patch.set_edgecolor('#77848C')
        for line in ax.lines:
            if line.get_zorder() in (2, 3):
                line.set_color('#87949B')
    return ax


def heading(fig, index, title, subtitle, x, color):
    fig.text(x, .596, index, fontsize=9, color=color, weight='bold')
    fig.text(x + .028, .596, title, fontsize=9, color=INK)


def box(fig, x, y, width, height, text, color, *, size=7):
    fig.add_artist(FancyBboxPatch((x, y), width, height,
        transform=fig.transFigure, boxstyle='round,pad=.004,rounding_size=.006',
        facecolor='white', edgecolor=color, lw=.65))
    fig.text(x + width / 2, y + height / 2, text, ha='center', va='center', fontsize=size, color=color)


def notebook_gui(fig, left=.040, bottom=.631, width=.620, *, compact=True):
    """Redraw the saved notebook state as a wide native vector microscope view."""
    notebook_path = DATA.parent.parent / 'sem_reflection.ipynb'
    notebook = json.loads(notebook_path.read_text())
    cell = next(''.join(c.get('source', [])) for c in notebook['cells']
                if 'camera_controls=widgets.VBox' in ''.join(c.get('source', [])))
    definitions = {}
    for node in ast.parse(cell).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'FloatSlider':
                definitions[node.targets[0].id] = {
                    kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords
                    if kw.arg in ('min', 'max', 'step', 'description', 'readout_format')}
    saved = json.loads((DATA.parent / 'copper/camera_view.json').read_text())
    values = [saved['parameters']['incidence_deg'] + saved['parameters'].get('rocking_deg', 0.),
              saved['camera_pose']['z_mm'], saved['camera_pose']['height_mm']]
    # Execute only the column setup, never image rendering or saving cells.
    # The live notebook and its outputs remain unchanged.
    env = {}
    with plt.rc_context(), contextlib.redirect_stdout(io.StringIO()):
        exec(compile(''.join(notebook['cells'][1]['source']), str(notebook_path), 'exec'), env)
        env['P'].update(saved['parameters'])
        exec(compile(''.join(notebook['cells'][3]['source']), str(notebook_path), 'exec'), env)
    jnp = env['jnp']
    kicks, _ = env['solve_pivot'](values[0], env['P']['azimuth_deg'], coeffs=env['objective_coeffs']())
    components = env['column'](kicks, env['objective_coeffs']())
    beam = env['source']; rows = []
    for component in components:
        distance = float(component.z - beam.z)
        segment = env['fs'](beam, jnp.linspace(0., distance, 100))
        covariance = jnp.linalg.inv(segment.Q_inv.imag) / (2 * segment.k[..., None, None])
        rows.extend(np.column_stack((segment.z, segment.x, 2 * jnp.sqrt(covariance[:, 0, 0]))))
        beam = component(env['fs'](beam, distance))
    incoming = env['column_frame'].basis_xyz @ jnp.array([beam.dx, beam.dy, 1.])
    incoming /= jnp.linalg.norm(incoming)
    period = env['P']['lattice_constant'] / np.sqrt(2)
    q = np.arange(-6, 7) * float(beam.wavelength) / period
    tu = float(incoming @ env['surface'].u_xyz)
    tv = float(incoming @ env['surface'].v_xyz) + q
    rad = 1 - tu ** 2 - tv ** 2
    order_indices = np.arange(-6, 7)[rad > np.sin(np.deg2rad(.25)) ** 2]
    tv = tv[rad > np.sin(np.deg2rad(.25)) ** 2]
    rad = rad[rad > np.sin(np.deg2rad(.25)) ** 2]
    directions = (tu * env['surface'].u_xyz + tv[:, None] * env['surface'].v_xyz
                  + np.sqrt(rad)[:, None] * env['surface'].normal_xyz)
    frame = env['replace'](env['out_frame'], origin_xyz=jnp.asarray(saved['camera_pose']['centre_world_xyz_m']))
    hits, _ = frame.intersect_line(env['pivot'], directions)
    paths = np.stack((np.zeros_like(hits), hits), axis=1)[..., [0, 2]]
    half_height = env['P']['detector_shape'][0] * env['P']['pixel_pitch'] / 2
    edge = np.asarray(frame.to_world(jnp.array([[-half_height, 0., 0.], [half_height, 0., 0.]])))[:, [0, 2]]
    trace = env['trace_gaussian_basis'](env['column'](kicks), env['source_w'], voltage=env['P']['voltage'])
    ax = fig.add_axes([left + .03, bottom + .035, width - .045, .234])
    draw_microscope(ax, components, trace, sample_z=env['P']['sample_z'],
        outgoing_paths_xz=paths, detector_segment_xz=edge, column_frame=env['column_frame'],
        component_labels=('C1', 'C2', 'D1', 'D2', 'Objective', 'Reference'),
        specimen_label='Cu(001)', incident_envelope=np.asarray(rows))
    component_names = ['Condenser lens 1', 'Condenser lens 2',
                       'Deflector 1', 'Deflector 2', 'Objective lens']
    offsets = [(0, -25), (-12, -27), (-23, 23), (12, 38), (13, -26)]
    for component, name, offset in zip(components[:-1], component_names, offsets):
        point = np.asarray(env['column_frame'].to_world(jnp.array([0., 0., component.z])))
        ax.annotate(name, (point[2] * 1000, point[0] * 1000), xytext=offset,
                    textcoords='offset points', ha='center', fontsize=6, color=INK,
                    arrowprops=dict(arrowstyle='-', color='#ACB6BD', lw=.4, shrinkB=10))
    point = np.asarray(env['column_frame'].to_world(jnp.array([0., 0., 0.])))
    ax.annotate('Electron\nsource', (point[2] * 1000, point[0] * 1000), xytext=(0, -5),
                textcoords='offset points', ha='center', va='top', fontsize=6, color=INK)
    ax.text(0, 3.2, 'Cu(001) specimen', ha='center', fontsize=6, color=INK)
    ax.text(values[1], 1.3, 'Detector', ha='center', fontsize=6, color=INK)
    ax.set(xlim=(-375, 150), ylim=(-17, 7), xlabel='Horizontal position (mm)', ylabel='Height (mm)')
    ax.xaxis.label.set_size(6); ax.yaxis.label.set_size(6)
    ax.tick_params(labelsize=5.5, length=2, colors=MUTED)
    for index, (name, value) in enumerate(zip(('angle_slider', 'z_slider', 'height_slider'), values)):
        definition = definitions[name]
        x0 = left + .008 + index * width / 3
        x1 = x0 + width / 3 - .020
        fig.text(x0, .943, definition['description'], fontsize=6.1, color=INK)
        fig.text(x1, .919, format(value, definition['readout_format']), fontsize=6, ha='right', color=INK)
        track_end = x1 - .046
        knob = x0 + (value - definition['min']) / (definition['max'] - definition['min']) * (track_end - x0)
        fig.add_artist(plt.Line2D([x0, track_end], [.921, .921], transform=fig.transFigure,
                                 color='#D4DCE1', lw=2.2, solid_capstyle='round'))
        fig.add_artist(plt.Line2D([knob], [.921], transform=fig.transFigure, marker='o', ms=4.3,
                                 color='#6C7983', markeredgecolor='white', markeredgewidth=.5))
    return dict(notebook=str(notebook_path), controls=definitions, values=values,
                detector_order_labels=[dict(order=int(order), lateral_mm=float(uv[1]*1000),
                                            height_mm=float(uv[0]*1000))
                                       for order, uv in zip(order_indices, np.asarray(frame.to_local(hits)))],
                fixed_deflector_positions_m=[float(c.z - env['P']['sample_z'])
                                             for c in components if isinstance(c, Deflector)],
                note='Native vector redraw of saved notebook column at wider aspect; compact controls')


def build(*, poster=False):
    spec = json.loads((DATA / 'experiment.json').read_text())
    fit = json.loads((DATA / 'fit.json').read_text())
    checks = json.loads((DATA / 'checks.json').read_text())
    p = np.asarray(fit['best']['parameters'])
    # Illustrative bounded geometry with a longer source drift and closer
    # condensers. All initial-view predictions use this same optical model.
    initial = np.array([-445., -215., -170., -58.,
                        29.752734675149718, 12., 11.923795237917972,
                        2.3106881033622066, 136.36740823900456])
    with np.load(DATA / 'images.npz') as data:
        images = data['images']
    use_arial()
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.linewidth': .5, 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(180 / 25.4, 165 / 25.4), facecolor='white')

    # Two full-width panels, vertically stacked.
    for letter, title, y, color in [('a', 'Interactive diffraction simulation of reflection SEM microscope', .975, OCHRE),
                                   ('b', 'Inverting an unknown reflection SEM microscope in simulation', .594, BLUE)]:
        fig.text(.035, y, letter, fontsize=10, weight='bold', color=color)
        fig.text(.065, y, title, fontsize=10, color=INK)
    gui_metadata = notebook_gui(fig)
    # Draw directly from saved intensity, so text is vector and independently
    # sized rather than embedded in a downscaled notebook screenshot.
    detector_source = DATA.parent / 'copper/camera_view.npz'
    with np.load(detector_source) as saved_detector:
        # Saved detector uses (lateral, height); imshow needs (height, lateral).
        intensity = saved_detector['intensity'].T
    detector_parameters = json.loads((DATA.parent / 'copper/camera_view.json').read_text())['parameters']
    pitch_mm = detector_parameters['pixel_pitch'] * 1000.
    rows, cols = intensity.shape
    extent = np.array([-cols, cols, -rows, rows]) * pitch_mm / 2.
    detector = fig.add_axes([.724, .647, .250, .285])
    detector.imshow(np.maximum(intensity / intensity.max(), 1e-5), origin='lower',
                    extent=extent, cmap='magma', norm=LogNorm(1e-5, 1),
                    interpolation='nearest', aspect='equal')
    detector.set_title('Detector image', loc='left', fontsize=9, pad=7, color=INK)
    detector.set_xlabel('Lateral position (mm)', fontsize=8, color=INK)
    detector.set_ylabel('Height (mm)', fontsize=8, color=INK)
    detector.tick_params(labelsize=7, length=3, colors=MUTED)
    for spine in detector.spines.values():
        spine.set_linewidth(.5); spine.set_color('#A5B2BB')
    for label in gui_metadata['detector_order_labels']:
        x, y = label['lateral_mm'], label['height_mm']
        if extent[0] < x < extent[1] and extent[2] < y < extent[3]:
            # Edge orders are aligned inward so their labels are not clipped.
            align = 'left' if x < .8*extent[0] else 'right' if x > .8*extent[1] else 'center'
            detector.annotate(str(label['order']), (x, y), xytext=(0, 9),
                              textcoords='offset points', ha=align, fontsize=7,
                              color='#B8CCD9', clip_on=True)
    arrow(fig, (.660, .787), (.699, .787), color=OCHRE, scale=7)

    fit_axes_start = len(fig.axes)
    fit_text_start = len(fig.texts)
    fit_artist_start = len(fig.artists)

    # Same recorded frame and detector coordinates on both sides; the model
    # contours alone change, so the visual agreement is a genuine fit result.
    fig.text(.229, .554, 'Initial guess', ha='center', fontsize=9, color=MUTED)
    fig.text(.758, .554, 'Recovered microscope', ha='center', fontsize=9, color=TEAL)
    mini_microscope(fig, [.046, .462, .367, .087], initial, spec,
                    gui_metadata['fixed_deflector_positions_m'], muted=True)
    mini_microscope(fig, [.578, .462, .367, .087], p, spec,
                    gui_metadata['fixed_deflector_positions_m'])
    row = 52
    with np.load(DATA / 'settings.npz') as data:
        controls, detector_dz = data['controls'][row:row+1], data['detector_dz_m'][row:row+1]
    _, predict = calibration.make_predictor(spec)
    pitch = spec['pixel_pitch_m'] * 1000
    nr, nc = images[row].shape
    for left, parameters, color, caption in [
            (.065, initial, '#E8B65F', 'Initial prediction'),
            (.595, p, '#44D3BB', 'Fitted prediction')]:
        mean, covariance = map(np.asarray, predict(parameters, controls, detector_dz))
        ax = fig.add_axes([left, .333, .328, .118])
        ax.imshow(np.maximum(images[row], 1), origin='upper', cmap='gray',
                  norm=LogNorm(1, images[row].max()),
                  extent=np.array([-nc, nc, -nr, nr]) * pitch / 2)
        for centre, matrix in zip(mean[0], covariance[0]):
            eigenvalues, vectors = np.linalg.eigh(matrix[np.ix_([1, 0], [1, 0])])
            angle = np.rad2deg(np.arctan2(vectors[1, 1], vectors[0, 1]))
            ax.add_patch(Ellipse(centre[::-1] * pitch,
                4*np.sqrt(eigenvalues[1])*pitch, 4*np.sqrt(eigenvalues[0])*pitch,
                angle=angle, fill=False, edgecolor=color, lw=.9))
        ax.set(xlim=(-6., 6.), ylim=(-2.75, 3.6), aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#CFD9DF'); spine.set_linewidth(.55)
        ax.text(.035, .08, caption, transform=ax.transAxes, color=color,
                fontsize=6.3, va='bottom')
    stack = image_stack(fig, images, [1, 7, 16, 28, 52])
    stack.set_position([.425, .458, .142, .094])
    fig.text(.50, .444, 'Recorded patterns', ha='center', fontsize=6.8, color=MUTED)
    fig.text(.50, .428, 'with perturbed lenses', ha='center', fontsize=6.0, color=MUTED)
    arrow(fig, (.50, .420), (.50, .407), color=BLUE, scale=6)
    arrow(fig, (.410, .400), (.584, .400), color=BLUE, scale=10)
    fig.text(.50, .378, 'Fit geometry +', fontsize=7, color=BLUE, ha='center')
    fig.text(.50, .361, 'lens strengths', fontsize=7, color=BLUE, ha='center')

    # Reserve the right-hand column for quantitative recovery evidence.
    from matplotlib.transforms import Affine2D
    fit_width = .62
    for ax in fig.axes[fit_axes_start:]:
        bounds = ax.get_position(original=True)
        ax.set_position([bounds.x0*fit_width, bounds.y0,
                         bounds.width*fit_width, bounds.height])
    for artist in [*fig.texts[fit_text_start:], *fig.artists[fit_artist_start:]]:
        artist.set_transform(Affine2D().scale(fit_width, 1.) + fig.transFigure)
    truth = np.asarray(json.loads((DATA / 'scoring_truth.json').read_text())['parameters'])
    uncertainty = np.asarray(fit['standard_errors'])
    fig.text(.647, .555, 'Parameter', fontsize=7, weight='bold', color=INK)
    fig.text(.846, .555, 'True', fontsize=7, weight='bold', ha='right', color=INK)
    fig.text(.981, .555, 'Fit ± 1 SE', fontsize=7, weight='bold', ha='right', color=TEAL)
    fig.add_artist(plt.Line2D([.643, .985], [.546, .546],
        transform=fig.transFigure, color='#B9C8CE', lw=.6))
    labels = ['Source z (mm)', 'Condenser lens 1 z (mm)',
              'Condenser lens 2 z (mm)', 'Objective lens z (mm)',
              'Condenser lens 1 f (mm)', 'Condenser lens 2 f (mm)',
              'Objective lens f (mm)', 'Source RMS size (nm)', 'Camera length (mm)']
    decimals = [1, 1, 1, 2, 1, 2, 2, 3, 5]
    truth_decimals = [0, 0, 0, 0, 1, 2, 2, 2, 0]
    for i, (label, places, true_places) in enumerate(zip(labels, decimals, truth_decimals)):
        y = .524 - i*.0235
        if i % 2 == 0:
            fig.add_artist(FancyBboxPatch((.643, y-.007), .342, .022,
                boxstyle='square,pad=0', transform=fig.transFigure,
                facecolor='#F1F6F5', edgecolor='none', zorder=-1))
        fig.text(.647, y, label, fontsize=6.5, color=INK)
        fig.text(.846, y, f'{truth[i]:.{true_places}f}', fontsize=6.5,
                 color=MUTED, ha='right')
        fig.text(.981, y, f'{p[i]:.{places}f} ± {uncertainty[i]:.{places}f}',
                 fontsize=6.5, color=TEAL, ha='right')


    # Remove the former third row while preserving the physical size and
    # relative layout of the two retained panels and their point-sized symbols.
    from matplotlib.transforms import Affine2D
    bottom, retained_height = .315, .685
    for ax in fig.axes:
        bounds = ax.get_position(original=True)
        ax.set_position([bounds.x0, (bounds.y0-bottom)/retained_height,
                         bounds.width, bounds.height/retained_height])
    layout = Affine2D().translate(0., -bottom).scale(1., 1./retained_height) + fig.transFigure
    for artist in [*fig.texts, *fig.artists]:
        artist.set_transform(artist.get_transform() - fig.transFigure + layout)
    fig.set_size_inches(230/25.4, 165*retained_height/25.4)

    width_mm, height_mm = 230, 165*retained_height
    if poster:
        factor = POSTER_WIDTH_MM / width_mm
        scale_for_poster(fig, factor, POSTER_MIN_FONT_PT)
        width_mm, height_mm = width_mm*factor, height_mm*factor
        fig.set_size_inches(width_mm/25.4, height_mm/25.4)

    # The native column renderer sets its own tick-label family.
    for ax in fig.axes:
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontfamily('Arial')

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / ('microscope_model_to_ptychography' + ('_poster_b1' if poster else ''))
    files = export_ray_figure(fig, stem, dpi=300 if poster else 600, metadata=dict(
        title='Interactive diffraction simulation of reflection SEM microscope and microscope fitting', source=str(DATA),
        exploration=gui_metadata, exploration_detector_source=str(detector_source),
        exploration_detector_rendering='Native saved intensity; log scale 1e-5 to 1 relative to its maximum; vector labels',
        fit_initial_parameters=initial.tolist(),
        fit_initial_source='Illustrative bounded geometry with shifted source/lenses; focal lengths selected for moderate detector disagreement; not a separately rerun calibration',
        fit_comparison_record_zero_based=row,
        fit_comparison='Same measured image and crop; initial and fitted model 2-sigma contours in detector coordinates',
        fit_envelope_magnification=40,
        fit_envelope_magnification_at_specimen=1,
        fit_envelope_display='Native transverse display magnification: 40x upstream, smooth reduction to physical 1x at specimen; final drift continued and clipped to surface; no axial or horizontal shear',
        fit_deflectors='Fixed notebook positions; zero steering; not inferred calibration parameters',
        demonstrated='Saved synthetic fit and saved interactive microscope/detector state',
        parameter_table=dict(truth=truth.tolist(), fitted=p.tolist(), standard_errors=uncertainty.tolist(),
                             uncertainty='Local 1-standard-error estimates from saved fit'),
        width_mm=width_mm, height_mm=height_mm,
        poster=dict(sheet='B1 portrait, full width', min_font_pt=POSTER_MIN_FONT_PT) if poster else None,
        layout="Two vertically stacked panels: explore configurations and fit an unknown microscope"))
    fig.savefig(stem.with_name(stem.name + '_preview.png'), dpi=180 / (factor if poster else 1),
                facecolor='white')
    plt.close(fig)
    return files


if __name__ == '__main__':
    import sys
    for kind, path in build(poster='--poster' in sys.argv).items():
        print(f'{kind}: {path}')
