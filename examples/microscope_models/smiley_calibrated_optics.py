"""Operating controls for the saved fitted microscope, using native solenoids."""
from functools import lru_cache
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.calibration.condenser_model import condenser_components, design_condenser_curve
from temgym_core.calibration.projector_model import projector_components
from temgym_core.components import Detector, Plane
from temgym_core.microscope import Microscope, find_imaging_condition
from temgym_core.source import gaussian_waist
from temgym_core.run import column_matrix

HERE = Path(__file__).resolve().parent
SPOTS = np.arange(1., 9.)
MAGNIFICATIONS = np.array([40000., 60000., 80000., 100000., 150000., 200000.,
                           250000., 300000., 400000., 600000.])


@lru_cache(maxsize=1)
def calibration():
    data = json.loads((HERE/'smiley_calibration.json').read_text())
    cp = jnp.asarray(data['condenser_parameters'])
    # Same source-image design and mini setting as the calibration acquisition.
    # Spot 3 retains its nominal 0.2 source-image magnification.
    targets = .2*2**((SPOTS-3)/2)
    curve = design_condenser_curve(cp, image_magnifications=targets,
        mini_control=data['condenser_reference_controls'][2],
        initial_controls=data['condenser_reference_controls'][:2])
    return data, np.asarray(curve['controls'])


@lru_cache(maxsize=32)
def microscope(spot=3., magnification=40000.):
    data, controls = calibration()
    cp, pp = map(jnp.asarray, (data['condenser_parameters'], data['projector_parameters']))
    indices = np.flatnonzero(SPOTS == spot)
    if not len(indices):
        raise ValueError('Choose a calibrated spot setting 1–8.')
    condenser = condenser_components(cp, controls[indices[0]])
    projector = projector_components(pp, jnp.asarray(data['projector_reference_controls']))
    names = ('CL1', 'CL3', 'C_mini', 'Obj_prefield', 'sample',
             'Obj_post', 'IL1', 'IL2', 'IL3', 'PL1', 'detector')
    components = dict(zip(names, (*condenser, *projector)))
    components['sample'] = Plane(z=0.)
    components['detector'] = Detector(z=float(projector[-1].z), shape=(128, 128),
                                      pixel_size=(55e-6, 55e-6))
    # The fitted aperture lies after CL3, in the same native free gap.
    aperture_z = float(condenser[1].z+condenser[1].length+jax.nn.sigmoid(cp[19])*jnp.exp(cp[2]))
    config = {'apertures': {'C_aperture': {'z_m': aperture_z,
              'radii_m': [10e-6, 25e-6, 80e-6, 150e-6]}}, 'calibration': data}
    model = Microscope(components, voltage=data['voltage_v'], config=config)
    # Retain the calibrated baseline rotation as magnification changes.
    baseline = np.asarray(column_matrix(projector))
    rotation = float(np.arctan2(baseline[1, 0], baseline[0, 0]))
    lower = MAGNIFICATIONS[MAGNIFICATIONS < magnification]
    if len(lower):
        previous = microscope(3., float(lower[-1]))
        model = model.with_components(**{name: previous.components[name]
            for name in ('IL1', 'IL2', 'IL3', 'PL1')})
    model = find_imaging_condition(model, magnification=magnification,
        vary=['IL1.q', 'IL2.q', 'IL3.q', 'PL1.q'], rotation=rotation,
        bounds=([.01, .01, .01, -2.8], [2.8, 2.8, 2.8, -.01])).microscope
    return model


def source():
    data, _ = calibration()
    p = jnp.asarray(data['condenser_parameters'])
    return gaussian_waist(z=-jnp.exp(p[:9]).sum(), voltage=data['voltage_v'],
        rms_size=jnp.exp(p[17]), x=p[13]*1e-8, y=p[14]*1e-8,
        theta_x=p[15]*1e-4, theta_y=p[16]*1e-4)
