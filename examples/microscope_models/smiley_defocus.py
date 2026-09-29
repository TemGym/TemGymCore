"""Forward model for the interactive Gaussian smiley microscope notebook."""
from copy import copy
from dataclasses import replace
from operator import index
from pathlib import Path
from types import CodeType, SimpleNamespace
import inspect
import os
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter

from temgym_core.aberrations import KrivanekCoeffs
from temgym_core.components import (
    ABCDTransfer, DistortedLens, ElectromagneticLens, InterpolatedSample2D,
    KrivanekLens, Lens, Plane, SigmoidAperture, HardEdgeSolenoid, SeidelLens,
)
from temgym_core.evaluate import evaluate_gaussians, gpu_kernel_available
from temgym_core.gaussian import FreeSpacePropagator, GaussianBeam, make_gaussian
from temgym_core.run import column_matrix, run_iter, run_to_end
from temgym_core.ray import Ray
from temgym_core.source import decompose_gaussian
import smiley_calibrated_optics as calibrated

jax.config.update("jax_enable_x64", True)
HERE = Path(__file__).resolve().parent


def _code_signature(code):
    """Compare executable Python code, ignoring filenames and line numbers."""
    return (code.co_code, code.co_names, code.co_varnames, code.co_freevars,
            code.co_cellvars, tuple(_code_signature(v) if isinstance(v, CodeType)
                                    else v for v in code.co_consts))


def assert_current_transport():
    """Detect stale core imports before compiling another notebook renderer.

    Reloading this helper does not reload classes imported from TemGymCore.
    Compare their actual in-memory bytecode with the workspace sources without
    executing/reloading those modules or mixing incompatible class instances.
    """
    compiled = {}
    stale = []
    for cls, method in ((HardEdgeSolenoid, "_call_gaussian"),
                        (SeidelLens, "_joint_gaussian"),
                        (ABCDTransfer, "_call_gaussian"),
                        (FreeSpacePropagator, "__call__")):
        path = Path(inspect.getsourcefile(cls))
        if path not in compiled:
            compiled[path] = compile(path.read_text(), str(path), "exec", dont_inherit=True)
        class_code = next(v for v in compiled[path].co_consts
                          if isinstance(v, CodeType) and v.co_name == cls.__name__)
        expected = next(v for v in class_code.co_consts
                        if isinstance(v, CodeType) and v.co_name == method)
        actual = getattr(getattr(cls, method, None), "__code__", None)
        if actual is None or _code_signature(actual) != _code_signature(expected):
            stale.append(f"{cls.__name__}.{method}")
    if stale:
        raise RuntimeError("Stale TemGymCore code is loaded: "+", ".join(stale)
            +". Restart the notebook kernel and Run All; reloading smiley_defocus alone is insufficient.")


def _smiley(size):
    '''
    Smiley face test object from https://doi.org/10.1093/micmic/ozad021
    '''
    obj = np.ones((size, size), dtype=np.complex64)
    y, x = np.ogrid[-size // 2:size // 2, -size // 2:size // 2]

    outline = (((y * 1.2) ** 2 + x**2) > (110 / 256 * size) ** 2) & (
        ((y * 1.2) ** 2 + x**2) < (120 / 256 * size) ** 2
    )
    obj[outline] = 0.0

    left_eye = ((y + 40 / 256 * size) ** 2 + (x + 40 / 256 * size) ** 2) < (20 / 256 * size) ** 2
    obj[left_eye] = 0
    right_eye = (np.abs(y + 40 / 256 * size) < 15 / 256 * size) & (
        np.abs(x - 40 / 256 * size) < 30 / 256 * size
    )
    obj[right_eye] = 0

    nose = (y + 20 / 256 * size + x > 0) & (x < 0) & (y < 10 / 256 * size)
    obj[nose] = (0.05j * x + 0.05j * y)[nose]

    mouth = (((y * 1) ** 2 + x**2) > (50 / 256 * size) ** 2) & (
        ((y * 1) ** 2 + x**2) < (70 / 256 * size) ** 2
    ) & (y > 20 / 256 * size)
    obj[mouth] = 0

    tongue = (((y - 50 / 256 * size) ** 2 + (x - 50 / 256 * size) ** 2) < (20 / 256 * size) ** 2) & (
        (y**2 + x**2) > (70 / 256 * size) ** 2
    )
    obj[tongue] = 0

    signature_wave = np.exp(1j * (3 * y + 7 * x) * 2 * np.pi / size)
    obj += 0.3 * signature_wave - 0.3
    return obj


class SmileyDefocus:
    """Interactive microscope with table-driven illumination and imaging.

    A fitted virtual-source mode traverses the calibrated finite lenses. Its
    numerical basis resolves the specimen. Inserted apertures trigger resolved
    aperture propagation and 2D field recomposition at the specimen.
    """

    # Lets a rerun of the controls refresh instances with older viewing behavior.
    sampling_api_version = 8
    aperture_quadrature_across = 512

    def __init__(self, lattice_shape=(500, 1000), *, propagation_mode="source"):
        assert_current_transport()
        if propagation_mode not in ("source", "recompose"):
            raise ValueError("propagation_mode must be 'source' or 'recompose'.")
        self.propagation_mode = propagation_mode
        self.view_plane = "detector"
        self.magnification = 40_000.
        self.spot_size = 3.
        self.aperture_radius_m = None
        self.aperture = None
        self.model = model = calibrated.microscope(self.spot_size, self.magnification)
        self.spot_options = calibrated.SPOTS
        self.magnification_options = calibrated.MAGNIFICATIONS
        self.z_sample = float(self.model.components["sample"].z)
        self.detector = replace(self.model.components["detector"], shape=(128, 128))
        self.upstream = tuple(c for c in self.model.column if float(c.z) < self.z_sample)
        self.downstream = tuple(
            c for c in self.model.column
            if self.z_sample < float(c.z) < float(self.detector.z)
        ) + (self.detector,)
        self.propagator = FreeSpacePropagator()

        # Coherent member of the fitted virtual source, at its fitted waist.
        # Its size, position and mean tilt come from the saved calibration.
        self.source = calibrated.source()
        to_sample = self.upstream + (Plane(z=self.z_sample),)
        matrix = column_matrix(to_sample, source_z=self.source.z, voltage=model.voltage)
        self.source_to_sample_inverse = jnp.linalg.inv(matrix[:4, :4])
        self.source_to_sample_offset = matrix[:4, 4]
        returned = run_to_end(self.source, to_sample, propagator=self.propagator)
        # Use relative intensity with a unit nominal specimen peak. This one
        # constant is retained across spot/aperture changes.
        self.source = self.source.derive(amplitude=self.source.amplitude/abs(returned.amplitude))
        self.illumination = run_to_end(self.source, self.upstream, propagator=self.propagator)
        self.nominal_illumination = self.propagator(
            self.illumination, self.z_sample-self.illumination.z)
        target = self.nominal_illumination

        # Retain the original smiley geometry and its signature. Smooth and
        # bound its amplitude; add a modest phase for visible defocus contrast.
        raw = _smiley(256)
        amplitude = np.abs(raw)
        amplitude /= amplitude.max()
        # Pad into vacuum before smoothing to avoid an artificial sharp jump
        # at the map boundary; keep the smiley outline inside the map.
        phase = 0.8 * gaussian_filter(np.pad(1. - amplitude, 32), 2.)
        amplitude = gaussian_filter(np.pad(0.15 + 0.85 * amplitude, 32,
                                           constant_values=1.), 2.)
        self.sample_width_m = 140e-9
        axis = (np.arange(320) - 159.5) * self.sample_width_m / 256
        self.sample_map_width_m = 320 * self.sample_width_m / 256
        # Match the notebook's original top-down smiley orientation in a
        # physical (positive-y-up) coordinate system.
        self.sample_amplitude = amplitude[::-1].copy()
        self.sample_phase = phase[::-1].copy()
        self.sample = InterpolatedSample2D.from_action_maps(
            pathlength_yx_m=self.sample_phase / float(target.k),
            log_amplitude_yx=np.log(self.sample_amplitude),
            x_coords_m=axis, y_coords_m=axis, z=self.z_sample, boundary="vacuum",
        )

        self.method = "spatial" if gpu_kernel_available() else "scan"
        self.objective = self.model.components["Obj_post"]
        # Aberration actions supplement the native finite lenses, whose
        # focusing, principal planes and magnetic rotation remain intact.
        self.objective_focal_length = float(self.objective.length/(self.objective.q*jnp.sin(self.objective.q)))
        self.objective_ideal_inverse = Lens(z=self.objective.z, focal_length=-self.objective_focal_length)
        # Shared by count variants so all use the same nominal diagram scales.
        self.diagram_scale = {}
        self._configure_projector()
        self._configure_view_plane()
        self._configure_sampling(lattice_shape)

    def _configure_projector(self):
        self.objective = self.model.components["Obj_post"]
        self.objective_focal_length = float(self.objective.length/(self.objective.q*jnp.sin(self.objective.q)))
        self.objective_ideal_inverse = Lens(z=self.objective.z, focal_length=-self.objective_focal_length)
        self.projector = self.model.components["PL1"]
        self.projector_focal_length = np.inf
        matrix = np.asarray(column_matrix((self.projector, self.detector),
            source_z=float(self.projector.z), voltage=self.model.voltage))
        a, b = matrix[:2, :2], matrix[:2, 2:4]
        self.projector_object_distance = -float(np.sum(a*b)/np.sum(a*a))

    def _configure_view_plane(self):
        """Locate the nominal specimen Fourier plane (A=0) after native IL1.

        Keep this plane fixed when specimen height/aberrations change, just as
        the physical detector stays fixed. Recompute it for new lens currents.
        """
        if self.view_plane == "detector":
            self.view_detector = self.detector
            self.view_label = "Detector image"
            return
        il1 = self.model.components["IL1"]
        preceding = tuple(c for c in self.downstream if float(c.z) <= float(il1.z))
        matrix = np.asarray(column_matrix(preceding, source_z=self.z_sample,
                                          voltage=self.model.voltage))
        a, c = matrix[:2, :2], matrix[2:4, :2]
        distance = -np.sum(a*c)/np.sum(c*c)
        z = float(il1.z+il1.length)+distance
        next_z = min(float(c.z) for c in self.downstream if float(c.z) > float(il1.z))
        if not (float(il1.z+il1.length) < z < next_z) or not np.allclose(a+distance*c, 0., atol=1e-8):
            raise ValueError("These lens settings have no specimen Fourier plane between IL1 and IL2.")
        b = matrix[:2, 2:4]+distance*matrix[2:4, 2:4]
        self.diffraction_camera_length_m = float(np.linalg.norm(b)/np.sqrt(2.))
        # ±0.12 nm^-1 covers the smiley's spatial frequencies. The native B
        # block maps specimen angle to physical position, including rotation.
        wavelength = 2*np.pi/float(self.source.k)
        width = 2*.12e9*wavelength*self.diffraction_camera_length_m
        self.view_detector = replace(self.detector, z=z,
            pixel_size=tuple(width/n for n in self.detector.shape))
        self.view_label = "IL1 crossover · diffraction"

    def with_view_plane(self, plane):
        """Select a virtual screen without changing the microscope currents."""
        if plane not in ("detector", "il1_crossover"):
            raise ValueError("plane must be 'detector' or 'il1_crossover'.")
        result = copy(self)
        result.view_plane = plane
        result._configure_view_plane()
        result._configure_sampling(self.lattice_shape)
        return result

    def viewing_column(self, *coefficients):
        column = self.objective_column(*coefficients)
        if self.view_plane == "detector":
            return column
        return tuple(c for c in column if float(c.z) < float(self.view_detector.z)) + (self.view_detector,)

    @property
    def aperture_options(self):
        radii = self.model.config["apertures"]["C_aperture"]["radii_m"]
        return tuple(radii)

    def with_settings(self, *, spot_size=None, magnification=None, aperture_radius_m=None):
        """Independent optics/cache, preserving the original coherent source.

        The aperture radius is in metres; None withdraws it. Table values set
        native solenoid controls, with an explicit IL1–PL1 imaging solve.
        """
        result = copy(self)
        result.spot_size = self.spot_size if spot_size is None else float(spot_size)
        result.magnification = self.magnification if magnification is None else float(magnification)
        model = calibrated.microscope(result.spot_size, result.magnification)
        if aperture_radius_m is not None:
            radius = next((r for r in self.aperture_options
                           if np.isclose(aperture_radius_m, r, rtol=1e-12, atol=0.)), None)
            if radius is None:
                raise ValueError("Choose a condenser aperture radius from aperture_options, or None.")
            aperture_radius_m = radius
        result.model = model
        result.aperture_radius_m = aperture_radius_m
        result.upstream = tuple(c for c in model.column if float(c.z) < result.z_sample)
        result.downstream = tuple(c for c in model.column
                                  if result.z_sample < float(c.z) < float(result.detector.z)) + (result.detector,)
        matrix = column_matrix(result.upstream + (Plane(z=result.z_sample),),
                               source_z=result.source.z, voltage=model.voltage)
        result.source_to_sample_inverse = jnp.linalg.inv(matrix[:4, :4])
        result.source_to_sample_offset = matrix[:4, 4]
        result.illumination = run_to_end(result.source, result.upstream, propagator=result.propagator)
        result.nominal_illumination = result.propagator(
            result.illumination, result.z_sample-result.illumination.z)
        result.aperture = None
        if aperture_radius_m is not None:
            z = model.config["apertures"]["C_aperture"]["z_m"]
            calibration = model.config["calibration"]
            parameters = calibration["condenser_parameters"]
            centre = parameters[29:31] if aperture_radius_m == 25e-6 else parameters[20:22]
            result.aperture = SigmoidAperture(z=z, radius=aperture_radius_m,
                x0=centre[0]*1e-6, y0=centre[1]*1e-6,
                edge_width=calibration["aperture_edge_width_m"])
            before = tuple(c for c in result.upstream if float(c.z) < z) + (Plane(z=z),)
            # Resolve the sigmoid at its own plane, before coherent propagation.
            bundle = decompose_gaussian(result.source, num_across=self.aperture_quadrature_across, through=before)
            at_aperture = run_to_end(bundle, before, propagator=result.propagator)
            result.aperture_beams = jax.jit(result.aperture)(at_aperture)
            result.after_aperture = tuple(c for c in result.upstream if float(c.z) > z)
        result._configure_projector()
        result._configure_view_plane()
        result._configure_sampling(self.lattice_shape)
        return result

    def _aperture_correction(self, incident, x, y, support_width):
        """Recompose the transmitted field on a 2D specimen grid.

        Fitted source offsets and tilt break radial symmetry. All illumination
        beamlets originate at the fitted source and resolve the actual aperture.
        """
        beam = run_to_end(self.aperture_beams,
                          self.after_aperture + (Plane(z=incident.z),),
                          propagator=self.propagator)
        count = 256
        axis = jnp.linspace(-3.6*support_width, 3.6*support_width, count)
        xx, yy = jnp.meshgrid(axis+incident.x, axis+incident.y)
        grid = SimpleNamespace(coords=jnp.stack((xx.ravel(), yy.ravel()), axis=-1),
                               shape=(count, count))
        transmitted = evaluate_gaussians(beam, grid, method=self.method, tile_beams=32)
        ideal = evaluate_gaussians(incident, grid, method="scan")
        ratio = transmitted/ideal
        from jax.scipy.ndimage import map_coordinates
        coordinates = jnp.stack(((y/(7.2*support_width)+.5)*(count-1),
                                  (x/(7.2*support_width)+.5)*(count-1)))
        return (map_coordinates(ratio.real, coordinates, order=1, mode="nearest")
                +1j*map_coordinates(ratio.imag, coordinates, order=1, mode="nearest"))

    def _configure_sampling(self, lattice_shape):
        """Configure a fresh instance: JIT functions capture this lattice."""
        shape = tuple(index(v) for v in lattice_shape)
        if len(shape) != 2 or min(shape) < 8:
            raise ValueError("lattice_shape must contain two integers of at least 8.")
        self.lattice_shape = shape
        self.num_beams = int(np.prod(shape))
        # Same physical support and illumination for every numerical count.
        ny, nx = shape
        x, y = np.meshgrid((np.arange(nx)-(nx-1)/2)/nx,
                           (np.arange(ny)-(ny-1)/2)/ny)
        self.lattice_x = jnp.asarray(x.ravel())
        self.lattice_y = jnp.asarray(y.ravel())
        self.propagate = jax.jit(self._propagate)
        self.render = jax.jit(self._render)
        self.cache = {}
        # A stratified subset of actual lattice members within 2 intensity RMS.
        iy = np.unique(np.rint(np.linspace(0.3*(ny-1), 0.7*(ny-1), 33)).astype(int))
        ix = np.unique(np.rint(np.linspace(0.3*(nx-1), 0.7*(nx-1), 33)).astype(int))
        yy, xx = np.meshgrid(iy, ix, indexing="ij")
        indices = (yy*nx+xx).ravel()
        inside = x.ravel()[indices]**2+y.ravel()[indices]**2 <= 0.2**2
        self.diagram_indices = jnp.asarray(indices[inside])
        self.trace_preview = jax.jit(self._trace_preview)
        self.diagram_cache = {}

    def with_lattice_shape(self, lattice_shape):
        """Reuse fixed optics/sample with an independent Gaussian basis/cache.

        Never mutate the lattice captured by an existing compiled renderer.
        The new count compiles on its first render; callers can keep instances
        to reuse that compiled renderer when switching back to an earlier count.
        """
        result = copy(self)
        result._configure_sampling(lattice_shape)
        return result

    def with_propagation_mode(self, mode):
        """Independent renderer/cache for continuous propagation or recomposition."""
        if mode not in ("source", "recompose"):
            raise ValueError("mode must be 'source' or 'recompose'.")
        result = copy(self)
        result.propagation_mode = mode
        result._configure_sampling(self.lattice_shape)
        return result

    def incident_beamlets(self, height_nm, indices=None):
        if self.propagation_mode == "source" and self.aperture is None:
            return self.continuous_incident_beamlets(height_nm, indices)
        return self._specimen_basis(height_nm, indices)

    def _specimen_basis(self, height_nm, indices=None, *, apply_aperture=True):
        """Gaussian convolution quadrature of the transported incident wave."""
        incident = self.propagator(
            self.illumination,
            self.z_sample + height_nm * 1e-9 - self.illumination.z,
        )
        q = incident.Q_inv[0, 0]
        k = incident.k
        width = jnp.sqrt(2. / (k * jnp.imag(q)))
        curvature = jnp.real(q)
        ny, nx = self.lattice_shape
        # Cover the full calibrated illumination, including the broadest spot.
        # Count controls the sampling resolution; never truncate the wave to
        # the detector object field, which would change its diffraction.
        support_width = width
        sx, sy = 5 * support_width / nx, 5 * support_width / ny
        wx, wy = 1.1 * sx, 1.1 * sy
        vx, vy = width**2 - wx**2, width**2 - wy**2
        lattice_x, lattice_y = self.lattice_x, self.lattice_y
        if indices is not None:
            lattice_x, lattice_y = lattice_x[indices], lattice_y[indices]
        x, y = 5 * support_width * lattice_x, 5 * support_width * lattice_y
        weights = sx * sy * width**2 / (jnp.pi * wx * wy * jnp.sqrt(vx * vy))
        weights *= jnp.exp(-x*x/vx - y*y/vy)
        q_beam = jnp.diag(jnp.array([curvature + 2j/(k*wx*wx),
                                    curvature + 2j/(k*wy*wy)]))
        if apply_aperture and self.aperture is not None:
            weights = weights * self._aperture_correction(incident, x, y, support_width)
        return GaussianBeam(
            x=incident.x+x, y=incident.y+y,
            dx=incident.dx+curvature*x, dy=incident.dy+curvature*y,
            z=incident.z, voltage=incident.voltage,
            amplitude=incident.amplitude*weights,
            pathlength=incident.pathlength + incident.dx*x + incident.dy*y
            + 0.5*curvature*(x*x+y*y),
            Q_inv=jnp.broadcast_to(q_beam, (x.size, 2, 2)),
        )

    def source_beamlets(self, height_nm, indices=None):
        """Choose a specimen-resolving basis, then launch it at the source.

        The source represents the same incident wave over the specimen window.
        Its numerical basis follows the specimen height, so a height change does
        not broaden the packets used for the sample's local action expansion.
        No aperture action is included when choosing this incident basis.
        """
        target = self._specimen_basis(height_nm, indices, apply_aperture=False)
        nominal = self.propagator(target, -height_nm*1e-9)
        offset = self.source_to_sample_offset
        shifted = nominal.derive(x=nominal.x-offset[0], y=nominal.y-offset[1],
                                 dx=nominal.dx-offset[2], dy=nominal.dy-offset[3])
        inverse = self.source_to_sample_inverse
        source = ABCDTransfer(A=inverse[:2, :2], B=inverse[:2, 2:],
                              C=inverse[2:, :2], D=inverse[2:, 2:])(shifted).derive(
            z=self.source.z, amplitude=jnp.ones_like(target.amplitude),
            pathlength=jnp.zeros_like(target.pathlength))
        returned = run_to_end(source, self.upstream+(Plane(z=target.z),),
                              propagator=self.propagator)
        return source.derive(amplitude=target.amplitude/returned.amplitude,
                             pathlength=target.pathlength-returned.pathlength)

    @property
    def illumination_column(self):
        if self.aperture is None:
            return self.upstream
        return tuple(sorted((*self.upstream, self.aperture), key=lambda c: float(c.z)))

    def continuous_incident_beamlets(self, height_nm, indices=None):
        source = self.source_beamlets(height_nm, indices)
        return run_to_end(source, self.illumination_column+(
            Plane(z=self.z_sample+height_nm*1e-9),), propagator=self.propagator)

    def objective_column(self, astigmatism_um=0., coma_mm=0., spherical_mm=0., projector_distortion=0.):
        """Native objective and PL1, each retaining its magnetic rotation.

        projector_distortion is the radial Seidel E coefficient in 1e9 m^-3.
        """
        objective = KrivanekLens(
            z=self.objective.z, focal_length=self.objective_focal_length,
            coeffs=KrivanekCoeffs(C12=astigmatism_um*1e-6,
                                  C21=coma_mm*1e-3, C30=spherical_mm*1e-3),
        )
        projector = DistortedLens(
            z=self.projector.z, focal_length=np.inf,
            object_plane_dist=self.projector_object_distance,
            IsoDist=projector_distortion*1e9,
        )
        column = []
        for component in self.downstream:
            if component is self.objective:
                column.extend((objective, self.objective_ideal_inverse))
            elif component is self.projector:
                column.append(projector)
            column.append(component)
        return tuple(column)

    def _propagate(self, height_nm, astigmatism_um=0., coma_mm=0., spherical_mm=0., projector_distortion=0.):
        incident = self.incident_beamlets(height_nm)
        sample = replace(self.sample, z=self.z_sample + height_nm*1e-9)
        return run_to_end(incident, (sample, *self.viewing_column(
            astigmatism_um, coma_mm, spherical_mm, projector_distortion)), propagator=self.propagator)

    def _render(self, height_nm, astigmatism_um=0., coma_mm=0., spherical_mm=0., projector_distortion=0.):
        output = self._propagate(height_nm, astigmatism_um, coma_mm, spherical_mm, projector_distortion)
        field = evaluate_gaussians(output, self.view_detector, method=self.method,
                                   tile_beams=32)
        return jnp.abs(field * (self.magnification if self.view_plane == "detector" else 1.))**2

    def image(self, height_nm, *, astigmatism_um=0., coma_mm=0., spherical_mm=0., projector_distortion=0., use_cache=True):
        """Render the selected screen and return intensity, seconds, cached.

        Each optics variant has its own cache; keys include height and aberrations. Every fresh image transports
        all selected Gaussian beamlets through the actual aberrated objective.
        """
        coefficients = tuple(float(v) for v in (astigmatism_um, coma_mm, spherical_mm, projector_distortion))
        if not np.isfinite((height_nm, *coefficients)).all():
            raise ValueError("Height and aberration coefficients must be finite.")
        key = (float(height_nm), *coefficients) if any(coefficients) else float(height_nm)
        start = perf_counter()
        cached = use_cache and key in self.cache
        if cached:
            image = self.cache[key]
        else:
            image = np.asarray(self.render(*(jnp.asarray(v, dtype=jnp.float64)
                for v in (height_nm, *coefficients))))
        if not np.isfinite(image).all() or image.max() <= 0:
            reason = (f"{np.count_nonzero(~np.isfinite(image))} non-finite pixels"
                      if not np.isfinite(image).all() else "zero intensity on the screen")
            aperture = "out" if self.aperture_radius_m is None else f"{self.aperture_radius_m*1e6:g} µm radius"
            raise RuntimeError(
                f"Gaussian render failed ({reason}); PL1={projector_distortion:g}, "
                f"magnification={self.magnification:g}, spot={self.spot_size:g}, "
                f"aperture={aperture}, beams={self.num_beams}, height_nm={height_nm:g}, "
                f"C12_um={astigmatism_um:g}, C21_mm={coma_mm:g}, C30_mm={spherical_mm:g}, "
                f"view={self.view_plane}, propagation={self.propagation_mode}.")
        if use_cache and not cached:
            self.cache[key] = image
            while len(self.cache) > 128:
                del self.cache[next(iter(self.cache))]
        return image, perf_counter() - start, cached


    def _trace_preview(self, height_nm, astigmatism_um, coma_mm, spherical_mm, projector_distortion=0.):
        """Central trajectories of actual Gaussian lattice members.

        Without an aperture, source mode traces the same Gaussian states
        through every component. Recomposition back-maps the specimen basis
        for the upstream guide; downstream trajectories are actual beamlets.
        Both modes use Gaussian objective/projector actions, not the legacy
        ordinary-ray aberration convention.
        """
        continuous = self.propagation_mode == "source" and self.aperture is None
        if continuous:
            source = self.source_beamlets(height_nm, self.diagram_indices)
            upstream = self.illumination_column
        else:
            incident = self.incident_beamlets(height_nm, self.diagram_indices)
            nominal_xy = incident.r_xy - height_nm*1e-9*incident.d_xy
            state = jnp.concatenate((nominal_xy, incident.d_xy), axis=-1)
            source_state = (state-self.source_to_sample_offset) @ self.source_to_sample_inverse.T
            source = Ray(x=source_state[:, 0], y=source_state[:, 1],
                         dx=source_state[:, 2], dy=source_state[:, 3],
                         z=self.source.z, pathlength=0., voltage=self.model.voltage)
            upstream = self.upstream
        zs, coordinates, rotations = [], [], []
        angle = jnp.asarray(0.)

        def record(beam):
            zs.append(jnp.ravel(jnp.asarray(beam.z))[0]-self.z_sample)
            coordinates.append(jnp.stack((beam.x, beam.y), axis=-1))
            rotations.append(angle)

        record(source)
        specimen_z = self.z_sample+height_nm*1e-9
        for operation, beam in run_iter(source, upstream+(Plane(z=specimen_z),),
                                        propagator=self.propagator if continuous else None):
            if isinstance(operation, HardEdgeSolenoid):
                angle = angle-operation.q
            elif isinstance(operation, ElectromagneticLens):
                angle = angle + operation.rotation_angle(self.model.voltage)
            record(beam)
        if continuous:
            incident = beam
        sample = replace(self.sample, z=self.z_sample+height_nm*1e-9)
        for operation, beam in run_iter(incident, (sample, *self.objective_column(
                astigmatism_um, coma_mm, spherical_mm, projector_distortion)), propagator=self.propagator):
            if isinstance(operation, HardEdgeSolenoid):
                angle = angle-operation.q
            elif isinstance(operation, ElectromagneticLens):
                angle = angle + operation.rotation_angle(self.model.voltage)
            record(beam)
        return jnp.stack(zs), jnp.stack(coordinates), jnp.stack(rotations)

    def diagram_segments(self, height_nm=0., *, astigmatism_um=0., coma_mm=0., spherical_mm=0., projector_distortion=0.):
        """Cache the shaded range of sampled Gaussian centres for these controls."""
        key = tuple(float(v) for v in (height_nm, astigmatism_um, coma_mm, spherical_mm, projector_distortion))
        if key not in self.diagram_cache:
            z, xy, rotation = (np.asarray(v) for v in self.trace_preview(
                *(jnp.asarray(v, dtype=jnp.float64) for v in key)))
            if not (np.isfinite(xy).all() and np.isfinite(z).all()):
                raise RuntimeError("Non-finite Gaussian preview trajectory.")
            x = xy[..., 0]*np.cos(rotation[:, None])+xy[..., 1]*np.sin(rotation[:, None])
            t = np.linspace(0., 1., 65)[:, None]
            segments = []
            for i in range(len(z)-1):
                # Component interactions may shift centres at one z; separate
                # drift segments avoid drawing spurious connecting diagonals.
                if z[i+1] > z[i]:
                    positions = (1-t)*x[i]+t*x[i+1]
                    segments.append((np.linspace(z[i], z[i+1], len(t)),
                                     np.column_stack((positions.min(axis=1), positions.max(axis=1)))))
            self.diagram_cache[key] = segments
            while len(self.diagram_cache) > 32:
                del self.diagram_cache[next(iter(self.diagram_cache))]
        return self.diagram_cache[key]

    def draw_rays(self, ax, height_nm=0., section=0, *,
                  astigmatism_um=0., coma_mm=0., spherical_mm=0., projector_distortion=0.):
        """Calibration-style view of the actual Gaussian central trajectories."""
        from temgym_core.calibration.viewer import _draw_ray_section
        from temgym_core.components import DoubleDeflector

        coefficients = dict(astigmatism_um=astigmatism_um, coma_mm=coma_mm,
                            spherical_mm=spherical_mm, projector_distortion=projector_distortion)
        names = {float(c.z): name for name, c in self.model.components.items()}
        sample = replace(self.sample, z=self.z_sample+height_nm*1e-9)
        upstream = self.upstream if self.aperture is None else tuple(
            sorted((*self.upstream, self.aperture), key=lambda c: float(c.z)))
        components = (*upstream, sample, *self.objective_column(**coefficients))
        column, labels = [], []
        for component in components:
            if component is sample:
                name = f"Smiley ({height_nm/1000:+.2f} µm)"
            elif component is self.objective_ideal_inverse:
                name = ""  # ideal kick cancels the extra thin-lens focusing
            elif component is self.aperture:
                name = f"C aperture (r={self.aperture.radius*1e6:g} µm)"
            elif isinstance(component, DistortedLens):
                name = ""
            elif isinstance(component, KrivanekLens):
                name = ""
            else:
                name = names[float(component.z)]
            column.append(replace(component, z=float(component.z)-self.z_sample))
            labels.extend(("Deflector 1", "Deflector 2")
                          if isinstance(component, DoubleDeflector) else (name,))
        column = tuple(column)
        segments = self.diagram_segments(height_nm, **coefficients)
        relative_z = lambda name: float(self.model.components[name].z)-self.z_sample
        bounds = [
            (float(self.source.z)-self.z_sample, relative_z("C_mini")),
            (relative_z("C_mini"), relative_z("Obj_post")+0.001),
            (relative_z("Obj_post")-0.001, relative_z("PL1")),
            (relative_z("PL1"), float(self.detector.z)-self.z_sample),
            (-25e-6, 25e-6),
        ]
        if "segments" not in self.diagram_scale:
            self.diagram_scale["segments"] = self.diagram_segments(0.)
        ax.clear()
        _draw_ray_section(ax, column, segments, section,
                          section_bounds=bounds, component_labels=labels,
                          source_label="Virtual source (fit)", mark_source_image=False,
                          scale_segments=self.diagram_scale["segments"])
        if any(coefficients.values()):
            # Same specimen height and basis: isolate the objective's effect.
            reference = self.diagram_segments(height_nm)
            lo, hi = bounds[section]
            factor = 1e3 if section == 3 else 1e6
            for z, edges in reference:
                if z[-1] < lo or z[0] > hi:
                    continue
                keep = (z >= lo) & (z <= hi)
                for edge in edges.T:
                    ax.plot(edge[keep]*factor, z[keep]*1000, color="#765a99",
                            ls="--", lw=0.8, alpha=0.8)
            ax.plot([], [], color="#765a99", ls="--", lw=0.8, label="Zero aberrations")
            ax.legend(loc="upper left", bbox_to_anchor=(0., -0.14),
                      fontsize=8, frameon=False, borderaxespad=0.)
        if self.view_plane != "detector":
            screen_z = float(self.view_detector.z)-self.z_sample
            if bounds[section][0] <= screen_z <= bounds[section][1]:
                ax.axhline(screen_z*1000, color="#c66b16", ls=":", lw=1.5)
                ax.text(.02, screen_z*1000, " IL1 diffraction screen",
                        transform=ax.get_yaxis_transform(), color="#a44f08",
                        fontsize=8, va="bottom")
        ax.set_ylabel("z from nominal specimen (mm)")
        ax.set_xlabel("Rotating-frame x (mm)" if section == 3 else "Rotating-frame x (µm)")
        titles = ("Source → mini", "Mini → specimen", "Projector lenses",
                  "Detector", "Specimen ±25 µm")
        ax.set_title(titles[section]+" · Gaussian centres", loc="left", fontsize=10)
        return column
