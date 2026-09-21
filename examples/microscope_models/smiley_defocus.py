"""Forward model for the interactive Gaussian smiley microscope notebook."""
from copy import copy
from dataclasses import replace
from operator import index
from pathlib import Path
import os
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter

from temgym_core.aberrations import KrivanekCoeffs
from temgym_core.components import ABCDTransfer, ElectromagneticLens, InterpolatedSample2D, KrivanekLens, Plane
from temgym_core.evaluate import evaluate_gaussians, gpu_kernel_available
from temgym_core.gaussian import FreeSpacePropagator, GaussianBeam, make_gaussian
from temgym_core.microscope import Microscope, find_imaging_condition
from temgym_core.run import column_matrix, run_iter, run_to_end
from temgym_core.ray import Ray

jax.config.update("jax_enable_x64", True)
HERE = Path(__file__).resolve().parent


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
    """Fixed full microscope; only the specimen's axial position changes.

    The lossless condenser transports one coherent Gaussian exactly. At each
    specimen height its wave is resolved into a rectangular Gaussian lattice;
    all these beamlets interact with the specimen and traverse the native
    post-specimen components. Re-decomposition avoids applying the local sample
    action expansion to beamlets that have broadened during a specimen drift.
    """

    # Lets a rerun of the controls refresh instances with older diagram behavior.
    sampling_api_version = 3

    def __init__(self, lattice_shape=(500, 1000)):
        self.magnification = 40_000.
        model = Microscope.from_toml(HERE / "microscope.toml").with_settings(
            parallel=3., magnification=self.magnification,
        )
        self.model = find_imaging_condition(
            model, magnification=self.magnification, vary=["IL1", "IL2", "IL3"],
        ).microscope
        self.z_sample = float(self.model.components["sample"].z)
        self.detector = replace(self.model.components["detector"], shape=(128, 128))
        self.upstream = tuple(c for c in self.model.column if float(c.z) < self.z_sample)
        self.downstream = tuple(
            c for c in self.model.column
            if self.z_sample < float(c.z) < float(self.detector.z)
        ) + (self.detector,)
        self.propagator = FreeSpacePropagator(include_axial_path=False)

        # Choose a coherent illumination mode with a 100 nm amplitude radius
        # at nominal sample height, then infer its source-plane curvature.
        # The TOML source size is not imposed on this illustrative mode.
        target = make_gaussian(z=self.z_sample, voltage=model.voltage,
                               waist_x=100e-9, waist_y=100e-9)
        to_sample = self.upstream + (Plane(z=self.z_sample),)
        matrix = column_matrix(to_sample, source_z=0., voltage=model.voltage)
        inverse = jnp.linalg.inv(matrix[:4, :4])
        self.source_to_sample_inverse = inverse
        self.source_to_sample_offset = matrix[:4, 4]
        source = ABCDTransfer(A=inverse[:2, :2], B=inverse[:2, 2:],
                              C=inverse[2:, :2], D=inverse[2:, 2:])(target).derive(z=0.)
        returned = run_to_end(source, to_sample, propagator=self.propagator)
        self.source = source.derive(amplitude=1. / returned.amplitude,
                                    pathlength=-returned.pathlength)
        self.illumination = run_to_end(self.source, self.upstream,
                                        propagator=self.propagator)
        self.nominal_illumination = self.propagator(
            self.illumination, self.z_sample - self.illumination.z,
        )
        np.testing.assert_allclose(self.nominal_illumination.Q_inv, target.Q_inv,
                                   rtol=1e-6, atol=1e-5)

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
        self.objective_focal_length = float(self.objective.focal_length(self.model.voltage))
        theta = float(self.objective.rotation_angle(self.model.voltage))
        rotation = jnp.asarray([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
        # A pure transverse rotation has unit determinant and no optical action.
        # Keep it after the Krivanek thin-lens kick, as in ElectromagneticLens.
        self.objective_rotation = ABCDTransfer(
            z=self.objective.z, A=rotation, B=jnp.zeros((2, 2)), D=rotation)

        # Shared by count variants so all use the same nominal diagram scales.
        self.diagram_scale = {}
        self._configure_sampling(lattice_shape)

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

    def incident_beamlets(self, height_nm, indices=None):
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
        sx, sy = 5 * width / nx, 5 * width / ny
        wx, wy = 1.1 * sx, 1.1 * sy
        vx, vy = width**2 - wx**2, width**2 - wy**2
        lattice_x, lattice_y = self.lattice_x, self.lattice_y
        if indices is not None:
            lattice_x, lattice_y = lattice_x[indices], lattice_y[indices]
        x, y = 5 * width * lattice_x, 5 * width * lattice_y
        weights = sx * sy * width**2 / (jnp.pi * wx * wy * jnp.sqrt(vx * vy))
        weights *= jnp.exp(-x*x/vx - y*y/vy)
        q_beam = jnp.diag(jnp.array([curvature + 2j/(k*wx*wx),
                                    curvature + 2j/(k*wy*wy)]))
        return GaussianBeam(
            x=incident.x+x, y=incident.y+y,
            dx=incident.dx+curvature*x, dy=incident.dy+curvature*y,
            z=incident.z, voltage=incident.voltage,
            amplitude=incident.amplitude*weights,
            pathlength=incident.pathlength + incident.dx*x + incident.dy*y
            + 0.5*curvature*(x*x+y*y),
            Q_inv=jnp.broadcast_to(q_beam, (x.size, 2, 2)),
        )

    def objective_column(self, astigmatism_um=0., coma_mm=0., spherical_mm=0.):
        """Native Krivanek objective followed by the original magnetic rotation."""
        objective = KrivanekLens(
            z=self.objective.z, focal_length=self.objective_focal_length,
            x0=self.objective.x0, y0=self.objective.y0,
            coeffs=KrivanekCoeffs(C12=astigmatism_um*1e-6,
                                  C21=coma_mm*1e-3, C30=spherical_mm*1e-3),
        )
        return (objective, self.objective_rotation, *self.downstream[1:])

    def _propagate(self, height_nm, astigmatism_um=0., coma_mm=0., spherical_mm=0.):
        incident = self.incident_beamlets(height_nm)
        sample = replace(self.sample, z=self.z_sample + height_nm*1e-9)
        return run_to_end(incident, (sample, *self.objective_column(
            astigmatism_um, coma_mm, spherical_mm)), propagator=self.propagator)

    def _render(self, height_nm, astigmatism_um=0., coma_mm=0., spherical_mm=0.):
        output = self._propagate(height_nm, astigmatism_um, coma_mm, spherical_mm)
        field = evaluate_gaussians(output, self.detector, method=self.method,
                                   tile_beams=32)
        return jnp.abs(field * self.magnification)**2

    def image(self, height_nm, *, astigmatism_um=0., coma_mm=0., spherical_mm=0., use_cache=True):
        """Render the native Krivanek objective and return image, seconds, cached.

        Cache keys include every optical control. Every fresh image transports
        all selected Gaussian beamlets through the actual aberrated objective.
        """
        coefficients = tuple(float(v) for v in (astigmatism_um, coma_mm, spherical_mm))
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
            raise RuntimeError("The Gaussian image is non-finite or empty.")
        if use_cache and not cached:
            self.cache[key] = image
            while len(self.cache) > 128:
                del self.cache[next(iter(self.cache))]
        return image, perf_counter() - start, cached


    def _trace_preview(self, height_nm, astigmatism_um, coma_mm, spherical_mm):
        """Central trajectories of actual Gaussian lattice members.

        Back-map their incident centres to the source for the upstream drawing.
        At the specimen use their exact Gaussian states, then apply the same
        specimen, Krivanek objective, rotation, and projector as the image.
        Never use KrivanekLens's legacy ordinary-ray aberration convention.
        """
        incident = self.incident_beamlets(height_nm, self.diagram_indices)
        nominal_xy = incident.r_xy - height_nm*1e-9*incident.d_xy
        state = jnp.concatenate((nominal_xy, incident.d_xy), axis=-1)
        source_state = (state-self.source_to_sample_offset) @ self.source_to_sample_inverse.T
        source = Ray(x=source_state[:, 0], y=source_state[:, 1],
                     dx=source_state[:, 2], dy=source_state[:, 3],
                     z=self.source.z, pathlength=0., voltage=self.model.voltage)
        zs, coordinates, rotations = [], [], []
        angle = jnp.asarray(0.)

        def record(beam):
            zs.append(jnp.ravel(jnp.asarray(beam.z))[0]-self.z_sample)
            coordinates.append(jnp.stack((beam.x, beam.y), axis=-1))
            rotations.append(angle)

        record(source)
        for operation, beam in run_iter(source, self.upstream+(Plane(z=incident.z),)):
            if isinstance(operation, ElectromagneticLens):
                angle = angle + operation.rotation_angle(self.model.voltage)
            record(beam)
        sample = replace(self.sample, z=self.z_sample+height_nm*1e-9)
        for operation, beam in run_iter(incident, (sample, *self.objective_column(
                astigmatism_um, coma_mm, spherical_mm)), propagator=self.propagator):
            if operation is self.objective_rotation:
                angle = angle + self.objective.rotation_angle(self.model.voltage)
            elif isinstance(operation, ElectromagneticLens):
                angle = angle + operation.rotation_angle(self.model.voltage)
            record(beam)
        return jnp.stack(zs), jnp.stack(coordinates), jnp.stack(rotations)

    def diagram_segments(self, height_nm=0., *, astigmatism_um=0., coma_mm=0., spherical_mm=0.):
        """Cache the shaded range of sampled Gaussian centres for these controls."""
        key = tuple(float(v) for v in (height_nm, astigmatism_um, coma_mm, spherical_mm))
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
                  astigmatism_um=0., coma_mm=0., spherical_mm=0.):
        """Calibration-style view of the actual Gaussian central trajectories."""
        from temgym_core.calibration.viewer import _draw_ray_section
        from temgym_core.components import DoubleDeflector

        coefficients = dict(astigmatism_um=astigmatism_um, coma_mm=coma_mm,
                            spherical_mm=spherical_mm)
        names = {float(c.z): name for name, c in self.model.components.items()}
        sample = replace(self.sample, z=self.z_sample+height_nm*1e-9)
        components = (*self.upstream, sample, *self.objective_column(**coefficients))
        column, labels = [], []
        for component in components:
            if component is sample:
                name = f"Smiley ({height_nm/1000:+.2f} µm)"
            elif component is self.objective_rotation:
                name = ""  # rotation is part of the labelled objective
            elif isinstance(component, KrivanekLens):
                name = "Obj_post (Krivanek)"
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
                          source_label="Source", mark_source_image=False,
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
        ax.set_ylabel("z from nominal specimen (mm)")
        ax.set_xlabel("Rotating-frame x (mm)" if section == 3 else "Rotating-frame x (µm)")
        titles = ("Source → mini", "Mini → specimen", "Projector lenses",
                  "Detector", "Specimen ±25 µm")
        ax.set_title(titles[section]+" · Gaussian centres", loc="left", fontsize=10)
        return column
