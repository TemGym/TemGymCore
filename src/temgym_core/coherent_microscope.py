"""Coherent source-to-detector microscope experiments using TemGym components.

Geometry is shared across voltages; source states may differ. Image frames use
an open condenser aperture. Separate current observations use the selected hole.
Gaussian counts and quadrature orders are fixed numerical settings during a fit.
"""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np

from .coherent_image import CoherentDetector, microscope_image
from .components import Plane, SigmoidAperture
from .constants import energy2wavelength
from .evaluate import evaluate_gaussians
from .gaussian import FreeSpacePropagator, GaussianBeam
from .run import run_to_end
from .solenoid import column_matrix, solenoid_column


def gaussian_phase_space_moments(beam):
    """Exact position/angle moments of a single coherent Gaussian beam."""
    q = jnp.asarray(beam.Q_inv).reshape(2, 2)
    k = jnp.asarray(beam.k).reshape(())
    spatial = jnp.linalg.inv(2 * k * q.imag)
    correlation = spatial @ q.real.T
    angular = q.real @ spatial @ q.real.T + jnp.linalg.inv(spatial) / (4 * k**2)
    mean = jnp.concatenate((beam.r_xy.reshape(2), beam.d_xy.reshape(2)))
    return mean, jnp.block([[spatial, correlation], [correlation.T, angular]])


def propagate_gaussian_mixture(source, components, extra_covariance):
    """Transport a Gaussian field plus Gaussian-distributed ray-centre offsets.

    The coherent component traverses the actual lens components. First-order
    transport maps the incoherent centre distribution through the same column.
    Their covariances add; no ray/intensity replacement of field interference
    is made when the resulting state is later decomposed at the specimen.
    """
    output = run_to_end(
        source,
        components,
        propagator=FreeSpacePropagator(include_axial_path=False),
    )
    mean, covariance = gaussian_phase_space_moments(output)
    matrix = column_matrix(components, source_z=source.z)[:4, :4]
    return mean, covariance + matrix @ extra_covariance @ matrix.T


@jdc.pytree_dataclass
class GaussianSourceState:
    """Unit-flux coherent Gaussian waist plus an incoherent angular distribution.

    Position/size are metres, angles radians. rms_size is intensity RMS per
    axis; additional_angular_rms is spread between coherent modes, in addition
    to each mode's diffraction divergence. upstream_distance is before CL1.
    """

    upstream_distance: float
    x: float = 0.0
    y: float = 0.0
    theta_x: float = 0.0
    theta_y: float = 0.0
    rms_size: float = 8e-9
    additional_angular_rms: float = 0.0

    def beam(self, z, voltage, angular_node):
        width = 2 * self.rms_size
        k = 2 * jnp.pi / energy2wavelength(voltage)
        return GaussianBeam(
            x=jnp.asarray(self.x),
            y=jnp.asarray(self.y),
            dx=self.theta_x + self.additional_angular_rms * angular_node[0],
            dy=self.theta_y + self.additional_angular_rms * angular_node[1],
            z=z,
            voltage=voltage,
            amplitude=jnp.sqrt(2 / (jnp.pi * width**2)) + 0j,
            Q_inv=1j * (2 / (k * width**2)) * jnp.eye(2),
            pathlength=0.0,
        )


@jdc.pytree_dataclass
class FiniteLensTrain:
    """Free-space gaps (N+1), body lengths (N), signed reference q gains (N)."""

    gaps: object
    lengths: object
    strengths: object

    @property
    def extent(self):
        return jnp.sum(self.gaps) + jnp.sum(self.lengths)

    def components(self, controls, voltage, reference_voltage, source_z):
        ratio = energy2wavelength(voltage) / energy2wavelength(reference_voltage)
        lenses = solenoid_column(
            jnp.asarray(self.gaps),
            jnp.asarray(self.lengths),
            jnp.asarray(self.strengths) * controls * ratio,
            source_z=source_z,
        )
        return tuple(
            replace(lens, include_axial_path=False) if hasattr(lens, "q") else lens
            for lens in lenses
        )


@jdc.pytree_dataclass
class ApertureGeometry:
    """One shared axial plane after CL3; independent hole centres, in metres."""

    gap_fraction: float
    centres_xy: object
    radii: object
    edge_width: float = 0.15e-6


@jdc.pytree_dataclass
class CoherentMicroscope:
    """Four condenser and five projector lenses, specimen at z=0.

    Condenser gap[0] is unused (set it to zero); source distances belong to the
    independent source states. Remaining gaps locate the four bodies backwards
    from the specimen. The aperture lies in gap[2], after CL3. The objective
    pre/post fields retain separate effective lengths and gains.
    """

    condenser: FiniteLensTrain
    projector: FiniteLensTrain
    sources: tuple
    aperture: ApertureGeometry
    reference_voltage: float = 200000.0

    def condenser_components(self, controls, voltage):
        return self.condenser.components(
            controls, voltage, self.reference_voltage, -self.condenser.extent
        )

    def projector_components(self, controls, voltage):
        return self.projector.components(controls, voltage, self.reference_voltage, 0.0)

    def source_modes(self, source_index, voltage, numerical):
        entrance = -self.condenser.extent
        source = jax.tree.map(
            lambda *values: jnp.stack(values)[source_index], *self.sources
        )
        z = entrance - source.upstream_distance
        modes = tuple(source.beam(z, voltage, node) for node in numerical.angular_nodes)
        if numerical.angular_order == 1:
            modes = tuple(
                beam.derive(
                    amplitude=jnp.where(
                        source.additional_angular_rms == 0, beam.amplitude, jnp.nan + 0j
                    )
                )
                for beam in modes
            )
        return modes

    def aperture_component(self, condenser_controls, voltage, hole):
        lenses = self.condenser_components(condenser_controls, voltage)
        z = (
            lenses[1].z
            + lenses[1].length
            + self.aperture.gap_fraction * self.condenser.gaps[2]
        )
        centre = self.aperture.centres_xy[hole]
        return SigmoidAperture(
            z=z,
            radius=self.aperture.radii[hole],
            edge_width=self.aperture.edge_width,
            x0=centre[0],
            y0=centre[1],
        )

    def components(
        self, condenser_controls, projector_controls, voltage, specimen, hole=None
    ):
        """Executable physical component column, for already-resolved source beams."""
        upstream = self.condenser_components(condenser_controls, voltage)
        if hole is not None:
            upstream = (
                *upstream[:2],
                self.aperture_component(condenser_controls, voltage, hole),
                *upstream[2:],
            )
        return (
            *upstream,
            specimen,
            *self.projector_components(projector_controls, voltage),
        )

    def transfer_matrix(
        self, condenser_controls, projector_controls, voltage, source_index
    ):
        """Source-to-detector 5×5 ray map, in x,y,theta_x,theta_y,1 order."""
        distance = jnp.stack([source.upstream_distance for source in self.sources])[
            source_index
        ]
        start = -self.condenser.extent - distance
        return column_matrix(
            (
                *self.condenser_components(condenser_controls, voltage),
                *self.projector_components(projector_controls, voltage),
            ),
            source_z=start,
        )

    def image(
        self,
        condenser_controls,
        projector_controls,
        voltage,
        source_index,
        specimen,
        numerical,
        *,
        dose=1.0,
        background=0.0,
        tilt_xy=(0.0, 0.0),
        shift_xy=(0.0, 0.0),
        method="scan",
    ):
        """Coherent image with aperture retracted; incoherent sum between source modes."""
        upstream = self.condenser_components(condenser_controls, voltage)
        downstream = self.projector_components(projector_controls, voltage)
        detector = CoherentDetector(
            numerical.detector_shape,
            numerical.detector_pitch,
            downstream[-1].z,
            numerical.pixel_samples,
            numerical.detector_psf,
        )
        image = jnp.zeros(numerical.detector_shape)
        for source, weight in zip(
            self.source_modes(source_index, voltage, numerical),
            numerical.angular_weights,
        ):
            at_sample = run_to_end(
                source,
                upstream,
                propagator=FreeSpacePropagator(include_axial_path=False),
            )
            at_sample = at_sample.derive(
                x=at_sample.x + shift_xy[0],
                y=at_sample.y + shift_xy[1],
                dx=at_sample.dx + tilt_xy[0],
                dy=at_sample.dy + tilt_xy[1],
            )
            resolved = numerical.resolve(at_sample)
            image = image + weight * microscope_image(
                resolved, (specimen, *downstream), detector, method=method
            )
        return dose * image + background

    def current_fraction(
        self, condenser_controls, voltage, source_index, hole, numerical
    ):
        """Integral of |t_aperture * psi|² / unit incident flux, over the aperture plane.

        No sum of individual beamlet powers replaces a coherent field sum.
        Radial/azimuthal quadrature is independent of the image detector crop.
        """
        aperture = self.aperture_component(condenser_controls, voltage, hole)
        upstream = self.condenser_components(condenser_controls, voltage)[:2]
        radius = aperture.radius + 16 * aperture.edge_width
        r = radius * numerical.radial_nodes
        theta = numerical.azimuthal_nodes
        xx = aperture.x0 + r[:, None] * jnp.cos(theta)
        yy = aperture.y0 + r[:, None] * jnp.sin(theta)
        points = jnp.stack((xx.ravel(), yy.ravel()), axis=-1)
        from types import SimpleNamespace

        grid = SimpleNamespace(shape=xx.shape, coords=points)
        transmission2 = jnp.exp(2 * aperture.log_transmission(points)).reshape(xx.shape)
        weights = radius**2 * numerical.radial_weights * numerical.radial_nodes
        result = 0.0
        for source, weight in zip(
            self.source_modes(source_index, voltage, numerical),
            numerical.angular_weights,
        ):
            incident = run_to_end(
                source,
                (*upstream, Plane(z=aperture.z)),
                propagator=FreeSpacePropagator(include_axial_path=False),
            )
            field = evaluate_gaussians(incident, grid, method="scan")
            result += (
                weight
                * jnp.sum(jnp.abs(field) ** 2 * transmission2 * weights[:, None])
                * (2 * jnp.pi / len(theta))
            )
        return result


@dataclass(frozen=True)
class WaveNumerics:
    """Fixed quadrature sizes; always refine separately from fitted parameters."""

    specimen_half_width: float = 80e-9
    beamlet_width: float = 1e-9
    overlap: float = 1.5
    detector_shape: tuple = (40, 40)
    detector_pitch: float = 110e-6
    pixel_samples: int = 2
    detector_psf: float = 0.0
    follow_source: bool = False
    angular_order: int = 1
    radial_order: int = 96
    azimuthal_order: int = 64

    def __post_init__(self):
        if (
            min(
                self.specimen_half_width,
                self.beamlet_width,
                self.overlap,
                self.detector_pitch,
            )
            <= 0
        ):
            raise ValueError("Numerical lengths and overlap must be positive.")
        if (
            min(
                self.angular_order,
                self.radial_order,
                self.azimuthal_order,
                self.pixel_samples,
            )
            < 1
        ):
            raise ValueError("Quadrature orders must be positive.")
        if self.follow_source and 4 * self.beamlet_width >= self.specimen_half_width:
            raise ValueError(
                "A source-following beamlet must be narrower than the source waist."
            )
        step = self.beamlet_width / self.overlap
        half = int(np.ceil(self.specimen_half_width / step))
        axis = jnp.arange(-half, half + 1) * step
        xx, yy = jnp.meshgrid(axis, axis)
        object.__setattr__(
            self, "centres", jnp.stack((xx.ravel(), yy.ravel()), axis=-1)
        )
        object.__setattr__(self, "step", step)
        nodes, weights = np.polynomial.hermite.hermgauss(self.angular_order)
        xx, yy = np.meshgrid(nodes * np.sqrt(2), nodes * np.sqrt(2))
        object.__setattr__(
            self,
            "angular_nodes",
            jnp.asarray(np.stack((xx.ravel(), yy.ravel()), axis=-1)),
        )
        object.__setattr__(
            self,
            "angular_weights",
            jnp.asarray(np.outer(weights, weights).ravel() / np.pi),
        )
        nodes, weights = np.polynomial.legendre.leggauss(self.radial_order)
        object.__setattr__(self, "radial_nodes", jnp.asarray((nodes + 1) / 2))
        object.__setattr__(self, "radial_weights", jnp.asarray(weights / 2))
        object.__setattr__(
            self,
            "azimuthal_nodes",
            jnp.arange(self.azimuthal_order) * (2 * jnp.pi / self.azimuthal_order),
        )

    def resolve(self, beam):
        """Resolve one analytic Gaussian into narrower, phase-matched Gaussians.

        Gaussian convolution deconvolves the envelope; the common quadratic
        phase is preserved exactly. Beam width must exceed basis width on both
        axes. The finite lattice must cover the illuminating field. follow_source moves
        and scales its footprint with the beam, at a fixed number of Gaussians.
        """
        precision = beam.k * jnp.imag(beam.Q_inv) / 2
        envelope = jnp.linalg.inv(precision)
        # Optional envelope-following lattice retains the whole beam as unknown
        # optics change its size/position. The number of beams stays static for
        # JAX; refine the ratio half_width/beamlet_width to check sampling.
        radius = jnp.sqrt(jnp.trace(envelope) / 2)
        scale = radius * 4 / self.specimen_half_width if self.follow_source else 1.0
        width = self.beamlet_width * scale
        centres = (
            beam.r_xy + self.centres * scale if self.follow_source else self.centres
        )
        step = self.step * scale
        spread = envelope - width**2 * jnp.eye(2)
        delta = centres - beam.r_xy
        exponent = jnp.einsum("ni,ij,nj->n", delta, jnp.linalg.inv(spread), delta)
        factor = jnp.sqrt(jnp.linalg.det(envelope) / jnp.linalg.det(spread))
        amplitude = (
            beam.amplitude * factor * step**2 / (jnp.pi * width**2) * jnp.exp(-exponent)
        )
        phase_q = jnp.real(beam.Q_inv)
        slope = beam.d_xy + delta @ phase_q.T
        path = (
            beam.pathlength
            + delta @ beam.d_xy
            + 0.5 * jnp.einsum("ni,ij,nj->n", delta, phase_q, delta)
        )
        valid = jnp.all(jnp.linalg.eigvalsh(spread) > 0)
        return GaussianBeam(
            x=centres[:, 0],
            y=centres[:, 1],
            dx=slope[:, 0],
            dy=slope[:, 1],
            z=jnp.broadcast_to(beam.z, path.shape),
            voltage=jnp.broadcast_to(beam.voltage, path.shape),
            amplitude=jnp.where(valid, amplitude, jnp.nan + 0j),
            pathlength=path,
            Q_inv=jnp.broadcast_to(
                phase_q + 2j / (beam.k * width**2) * jnp.eye(2), (len(path), 2, 2)
            ),
        )


def microscope_from_dict(values):
    """Load human-readable SI geometry/source values; no hidden generating state."""

    def train(record):
        return FiniteLensTrain(
            *(jnp.asarray(record[key]) for key in ("gaps", "lengths", "strengths"))
        )

    aperture = dict(values["aperture"])
    aperture["centres_xy"] = jnp.asarray(aperture["centres_xy"])
    aperture["radii"] = jnp.asarray(aperture["radii"])
    model = CoherentMicroscope(
        train(values["condenser"]),
        train(values["projector"]),
        tuple(GaussianSourceState(**row) for row in values["sources"]),
        ApertureGeometry(**aperture),
        values.get("reference_voltage", 200000.0),
    )
    if not all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(model)):
        raise ValueError("Microscope parameters must be finite.")
    if model.reference_voltage <= 0:
        raise ValueError("Reference voltage must be positive.")
    if np.shape(model.condenser.lengths) != (4,) or np.shape(
        model.projector.lengths
    ) != (5,):
        raise ValueError("Supply four condenser and five projector lenses.")
    for lens_train, n in ((model.condenser, 4), (model.projector, 5)):
        if np.shape(lens_train.gaps) != (n + 1,) or np.shape(lens_train.strengths) != (
            n,
        ):
            raise ValueError("Lens trains require N+1 gaps and N gains.")
        if not np.all(np.asarray(lens_train.lengths) > 0) or not np.all(
            np.asarray(lens_train.gaps) >= 0
        ):
            raise ValueError("Lengths must be positive and gaps nonnegative.")
    if float(model.condenser.gaps[0]) != 0:
        raise ValueError(
            "Condenser gap[0] must be zero; source distance belongs to each source state."
        )
    if not 0 < model.aperture.gap_fraction < 1:
        raise ValueError("Aperture fraction must lie inside the CL3-to-mini gap.")
    if (
        not np.all(np.asarray(model.aperture.radii) > 0)
        or model.aperture.edge_width <= 0
    ):
        raise ValueError("Aperture sizes must be positive.")
    if np.shape(model.aperture.centres_xy) != (len(model.aperture.radii), 2):
        raise ValueError("Each aperture needs an x/y centre.")
    if not model.sources or any(
        s.upstream_distance <= 0 or s.rms_size <= 0 or s.additional_angular_rms < 0
        for s in model.sources
    ):
        raise ValueError(
            "Source distance/size must be positive and angular spread nonnegative."
        )
    return model


def microscope_to_dict(model):
    from dataclasses import asdict

    def ordinary(value):
        if isinstance(value, dict):
            return {k: ordinary(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [ordinary(v) for v in value]
        return np.asarray(value).tolist()

    return ordinary(asdict(model))
