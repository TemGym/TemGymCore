"""Named native components, current tables, and an explicit imaging solve.

Components own the physics. Tables contain amperes and only supply settings;
loading a microscope or selecting a table never runs an optimisation.
"""

import tomllib
from dataclasses import dataclass, field, replace

import numpy as np

from .components import Component, Detector, DoubleDeflector, ElectromagneticLens, Plane
from .config import require_x64
from .constants import effective_accelerating_potential
from .run import column_matrix


@dataclass(frozen=True)
class CurrentTable:
    """Rows of lens currents [A] at increasing control values (e.g. magnification)."""

    lens_names: tuple[str, ...]
    control_values: np.ndarray
    currents: np.ndarray

    def __post_init__(self):
        names = tuple(self.lens_names)
        controls = np.asarray(self.control_values, dtype=float)
        currents = np.asarray(self.currents, dtype=float)
        if (not names or len(set(names)) != len(names)
                or controls.ndim != 1 or controls.size == 0
                or currents.shape != (controls.size, len(names))
                or not np.all(np.isfinite(controls))
                or not np.all(np.isfinite(currents))
                or np.any(np.diff(controls) <= 0)):
            raise ValueError("Use distinct lens names, increasing finite controls, and one current row per control.")
        object.__setattr__(self, "lens_names", names)
        object.__setattr__(self, "control_values", controls)
        object.__setattr__(self, "currents", currents)

    def at(self, value):
        """Interpolate named currents; reject extrapolation and nonfinite controls."""
        if not np.isfinite(value) or not self.control_values[0] <= value <= self.control_values[-1]:
            raise ValueError("Control value is outside the table range.")
        return {name: float(np.interp(value, self.control_values, self.currents[:, i]))
                for i, name in enumerate(self.lens_names)}


@dataclass(frozen=True)
class Microscope:
    """A named collection of TemGym components at a nominal voltage [V].

    ``column`` supplies the components in propagation order. ``with_currents``
    and ``with_components`` return updated microscopes, leaving the original
    intact. Column ordering is host-side; current and non-position parameters
    can be JAX tracers. Put the voltage on the source beam when propagating.
    """

    components: dict[str, Component]
    voltage: float = 200_000.
    tables: dict[str, CurrentTable] = field(default_factory=dict)
    config: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not np.isfinite(self.voltage) or self.voltage <= 0:
            raise ValueError("Voltage must be finite and positive.")
        if not all(isinstance(c, Component) for c in self.components.values()):
            raise TypeError("Microscope components must be native TemGym components.")
        for table in self.tables.values():
            if any(not isinstance(self.components.get(name), ElectromagneticLens)
                   for name in table.lens_names):
                raise ValueError("Current tables must name existing ElectromagneticLens components.")
        object.__setattr__(self, "components", dict(self.components))

    @property
    def column(self):
        """Native component tuple, sorted by entrance z."""
        return tuple(sorted(self.components.values(), key=lambda c: float(c.z)))

    def with_components(self, **components):
        """Replace or add named components, including samples and apertures."""
        return replace(self, components={**self.components, **components})

    def with_currents(self, **currents):
        """Replace the specified lens currents [A]; unknown names raise KeyError."""
        return self.with_components(**{
            name: replace(self.components[name], current=current)
            for name, current in currents.items()
        })

    def with_settings(self, **table_values):
        """Apply interpolated current tables; conflicting lens settings are errors."""
        currents = {}
        for name, value in table_values.items():
            for lens, current in self.tables[name].at(value).items():
                if lens in currents and currents[lens] != current:
                    raise ValueError(f"Tables specify conflicting currents for {lens}.")
                currents[lens] = current
        return self.with_currents(**currents)

    @classmethod
    def from_toml(cls, path):
        """Load the example microscope format, converting lens Gc once.

        TOML Gc is referenced to ``beam.voltage_kV``; native Gc includes the
        effective accelerating potential. Components then handle beam-voltage
        changes themselves. No extra empirical thickness scaling is applied.

        The sample starts as a Plane. Aperture choices and source/specimen
        recipes remain in ``config`` (the original TOML dictionary); choose
        an aperture radius or a specimen explicitly with ``with_components``.
        Deflectors start at zero kick; their native controls are angles [rad].
        """
        with open(path, "rb") as stream:
            config = tomllib.load(stream)
        voltage = config["beam"]["voltage_kV"] * 1000.
        potential = float(effective_accelerating_potential(voltage))
        components = {
            name: ElectromagneticLens(
                **{k: v for k, v in spec.items() if k not in ("z_m", "Gc")},
                z=spec["z_m"], Gc=spec["Gc"] * potential,
                **({} if "current" in spec else {"current": 0.}),
            ) for name, spec in config.get("lenses", {}).items()
        }
        for name, spec in config.get("deflectors", {}).items():
            components[name] = DoubleDeflector(
                z=spec["z_m"], spacing=spec["spacing_m"],
                **{k: v for k, v in spec.items() if k.endswith("_balance_x") or k.endswith("_balance_y")},
            )
        if "sample" in config:
            components["sample"] = Plane(z=config["sample"]["z_m"])
        if "detector" in config:
            detector = config["detector"]
            components["detector"] = Detector(
                z=detector["z_m"], shape=tuple(detector["shape"]),
                pixel_size=(detector["pixel_size_m"],) * 2,
            )
        tables = {}
        for group_name, group in config.get("modes", {}).items():
            for name, table in ({group_name: group} if "headers" in group else group).items():
                if name in tables:
                    raise ValueError(f"Duplicate current-table name: {name}")
                values = np.asarray(table["values"], dtype=float)
                tables[name] = CurrentTable(tuple(table["headers"][1:]), values[:, 0], values[:, 1:])
        return cls(components, voltage, tables, config)


@dataclass(frozen=True)
class ImagingSolution:
    """Focused microscope, local 5×5 matrix and scaled imaging residuals.

    Failed solves raise ValueError. A returned solution satisfies the requested
    A, B and optional reference-image position tolerances.
    """

    microscope: Microscope
    matrix: np.ndarray
    residual: np.ndarray


def find_imaging_condition(microscope, *, magnification, vary, voltage=None,
                           object_plane="sample", image_plane="detector",
                           rotation=0., bounds=(0.01, 5.),
                           reference=(0., 0., 0., 0.), image_position=None):
    """Adjust native component parameters to A = magnification × R, B = 0.

    ``vary`` names scalar controls: "IL1" means "IL1.current", while
    "solenoid.q" or "lens.focal_length" select other native parameters.
    Bounds are a lower/upper pair of scalars or arrays in those parameters'
    units. Component entrance positions and the two measurement planes stay
    fixed. Supply physically valid bounds for finite component bodies.

    ``reference`` is the input (x,y,theta_x,theta_y). Optionally constrain its
    image to ``image_position=(x,y)`` [m]. The matrix is local for nonlinear
    optics; this solve does not constrain wave transmission or aberrations
    away from the reference ray. Not every control set can meet every target.

    JAX differentiates native ray propagation to obtain the matrix, then
    differentiates its residuals for the shared bounded least-squares solver.
    Enable float64. Returned errors satisfy |A/M - R| < 1e-9, |B| < 1e-8 m
    and, when requested, reference-image position error < 1e-9 m.
    """
    import jax.numpy as jnp
    from .optimization import least_squares

    require_x64("find_imaging_condition")
    voltage = microscope.voltage if voltage is None else voltage
    if not np.all(np.isfinite((magnification, rotation, voltage))) or magnification <= 0 or voltage <= 0:
        raise ValueError("Magnification and voltage must be positive and rotation finite.")
    reference = np.asarray(reference, dtype=float)
    if reference.shape != (4,) or not np.all(np.isfinite(reference)):
        raise ValueError("reference must contain four finite ray coordinates.")
    if image_position is not None:
        image_position = np.asarray(image_position, dtype=float)
        if image_position.shape != (2,) or not np.all(np.isfinite(image_position)):
            raise ValueError("image_position must contain two finite positions in metres.")
    start = float(microscope.components[object_plane].z)
    end = float(microscope.components[image_plane].z)
    if not np.isfinite(end - start) or end <= start:
        raise ValueError("The image plane must be downstream of the object plane.")
    names = [name for name, c in sorted(microscope.components.items(), key=lambda item: float(item[1].z))
             if start < float(c.z) < end]
    components = tuple(microscope.components[name] for name in names) + (Plane(z=end),)
    controls = []
    for entry in vary:
        name, _, attribute = entry.partition(".")
        attribute = attribute or "current"
        if name not in names or attribute in ("z", "z_po"):
            raise ValueError("Vary controls between the measurement planes; entrance z positions are fixed.")
        control = (names.index(name), attribute)
        if control in controls:
            raise ValueError(f"Repeated control: {entry}")
        controls.append(control)
    if not controls:
        raise ValueError("Supply at least one adjustable control.")
    initial = np.array([getattr(components[i], attribute) for i, attribute in controls], dtype=float)
    if initial.shape != (len(controls),) or not np.all(np.isfinite(initial)):
        raise ValueError("Adjustable controls must be finite scalars.")
    lower, upper = (np.broadcast_to(np.asarray(b, dtype=float), initial.shape) for b in bounds)
    if not np.all(np.isfinite((lower, upper))) or np.any(lower >= upper):
        raise ValueError("Bounds must be finite and ordered.")

    def updated_column(parameters):
        column = list(components)
        for (i, attribute), value in zip(controls, parameters):
            column[i] = replace(column[i], **{attribute: value})
        return tuple(column)

    def matrix_at(parameters):
        return column_matrix(updated_column(parameters), source_z=start,
                             voltage=voltage, reference=reference)

    rotation_matrix = np.array(((np.cos(rotation), -np.sin(rotation)), (np.sin(rotation), np.cos(rotation))))
    target = np.c_[magnification * rotation_matrix, np.zeros((2, 2))]
    # Express the residual in coordinates at the first adjustable component.
    # The fixed upstream optics otherwise make A and B almost dependent at
    # high magnification. This is a constant change of basis, not another model.
    first = min(i for i, _ in controls)
    prefix = components[:first] + (Plane(z=components[first].z),)
    input_basis = np.linalg.inv(np.asarray(column_matrix(
        prefix, source_z=start, voltage=voltage, reference=reference))[:4, :4])
    scales = np.maximum(np.linalg.norm(target @ input_basis, axis=0), 1.)

    def residual(parameters):
        matrix = matrix_at(parameters)
        errors = ((matrix[:2, :4] - target) @ input_basis / scales).ravel()
        if image_position is not None:
            position = matrix[:2, :4] @ reference + matrix[:2, 4]
            errors = jnp.concatenate((errors, (position - image_position) / 1e-3))
        return errors

    result = least_squares(residual, np.clip(initial, lower, upper), bounds=(lower, upper),
                           x_scale="jac", ftol=1e-14, xtol=1e-14, gtol=1e-14, max_iter=300)
    matrix = np.asarray(matrix_at(result.x))
    position = matrix[:2, :4] @ reference + matrix[:2, 4]
    if (not result.success or not np.all(np.isfinite(matrix))
            or np.max(np.abs(matrix[:2, :2] / magnification - rotation_matrix)) > 1e-9
            or np.max(np.abs(matrix[:2, 2:4])) > 1e-8
            or (image_position is not None and np.max(np.abs(position - image_position)) > 1e-9)):
        raise ValueError(f"No converged imaging solution at {magnification:g}x: {result.message}")
    column = updated_column(result.x)
    focused = microscope.with_components(**{names[i]: column[i] for i, _ in controls})
    return ImagingSolution(replace(focused, voltage=voltage), matrix, np.asarray(result.residual))
