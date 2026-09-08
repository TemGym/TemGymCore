"""Named native components, current tables, and an explicit imaging solve.

Components own the physics. Tables contain amperes and only supply settings;
loading a microscope or selecting a table never runs an optimisation.
"""

from dataclasses import dataclass, field, replace
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import numpy as np

from .components import Component, Detector, DoubleDeflector, ElectromagneticLens, Plane
from .constants import compute_Rc_from_voltage, effective_accelerating_potential
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
    """Focused microscope, native 5×5 matrix and scaled three-equation residual.

    Failed solves raise ValueError; a returned solution has passed both the
    conjugate-plane residual and an independent native-component matrix check.
    """

    microscope: Microscope
    matrix: np.ndarray
    residual: np.ndarray


def _radial_chain(positions, powers, start_z, end_z):
    """Thin, round map in the rotating frame, in physical propagation order."""
    matrix, previous_z = np.eye(2), start_z
    for z, power in zip(positions, powers, strict=True):
        matrix = np.array(((1., 0.), (-power, 1.))) @ np.array(((1., z - previous_z), (0., 1.))) @ matrix
        previous_z = z
    return np.array(((1., end_z - previous_z), (0., 1.))) @ matrix


def _imaging_geometry(microscope, object_plane, image_plane, voltage):
    start = float(microscope.components[object_plane].z)
    end = float(microscope.components[image_plane].z)
    downstream = {name: c for name, c in sorted(microscope.components.items(), key=lambda item: float(item[1].z))
                  if start < float(c.z) < end}
    lenses = tuple(downstream.values())
    if len(lenses) != 5 or any(not isinstance(c, ElectromagneticLens) for c in lenses):
        raise ValueError("Imaging solve requires exactly five downstream ElectromagneticLens components.")
    if any(c.Tc != 0 or c.stigmator_strength_x != 0 or c.stigmator_strength_y != 0
           or c.x0 != 0 or c.y0 != 0 for c in lenses):
        raise ValueError("Imaging solve supports only centred, thin, round lenses.")
    z = np.array([c.z for c in lenses])
    turns = np.array([c.turns for c in lenses])
    gains = np.array([c.Gc for c in lenses]) * turns**2 / float(effective_accelerating_potential(voltage))
    if np.any(np.diff(z) <= 0) or np.any(gains <= 0) or not np.all(np.isfinite(gains)):
        raise ValueError("Lens positions must increase and focal-power gains must be finite and positive.")
    currents = np.array([c.current for c in lenses])
    powers = gains * currents**2
    u, v = z[0] - start, end - z[-1]
    denominators = (powers[0] - 1 / u, powers[-1] - 1 / v)
    if min(denominators) <= 0:
        raise ValueError("The fixed objective and projector must have real conjugate planes.")
    objective_v, projector_u = 1 / np.asarray(denominators)
    object_z, image_z = z[0] + objective_v, z[-1] - projector_u
    if not z[0] < object_z < z[1] or not z[-2] < image_z < z[-1]:
        raise ValueError("Fixed conjugate planes must lie outside the three-lens relay.")
    return dict(names=tuple(downstream), z=z, turns=turns, gains=gains, currents=currents,
                sample_z=start, detector_z=end, objective_image_z=object_z,
                projector_object_z=image_z, objective_magnification=-objective_v / u,
                projector_magnification=-v / projector_u)


def find_imaging_condition(microscope, *, magnification, vary, voltage=None,
                           object_plane="sample", image_plane="detector",
                           rotation=0., current_bounds=(0.01, 5.)):
    """Solve A = magnification × R(rotation), B = 0; only ``vary`` currents move.

    This local design solve supports a centred, thin, round five-lens column:
    a fixed objective, three adjustable relay lenses, and a fixed projector.
    Supply an initial setting (usually a current-table row). ``vary`` must name
    the three relay lenses in axial order. Rotation is in radians and bounds
    are in amperes. Enable JAX float64 before loading the microscope.

    The small NumPy/SciPy solve retains the intermediate conjugate planes to
    avoid cancellation at high magnification. The returned lab-frame matrix
    is checked through TemGym's native components. This is operating-condition
    design; it does not infer microscope geometry from measured images.
    """
    from scipy.optimize import least_squares
    import jax

    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 for the high-magnification imaging solve.")
    voltage = microscope.voltage if voltage is None else voltage
    if not np.all(np.isfinite((magnification, rotation, voltage))) or magnification <= 0 or voltage <= 0:
        raise ValueError("Magnification and voltage must be positive and rotation finite.")
    lower, upper = (np.broadcast_to(np.asarray(b, dtype=float), (3,)) for b in current_bounds)
    if not np.all(np.isfinite((lower, upper))) or np.any(lower >= upper):
        raise ValueError("Current bounds must be finite and ordered.")
    g = _imaging_geometry(microscope, object_plane, image_plane, voltage)
    if tuple(vary) != g["names"][1:-1]:
        raise ValueError("vary must name the three interior downstream lenses in axial order.")
    relay_target = magnification / (g["objective_magnification"] * g["projector_magnification"])
    target_ampturns = rotation / float(compute_Rc_from_voltage(voltage))
    fixed_ampturns = g["turns"][[0, -1]] @ g["currents"][[0, -1]]
    length_scale = g["objective_image_z"] - g["z"][0]

    def residual(currents):
        matrix = _radial_chain(g["z"][1:-1], g["gains"][1:-1] * currents**2,
                               g["objective_image_z"], g["projector_object_z"])
        return np.array((matrix[0, 0] / relay_target - 1., matrix[0, 1] / length_scale,
                         (g["turns"][1:-1] @ currents + fixed_ampturns - target_ampturns)
                         / max(abs(fixed_ampturns), 1.)))

    result = least_squares(residual, np.clip(g["currents"][1:-1], lower, upper),
                           bounds=(lower, upper), xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=300)
    if not result.success or np.max(np.abs(result.fun)) > 1e-10:
        raise ValueError(f"No converged imaging solution at {magnification:g}x within the current bounds.")
    focused = replace(microscope.with_currents(**dict(zip(vary, result.x))), voltage=voltage)
    column = [c for c in focused.column if g["sample_z"] < float(c.z) < g["detector_z"]]
    matrix = np.asarray(column_matrix(column + [Plane(z=g["detector_z"])],
                                      source_z=g["sample_z"], voltage=voltage))
    rotation_matrix = np.array(((np.cos(rotation), -np.sin(rotation)), (np.sin(rotation), np.cos(rotation))))
    if (not np.all(np.isfinite(matrix))
            or np.max(np.abs(matrix[:2, :2] / magnification - rotation_matrix)) > 1e-9
            or np.max(np.abs(matrix[:2, 2:4])) > 1e-8):
        raise ValueError("Imaging solution failed the native component matrix check.")
    return ImagingSolution(focused, matrix, result.fun)
