"""The condenser of the simulated microscope in TEM mode: from the electron source to the specimen.

In order down the column (``z`` in metres, the specimen at 0)::

    source -> CL1 -> CL3 -> condenser aperture -> CLA1 -> CLA2 -> [corrector] -> CM -> objective front field -> specimen

* The **source** is a round Gaussian beam: rms size and rms spread of angles, and the current it
  carries. It stands for the gun and its optics, which are not modelled.
* **CL1** (spot size) makes the first crossover; **CL3** (brightness) makes the second. Each lens is a
  TemGym :class:`~temgym_core.components.ElectromagneticLens`, so it focuses as 1/(ampere-turns)^2,
  turns the beam in proportion to its ampere-turns, and follows the high tension by itself.
* The **condenser aperture** (TemGym :class:`~temgym_core.components.SigmoidAperture`) has four
  holes; hole 0 means no aperture. Its position follows the Apt3 DAC values about 2048.
* **CLA1** (beam shift) and **CLA2** (beam tilt) are two TemGym
  :class:`~temgym_core.components.Deflector` planes after the aperture, so beam shift and tilt do not
  change how much the aperture cuts. The console's DAC values give a shift and tilt at the specimen
  (``Electronics``); the two deflectors' kicks are solved, through CM and the front field, to give
  exactly that shift and tilt.
* The **probe corrector**, if there is one, is only its round optics: a transfer doublet, two thin TemGym
  :class:`~temgym_core.components.Lens`es a quarter and three quarters of the way along it, so the beam leaves it as it
  entered, turned over (magnification -1). Its multipoles (the aberration correction) are not modelled.
* **CM** (condenser mini lens) and the **objective's front field** are fixed in TEM mode. The front
  field is the objective's own lens, mirrored to the source side of the specimen, at the objective's
  current.

How the beam is set (TEM mode): the beam is parallel at the specimen when CL3 puts its crossover in
the plane that CM images to the front field's front focal plane, and focused on the specimen when
CL3 puts its crossover in CM's front focal plane (CM then sends it on parallel, and the front field
focuses it in its back focal plane, at the specimen).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..components import Deflector, ElectromagneticLens, Lens, Plane, SigmoidAperture
from ..ray import Ray
from ..run import run_to_end
from .electronics import NEUTRAL_DAC

#: The lenses the condenser drives, in order down the column.
CONDENSER_LENSES = ("CL1", "CL3", "CM")
#: Apt3 position DAC value of an aperture on the axis.
APERTURE_CENTRE_DAC = 2048


@dataclass(frozen=True)
class Condenser:
    """The TEM-mode condenser, from the source to the specimen.

    Parameters
    ----------
    source_z_m, source_rms_m, source_rms_angle_rad, source_current_a : float
        Where the source is, its rms size and rms spread of angles (round), and the current it carries.
    lenses : dict
        ``CL1``, ``CL3`` and ``CM``: each ``{"z_m", "ampere_turns_per_dac", "Gc", "Tc"}``
        (:class:`~temgym_core.components.ElectromagneticLens`).
    front_field : dict
        The objective's front field: ``{"z_m", "ampere_turns", "Gc", "Tc"}``, where it ends and the objective's
        ampere-turns per unit of its control.
    aperture_z_m : float
        Plane of the condenser aperture.
    hole_diameters_m : tuple of float
        Diameters of holes 1 to 4.
    aperture_m_per_dac : float
        How far the aperture moves per Apt3 position DAC unit.
    aperture_edge_m : float
        Width of the aperture's soft edge.
    deflector_z_m : tuple of float
        Planes of CLA1 and CLA2.
    spot_table : dict
        Spot-size index -> ``{"CL1": DAC value, "CL3": DAC value}``: CL1 sets the spot and CL3 follows it, so the
        second crossover stays where it was.
    cm_dac : int
        CM's DAC value in TEM mode.
    corrector : dict, optional
        ``{"entrance_z_m", "exit_z_m"}`` of the probe corrector (round optics only), between CLA2 and CM.
    """

    source_z_m: float
    source_rms_m: float
    source_rms_angle_rad: float
    source_current_a: float
    lenses: dict
    front_field: dict
    aperture_z_m: float
    hole_diameters_m: tuple
    aperture_m_per_dac: float
    aperture_edge_m: float
    deflector_z_m: tuple
    spot_table: dict
    cm_dac: int
    corrector: dict = None

    @classmethod
    def from_dict(cls, d, front_field, *, dac_scale=1.0):
        """A condenser from a file's ``condenser`` entry and the objective's front field.

        ``dac_scale`` (p(HT) / p(reference)) scales the console's DAC values, so the optics at another high
        tension are those at the reference.
        """
        return cls(source_z_m=d["source"]["z_m"], source_rms_m=d["source"]["rms_size_m"],
                   source_rms_angle_rad=d["source"]["rms_angle_rad"], source_current_a=d["source"]["current_a"],
                   lenses={n: dict(d["lenses"][n]) for n in CONDENSER_LENSES}, front_field=dict(front_field),
                   aperture_z_m=d["aperture"]["z_m"], hole_diameters_m=tuple(d["aperture"]["hole_diameters_m"]),
                   aperture_m_per_dac=d["aperture"]["m_per_dac"], aperture_edge_m=d["aperture"]["edge_m"],
                   deflector_z_m=tuple(d["deflectors_z_m"]),
                   spot_table={int(s): {n: round(v * dac_scale) for n, v in row.items()}
                               for s, row in d["spot_table"].items()},
                   cm_dac=round(d["cm_dac"] * dac_scale), corrector=d.get("corrector"))

    def source(self):
        """The beam at the source: plane, mean and covariance of ``[x, y, theta_x, theta_y]``."""
        s, a = self.source_rms_m, self.source_rms_angle_rad
        return self.source_z_m, np.zeros(4), np.diag([s * s, s * s, a * a, a * a])

    def lens(self, name, dac):
        """One condenser lens at a DAC value."""
        e = self.lenses[name]
        return ElectromagneticLens(z=e["z_m"], turns=e["ampere_turns_per_dac"], current=float(dac), Gc=e["Gc"],
                                   Tc=e.get("Tc", 0.0))

    def front_field_lens(self, objective_control):
        """The objective's front field at the objective's control. ``front_field["z_m"]`` is where it ends (the
        objective's start, mirrored), so a thick front field ends there."""
        e = self.front_field
        turns, tc = e["ampere_turns"] * float(objective_control), e.get("Tc", 0.0)
        return ElectromagneticLens(z=e["z_m"] - max(tc, 0.0) * abs(turns), turns=1.0, current=turns, Gc=e["Gc"], Tc=tc)

    def corrector_lenses(self):
        """The corrector's transfer doublet: the beam leaves it as it entered, turned over (none without a corrector)."""
        if self.corrector is None:
            return ()
        start, end = self.corrector["entrance_z_m"], self.corrector["exit_z_m"]
        f = (end - start) / 4
        return (Lens(z=start + f, focal_length=f), Lens(z=end - f, focal_length=f))

    def aperture(self, hole, position_dac):
        """The condenser aperture: hole 1-4 at an Apt3 position, or ``None`` for hole 0 (no aperture)."""
        if hole == 0:
            return None
        x, y = ((np.asarray(position_dac, float) - APERTURE_CENTRE_DAC) * self.aperture_m_per_dac)
        return SigmoidAperture(z=self.aperture_z_m, radius=self.hole_diameters_m[hole - 1] / 2,
                               edge_width=self.aperture_edge_m, x0=float(x), y0=float(y))

    def deflectors(self, specimen_kick, after, voltage_v):
        """CLA1 and CLA2 kicks that give ``specimen_kick`` (x, y, theta_x, theta_y) at the specimen through the
        components ``after`` them (the corrector, CM and the front field)."""
        z1, z2 = self.deflector_z_m

        def at_specimen(kicks):
            ray = Ray(x=0.0, y=0.0, dx=0.0, dy=0.0, z=z1, pathlength=0.0, voltage=voltage_v)
            column = (Deflector(z=z1, def_x=kicks[0], def_y=kicks[1]),
                      Deflector(z=z2, def_x=kicks[2], def_y=kicks[3]), *after, Plane(z=0.0))
            out = run_to_end(ray, column)
            return jnp.stack((out.x, out.y, out.dx, out.dy))

        response = np.asarray(jax.jacfwd(at_specimen)(jnp.zeros(4)))
        k = np.linalg.solve(response, np.asarray(specimen_kick, float))
        return (Deflector(z=z1, def_x=float(k[0]), def_y=float(k[1])),
                Deflector(z=z2, def_x=float(k[2]), def_y=float(k[3])))

    def components(self, dac, *, hole, aperture_position_dac, specimen_kick, objective_control, voltage_v):
        """The condenser as TemGym components, from the source to just before the specimen.

        Parameters
        ----------
        dac : dict
            DAC values of ``CL1``, ``CL3`` and ``CM``.
        hole, aperture_position_dac : int, tuple of int
            The condenser aperture's hole (0 = none) and its Apt3 position.
        specimen_kick : tuple of float
            Shift (m) and tilt (rad) at the specimen asked for by CLA1 and CLA2.
        objective_control : float
            The objective's control (its front field's).
        voltage_v : float
            High tension.
        """
        after = (*self.corrector_lenses(), self.lens("CM", dac["CM"]), self.front_field_lens(objective_control))
        aperture = self.aperture(hole, aperture_position_dac)
        return (self.lens("CL1", dac["CL1"]), self.lens("CL3", dac["CL3"]),
                *(() if aperture is None else (aperture,)),
                *self.deflectors(specimen_kick, after, voltage_v), *after)

    def spot_dac(self, spot, name):
        """CL1 or CL3's DAC value from the spot table (``None`` if the spot is not in it)."""
        row = self.spot_table.get(int(spot))
        return None if row is None else row[name]

    def table_dac(self, spot, name):
        """A condenser lens's console DAC value: CL1 and CL3 from the spot table, CM fixed."""
        if name == "CM":
            return self.cm_dac
        value = self.spot_dac(spot, name)
        return NEUTRAL_DAC if value is None else value
