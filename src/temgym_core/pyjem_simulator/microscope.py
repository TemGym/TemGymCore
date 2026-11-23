"""The simulated microscope: DAC values in, camera frames out, truth hidden.

:class:`SimulatedMicroscope` keeps the state of a JEOL-like TEM column in the
units its control interface uses (16-bit DAC values, volts, nanometres, indices). A
frame is rendered only when the optical model can justify the state: anything
unmodelled (another function mode, a moved unmodelled lens or deflector, an
untabulated high tension, an inserted objective aperture, ...) raises
:class:`UnmodelledStateError` listing every reason, instead of returning an
image the model cannot produce.

The beam starts either at the electron source, traced through a TEM-mode condenser
(:class:`~temgym_core.pyjem_simulator.condenser.Condenser`: CL1, CL3, the condenser aperture, CLA1 and CLA2, CM
and the objective's front field), so spot size, brightness and the condenser aperture act on it; or, without a
condenser, at the specimen plane (:class:`SpecimenBeam`), where CL3 (brightness) may set its rms size
(:attr:`OpticalTruth.beam_size`) and the other condenser controls change nothing.
The hidden optics after the specimen are any TemGym column (:class:`OpticalTruth`); images
come from TemGym's ray images (:func:`temgym_core.evaluate.ray_image`).
"""

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from scipy.constants import elementary_charge

from ..components import Deflector, Detector, Plane, Rotator, Scanner
from ..config import require_x64
from ..evaluate import ray_image
from ..source import sample_rays
from .camera import CameraModel
from .electronics import NEUTRAL_DAC, Electronics

#: PyJEM free-lens-control identifiers.
LENS_IDS = {"CL1": 0, "CL2": 1, "CL3": 2, "CM": 3, "OLc": 6, "OLf": 7, "OM": 8, "OM2": 9,
            "IL1": 10, "IL2": 11, "IL3": 12, "IL4": 13, "PL1": 14, "PL2": 15, "PL3": 16}
PROJECTOR_LENSES = ("IL1", "IL2", "IL3", "PL1")
#: The console's condenser lenses. With a condenser, CL1, CL3 and CM are modelled and CL2 (not on the NEOARM) must
#: keep its value; without one, all are accepted and only CL3 may act (through a beam-size law).
CONDENSER_LENSES = ("CL1", "CL2", "CL3", "CM")
OBJECTIVE_LENSES = ("OLc", "OLf")
UNMODELLED_LENSES = ("OM", "OM2", "IL4", "PL2", "PL3")
#: PyJEM Def3 deflector pairs; CLA1 (beam shift) and CLA2 (beam tilt) are modelled, PLA only if the optics give it a plane.
DEFLECTORS = ("GunA1", "GunA2", "CLA1", "CLA2", "CLs", "IS1", "IS2", "PLA", "OLs", "ILs",
              "SpotA", "ShifBal", "TiltBal", "AngBal")
TEM_FUNCTION_MODES = {0: "MAG", 1: "MAG2", 2: "LowMAG", 3: "SAMAG", 4: "DIFF"}
STEM_FUNCTION_MODES = {0: "Align", 1: "SM-LMAG", 2: "SM-MAG", 3: "AMAG", 4: "uuDIFF", 5: "Rocking"}
#: Apt3 kinds. The condenser aperture (kind 1) is modelled with a condenser (else accepted, changing nothing); the
#: others (after the specimen, or analytical) must stay open.
APERTURE_KINDS = {1: "CLA", 2: "OLA", 3: "HCA", 4: "SAA", 5: "ENTA", 6: "EDS"}
CONDENSER_APERTURE = 1
APERTURE_HOME = (2048, 2048)


def _run_compiled(bundle, components):
    """``run_to_end`` compiled once for a column's structure: the components are arguments, so a new lens
    current needs no new compilation. Run op by op, a 16 129-beamlet bundle took about 0.9 s through the
    simulated NEOARM's lower column (2026-11-22)."""
    import jax

    from ..run import run_to_end

    global _RUN_COMPILED
    if _RUN_COMPILED is None:
        _RUN_COMPILED = jax.jit(run_to_end)
    return _RUN_COMPILED(bundle, components)


_RUN_COMPILED = None


class UnmodelledStateError(RuntimeError):
    """The microscope state lies outside what the optical model can render.

    Parameters
    ----------
    reasons : list of str
        Every unmodelled aspect of the state.
    """

    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("The simulated microscope cannot render this state: " + "; ".join(self.reasons))


@dataclass(frozen=True, eq=False)
class OpticalTruth:
    """Hidden optics: the column after the specimen, and the modelled operating points.

    Parameters
    ----------
    column : callable
        ``column(controls, voltage_v)`` returns the TemGym components from the specimen to the
        camera for relative lens strengths ``controls`` (objective, IL1, IL2, IL3, PL1) at high
        tension ``voltage_v``.
    voltages_v : tuple of float
        Modelled operating high tensions.
    parameters : dict, optional
        What the column was built from, for scoring a fit only.
    pla_z_m : float, optional
        Plane (m from the specimen) of the projector alignment deflector PLA: a TemGym
        :class:`~temgym_core.components.Deflector` is put there, in front of the first component
        at or after it. Without it PLA is not modelled and must stay neutral.
    pla_kick : callable, optional
        ``pla_kick(dac_x, dac_y, voltage_v, diffraction)`` gives PLA's slope kick (rad, x and y)
        for its DAC values at a high tension, in DIFF or not. Needed with ``pla_z_m``.
    beam_size : callable, optional
        ``beam_size(cl3_dac)`` gives the beam's rms size (m) at the specimen for a CL3 DAC value; its
        angle spread stays the ``incident_beam``'s. Without it CL3 changes nothing. Not used with a condenser.
    condenser : Condenser, optional
        The TEM-mode condenser: the beam then starts at the source and goes through it.
    whole_column : callable, optional
        The hidden optics as one function of the whole console: ``whole_column(console, voltage_v)`` returns
        ``(components, start_z, mean, covariance)``, the TemGym components from the source to the camera and the
        beam where it starts (its mean and covariance in x, y, theta_x, theta_y). ``console`` is a dict with every
        lens's DAC value (as the lens acts: with its hysteresis lag) and every deflector's ``(x, y)`` DAC pair.
        With it, ``column`` (give ``None``), the deflector law, PLA, the beam-size law and the condenser are not
        used, and every lens and deflector may be changed.
    """

    column: object
    voltages_v: tuple
    parameters: dict = field(default_factory=dict)
    pla_z_m: float = None
    pla_kick: object = None
    beam_size: object = None
    condenser: object = None
    whole_column: object = None

    def __post_init__(self):
        object.__setattr__(self, "voltages_v", tuple(float(v) for v in self.voltages_v))

    def scoring_truth(self):
        """What the hidden column was built from, for scoring a fit only.

        Returns
        -------
        dict
        """
        return dict(self.parameters)


@dataclass(frozen=True)
class PhysicalSetting:
    """Physical controls of one frame: relative lens strengths and the kick at the specimen.

    Parameters
    ----------
    voltage_v : float
        High tension.
    projector_controls : tuple of float
        Relative Objective, IL1, IL2, IL3 and PL1 controls.
    specimen_kick : tuple of float
        Specimen displacement (m) and angle (rad).
    pla_kick : tuple of float, optional
        Slope kick (rad, x and y) of the projector alignment deflector PLA.
    cl3_dac : int, optional
        CL3's DAC value, when the optics give it a beam-size law (else ``None``).
    condenser_dac : tuple of (str, float), optional
        With a condenser: the (effective) DAC values of CL1, CL3 and CM.
    aperture : tuple of int, optional
        With a condenser: the condenser aperture's hole (0 = none) and its Apt3 position x, y.
    console : tuple of (str, value), optional
        With :attr:`OpticalTruth.whole_column`: every lens's (effective) DAC value, every deflector's DAC pair and
        ``CondenserAperture``: the condenser aperture's hole and its Apt3 position (x, y).
    """

    voltage_v: float
    projector_controls: tuple
    specimen_kick: tuple
    pla_kick: tuple = (0.0, 0.0)
    cl3_dac: int = None
    condenser_dac: tuple = None
    aperture: tuple = None
    console: tuple = None


@dataclass(frozen=True)
class SpecimenBeam:
    """The incident beam at the specimen plane: round and Gaussian, of rms size ``rms_size_m``
    and rms angle spread ``rms_angle_rad``, centred where the beam-shift and beam-tilt
    deflectors put it."""

    rms_size_m: float
    rms_angle_rad: float


@dataclass(frozen=True)
class LowerCamera:
    """A second camera further down the column than the first (the first camera is where the column ends).

    The beam reaches it by a straight flight from the first camera's plane, so it is seen only while every camera
    above it is retracted.

    Parameters
    ----------
    model : CameraModel
        The camera itself (its name is the detector's name).
    below_m : float
        How far below the first camera's plane it sits.
    rotation_rad : float, optional
        How far it is turned about the axis, relative to the first camera.
    centre_m : tuple of float, optional
        Where its middle pixel is (x, y, in the first camera's own x and y).
    """

    model: CameraModel
    below_m: float
    rotation_rad: float = 0.0
    centre_m: tuple = (0.0, 0.0)

    def __post_init__(self):
        if self.below_m <= 0:
            raise ValueError("A lower camera sits below the first one: below_m must be positive.")


class SimulatedMicroscope:
    """A JEOL-like TEM column driven by DAC values, with hidden optics.

    Parameters
    ----------
    truth : OpticalTruth
        Hidden optics.
    electronics : Electronics
        Hidden DAC-to-control laws.
    magnification_table : dict
        Nominal magnification -> DAC values for ``IL1``, ``IL2``, ``IL3`` and ``PL1``
        (the TEM MAG selector, in increasing magnification).
    incident_beam : SpecimenBeam
        The beam at the specimen plane (the attribute of the same name can be changed later,
        e.g. to spread the beam for a specimen image).
    diffraction_table : dict, optional
        Camera length (m, at this camera) -> DAC values for ``IL1``, ``IL2``, ``IL3`` and
        ``PL1`` (the TEM DIFF selector, in increasing camera length). Without it DIFF
        mode is refused as unmodelled.
    diffraction_camera_length_m : float, optional
        The camera length (m) the DIFF selector starts at, as a console left on it: the row
        nearest to it. By default the first (shortest) row.
    magnification : float, optional
        The magnification the MAG selector starts at: the row nearest to it. By default the first
        (lowest) row.
    condenser_dac : dict, optional
        DAC values the console holds for condenser lenses (``CL1``, ``CL2``, ``CL3``, ``CM``); others are
        32768.
    spot_table : dict, optional
        Spot-size index -> DAC values of the condenser lenses the console changes with the spot (e.g. ``CL1``,
        ``CL3``); they take the place of ``condenser_dac``'s at that spot. Not used with a condenser. A row may
        also hold a deflector's ``(x, y)`` DAC values, such as the console's spot alignment ``SpotA``: selecting
        the spot (:meth:`select_spot`) sets that deflector to them.
    camera : CameraModel, optional
        The camera; ideal by default.
    lower_cameras : sequence of LowerCamera, optional
        More cameras further down the column, each read by its own name. A camera sees the beam only while every
        camera above it is retracted.
    beam_current_a : float, optional
        Beam current where the beam starts (the specimen; the source with a condenser, before the aperture);
        with the exposure it sets the dose.
    ray_sample_power : int, optional
        Trace ``2**ray_sample_power`` rays per image.
    detector_samples_per_pixel : int, optional
        Internal detector sub-samples per output pixel axis.
    seed : int, optional
        Seed of the camera noise.
    ht_wobbler_v : float, optional
        How far the HT wobbler moves the high tension each way (V). 0, the default, leaves it still.
    """

    def __init__(self, truth, electronics, *, magnification_table, incident_beam, diffraction_table=None,
                 diffraction_camera_length_m=None, magnification=None, condenser_dac=None, spot_table=None, camera=None,
                 lower_cameras=(), beam_current_a=1.602176634e-9, ray_sample_power=18, detector_samples_per_pixel=1,
                 seed=0, ht_wobbler_v=0.0):
        require_x64("the simulated microscope")
        if not isinstance(electronics, Electronics):
            raise TypeError("electronics must be an Electronics instance.")
        self.truth, self.electronics = truth, electronics
        self.magnification_table = tuple(sorted(
            (float(m), {k: int(w) for k, w in dac_values.items()}) for m, dac_values in magnification_table.items()))
        self.diffraction_table = tuple(sorted(
            (float(m), {k: int(w) for k, w in dac_values.items()}) for m, dac_values in (diffraction_table or {}).items()))
        if not self.magnification_table:
            raise ValueError("Supply at least one magnification row.")
        self.diffraction_camera_length_m = diffraction_camera_length_m
        self.start_magnification = magnification
        self.condenser_dac = {n: int(v) for n, v in (condenser_dac or {}).items() if n in CONDENSER_LENSES}
        self.spot_table = {int(spot): {n: int(v) for n, v in row.items() if n in CONDENSER_LENSES}
                           for spot, row in (spot_table or {}).items()}
        self.spot_deflectors = {int(spot): {n: [int(v) for v in row[n]] for n in row if n in DEFLECTORS}
                                for spot, row in (spot_table or {}).items()}
        self.camera = camera or CameraModel()
        self.lower_cameras = {c.model.name: c for c in sorted(lower_cameras, key=lambda c: c.below_m)}
        if self.camera.name in self.lower_cameras or len(self.lower_cameras) != len(tuple(lower_cameras)):
            raise ValueError("Every camera needs its own name.")
        self.beam_current_a = float(beam_current_a)
        self.ray_sample_power = int(ray_sample_power)
        self.detector_samples_per_pixel = int(detector_samples_per_pixel)
        self.rng = np.random.default_rng(seed)
        self.incident_beam = incident_beam
        self.ht_wobbler_v = float(ht_wobbler_v)
        self._images = OrderedDict()
        self._pyjem = None
        self.reset()

    @classmethod
    def from_file(cls, path, *, ht_v, second_ht_v=None, diffraction_camera_length_m=None, magnification=None,
                  condenser_path=None, ray_sample_power=22, detector_samples_per_pixel=8, seed=0):
        """A simulated microscope from a ground-truth file, at its power-on state at ``ht_v``.

        With ``condenser_path``, a second file (JSON) describing the TEM-mode condenser
        (:meth:`~temgym_core.pyjem_simulator.condenser.Condenser.from_dict`; written by
        ``examples/virtual_neoarm/design_condenser.py``), the beam starts at the source and goes through the
        condenser and the objective's front field (the file's objective, an ElectromagneticLens, mirrored); the
        file's ``beam`` and ``beam_size`` (the beam at the specimen) are then not needed or used, and the beam current
        is the condenser source's.

        The file (JSON) describes the microscope in plain physics, at ``reference_voltage_v``:

        * ``lenses``: the objective and IL1, IL2, IL3, PL1, each placed at ``z_m`` from the specimen, either

          - a uniform magnetic field (TemGym :class:`~temgym_core.components.HardEdgeSolenoid`) of ``length_m``,
            with its strength ``q`` (rad) for the objective, ``q_per_dac`` (rad per DAC unit) for the others; or
          - a TemGym :class:`~temgym_core.components.ElectromagneticLens` with its own ``Gc`` (strength) and ``Tc``
            (thickness per ampere-turn): ``ampere_turns`` for the objective, ``ampere_turns_per_dac`` for the
            others. Its focusing and image rotation follow the high tension by themselves;

          and ``camera_z_m``; optionally ``camera_rotation_rad``, a TemGym
          :class:`~temgym_core.components.Rotator` just before the camera;
        * ``deflectors``: ``shift_m_per_dac`` (CLA1) and ``tilt_rad_per_dac`` (CLA2), pure; or ``kick_per_dac``, the
          4 x 4 matrix from (CLA1 x, CLA1 y, CLA2 x, CLA2 y) DAC units to (x, y, theta_x, theta_y) at the specimen;
        * ``beam``: ``rms_size_m``, ``rms_angle_rad`` and ``current_a`` at the specimen;
        * ``objective_dac`` (``OLc``, ``OLf``) and ``lens_table``: ``MAG`` (nominal magnification
          -> IL1-IL3, PL1 DAC values) and ``DIFF`` (camera length in m -> DAC values);
        * optionally ``condenser_dac``: the DAC values the console holds for CL1, CL2, CL3 and CM;
        * optionally ``camera``: :class:`CameraModel` fields (``sensor_shape``, ``sensor_pixel_m``, ``binnings``,
          ``mirrored``, ...);
        * optionally ``lower_cameras``: a list of more cameras further down the column (:class:`LowerCamera`), each
          with ``below_camera_m``, ``camera`` (:class:`CameraModel` fields, with its own ``name``) and optionally
          ``rotation_rad`` and ``centre_m``;
        * optionally ``pla``: ``rad_per_dac``, the slope kick per DAC unit of PLA, a TemGym
          :class:`~temgym_core.components.Deflector` at PL1's ``z_m``, and ``diffraction_offset_rad``,
          a fixed kick (x, y) that is there in DIFF only, so the pattern starts off-centre;
        * optionally ``aberrations``: ``objective``: ``seidel``, Seidel coefficient name (``A`` ... ``F``,
          ``e``, ``f``, ``c``) -> value, a :class:`~temgym_core.pyjem_simulator.seidel_plate.SeidelPlate`
          (TemGym's Seidel wave aberration acting on rays) at the specimen plane: the objective's field
          aberrations by where the beam is at the specimen. There the field terms (``C``, ``E``, ``F``) all
          give a tilt that grows as the cube of the beam's distance from the axis, so a centred beam stays
          round and a tilt (CLA2) is untouched. (A plate after the specimen would be levered by the
          magnification in MAG.)
        * optionally, in a projection lens's entry, ``offset_m``: (x, y), how far that lens sits off the axis. Its
          field acts about its own axis (TemGym :class:`~temgym_core.components.Scanner` shifts the rays by
          -offset before it and back after it), so the beam lands elsewhere on the camera at each magnification;
        * optionally ``hysteresis``: lens name (IL1-IL3, PL1, CL1, CL3, CM) -> ``widths_dac`` and ``fractions`` of a
          :class:`~temgym_core.pyjem_simulator.electronics.LensHysteresis`: that lens's field lags behind its DAC
          value, depending on the values it had before;
        * optionally ``beam_size``: CL3 (brightness) sets the beam's rms size at the specimen, smallest
          (``smallest_rms_m``) at ``CL3_smallest_dac`` and growing by ``rms_m_per_dac`` per DAC unit either side:
          sqrt(smallest_rms_m^2 + (rms_m_per_dac x (CL3 - CL3_smallest_dac))^2). Without it, ``beam``'s
          ``rms_size_m`` is used and CL3 changes nothing.

        A solenoid's strength is ``q_per_dac`` x its DAC value x p(reference) / p(HT): the same field
        turns a slower beam more (an ElectromagneticLens does this itself). The console's DAC values are the file's times p(HT) / p(reference),
        so the optics at ``ht_v`` are those at the reference HT. ``second_ht_v`` is another HT it
        models, with the DAC values held. ``diffraction_camera_length_m`` is the camera length the
        DIFF selector starts at (the nearest row), ``magnification`` the MAG row.
        """
        import json
        from pathlib import Path

        from ..aberrations import SeidelCoeffs
        from ..components import ElectromagneticLens, HardEdgeSolenoid, Plane, Rotator, Scanner
        from ..constants import magnetic_strength_ratio
        from .electronics import DeflectorDrive, LensDrive, LensHysteresis
        from .seidel_plate import SeidelPlate

        g = json.loads(Path(path).read_text())
        v_ref, lenses = float(g["reference_voltage_v"]), g["lenses"]
        names = ("objective", *PROJECTOR_LENSES)
        aberration = g.get("aberrations", {}).get("objective")

        def lens(n, control, r):
            """One lens at its relative control; if it sits off the axis, the rays are shifted to its axis and back."""
            entry, z = lenses[n], lenses[n]["z_m"]
            if "Gc" in entry:                    # an ElectromagneticLens: it scales with the HT itself
                turns = entry["ampere_turns"] if n == "objective" else entry["ampere_turns_per_dac"]
                body = ElectromagneticLens(z=z, turns=turns, current=control, Gc=entry["Gc"], Tc=entry.get("Tc", 0.0))
                end = z + body.thickness
            else:                                # a uniform field (solenoid): its strength scales as p(reference) / p(HT)
                per_unit = entry["q"] if n == "objective" else entry["q_per_dac"]
                body = HardEdgeSolenoid(z=z, length=entry["length_m"], q=per_unit * control * r)
                end = z + entry["length_m"]
            if "offset_m" not in entry:
                return (body,)
            dx, dy = entry["offset_m"]
            return (Scanner(z=z, scan_pos_x=-dx, scan_pos_y=-dy), body, Scanner(z=end, scan_pos_x=dx, scan_pos_y=dy))

        camera_rotation = ()
        if "camera_rotation_rad" in g:           # the camera turned about the axis
            camera_rotation = (Rotator(z=g["camera_z_m"], angle=float(np.degrees(g["camera_rotation_rad"]))),)

        def column(controls, voltage_v):
            # controls: the objective's relative strength, then IL1-IL3, PL1 DAC values
            r = magnetic_strength_ratio(voltage_v, v_ref)
            plate = ()
            if aberration:                       # at the specimen plane, scaling with the field like the lenses
                plate = (SeidelPlate(z=0.0, object_plane_dist=0.0,
                                     coeffs=SeidelCoeffs(**{name: value * r for name, value in aberration["seidel"].items()})),)
            parts = [part for n, c in zip(names, controls) for part in lens(n, c, r)]
            return (*plate, *parts, *camera_rotation, Plane(z=g["camera_z_m"]))

        beam_size = None
        if "beam_size" in g:                     # CL3 sets the beam's rms size at the specimen
            law = g["beam_size"]

            def beam_size(cl3_dac):
                return float(np.hypot(law["smallest_rms_m"], law["rms_m_per_dac"] * (cl3_dac - law["CL3_smallest_dac"])))

        pla, pla_z_m, pla_kick = g.get("pla"), None, None
        if pla:                                  # PLA: a slope kick at the plane of PL1, scaled with HT like CLA1/CLA2
            pla_drive = DeflectorDrive(np.diag([1.0, 1.0, pla["rad_per_dac"], pla["rad_per_dac"]]),
                                       reference_voltage_v=v_ref)
            offset = np.asarray(pla.get("diffraction_offset_rad", (0.0, 0.0)), float)

            def pla_kick(dac_x, dac_y, voltage_v, diffraction):
                kick = pla_drive.kick((NEUTRAL_DAC, NEUTRAL_DAC, dac_x, dac_y), voltage_v)[2:]
                return kick + offset if diffraction else kick

            pla_z_m = lenses["PL1"]["z_m"]

        def camera_model(fields):
            return CameraModel(**{k: tuple(v) if isinstance(v, list) else v for k, v in fields.items()})

        ht_v = float(ht_v)
        second_ht_v = float(second_ht_v or ht_v)
        scale = 1 / float(magnetic_strength_ratio(ht_v, v_ref))          # p(HT) / p(reference)
        at_ht = lambda row: {n: round(row[n] * scale) for n in PROJECTOR_LENSES}
        d = g["deflectors"]
        kick_per_dac = d.get("kick_per_dac") or np.diag([d["shift_m_per_dac"]] * 2 + [d["tilt_rad_per_dac"]] * 2)
        electronics = Electronics({n: LensDrive(dac_per_unit=1.0) for n in PROJECTOR_LENSES},
                                  DeflectorDrive(kick_per_dac, reference_voltage_v=v_ref),
                                  objective_dac=tuple(round(g["objective_dac"][n] * scale) for n in OBJECTIVE_LENSES),
                                  objective_control=scale,
                                  hysteresis={n: LensHysteresis(**law) for n, law in g.get("hysteresis", {}).items()})
        truth = OpticalTruth(column=column, voltages_v=(ht_v, second_ht_v), parameters=g,
                             pla_z_m=pla_z_m, pla_kick=pla_kick, beam_size=beam_size)
        beam, incident_beam = g.get("beam"), None
        if condenser_path is None:
            if beam is None:
                raise ValueError(f"{path} has no beam at the specimen: give condenser_path, the beam's condenser.")
            incident_beam = SpecimenBeam(rms_size_m=beam["rms_size_m"], rms_angle_rad=beam["rms_angle_rad"])
            current_a = beam["current_a"]
        else:           # the beam starts at the source and goes through the condenser
            from .condenser import Condenser

            objective = lenses["objective"]
            if "Gc" not in objective:
                raise ValueError("A condenser needs the objective as an ElectromagneticLens (with Gc): its front field "
                                 "is the objective mirrored.")
            front_field = {"z_m": -objective["z_m"], "ampere_turns": objective["ampere_turns"], "Gc": objective["Gc"],
                           "Tc": objective.get("Tc", 0.0)}
            condenser = Condenser.from_dict(json.loads(Path(condenser_path).read_text()), front_field, dac_scale=scale)
            truth = OpticalTruth(column=column, voltages_v=(ht_v, second_ht_v), parameters=g,
                                 pla_z_m=pla_z_m, pla_kick=pla_kick, condenser=condenser)
            current_a = condenser.source_current_a
        return cls(truth, electronics,
                   magnification_table={float(m): at_ht(row) for m, row in g["lens_table"]["MAG"].items()},
                   diffraction_table={float(length): at_ht(row) for length, row in g["lens_table"]["DIFF"].items()},
                   diffraction_camera_length_m=diffraction_camera_length_m, magnification=magnification,
                   condenser_dac=g.get("condenser_dac"), incident_beam=incident_beam,
                   camera=camera_model(g["camera"]) if "camera" in g else None,
                   lower_cameras=[LowerCamera(model=camera_model(c["camera"]), below_m=c["below_camera_m"],
                                              rotation_rad=c.get("rotation_rad", 0.0),
                                              centre_m=tuple(c.get("centre_m", (0.0, 0.0))))
                                  for c in g.get("lower_cameras", ())],
                   beam_current_a=current_a, ray_sample_power=ray_sample_power,
                   detector_samples_per_pixel=detector_samples_per_pixel, seed=seed)

    @classmethod
    def from_console(cls, tables, whole_column, *, ht_v=200e3, alpha=0, camera=None, hysteresis=None, magnification=None,
                     diffraction_camera_length_m=None, spot=None, beam_current_a=1.602176634e-9, ray_sample_power=18,
                     detector_samples_per_pixel=2, seed=0):
        """A simulated microscope with a real console's lens tables and any hidden column behind them.

        Parameters
        ----------
        tables : dict
            What ``record_console_tables.py`` read from a console: ``rows`` (``MAG`` and ``DIFF``: per selector row
            ``mag`` = [value, unit, label] and every lens's DAC value) and ``spot_alpha`` (per spot and alpha: CL1,
            CL3, CM). The MAG and DIFF rows are kept in the console's order, so selector indices are the console's
            (a row out of order, like the NEOARM's last DIFF row, is left out). Selecting a row sets IL1-IL3 and
            PL1; selecting a spot sets CL1, CL3 and CM, and SpotA where the table has it; other deflectors stay
            where they are put.
        alpha : int, optional
            The console's alpha whose spot table is used. The simulated console's own alpha selector stays 0.
        whole_column : callable
            The hidden optics, as a function of the whole console (:attr:`OpticalTruth.whole_column`).
        ht_v : float, optional
            The high tension.
        camera : CameraModel, optional
            The camera, in the column's length unit.
        hysteresis : dict, optional
            Lens name -> :class:`~temgym_core.pyjem_simulator.electronics.LensHysteresis`.
        magnification, diffraction_camera_length_m, spot : optional
            The MAG row, DIFF row and spot the console starts on.
        """
        from .electronics import DeflectorDrive, LensDrive

        rows = {mode: [r for i, r in enumerate(tables["rows"][mode])
                       if all(r["mag"][0] > earlier["mag"][0] for earlier in tables["rows"][mode][:i])]
                for mode in ("MAG", "DIFF")}
        spots = {r["spot"]: {n: r[n] for n in ("CL1", "CL3", "CM", "SpotA") if n in r}
                 for r in tables["spot_alpha"] if r["alpha"] == alpha}
        electronics = Electronics({n: LensDrive(dac_per_unit=1.0) for n in PROJECTOR_LENSES},
                                  DeflectorDrive(np.eye(4), reference_voltage_v=ht_v),      # not used
                                  objective_dac=tuple(rows["MAG"][0][n] for n in OBJECTIVE_LENSES),
                                  hysteresis=hysteresis or {})
        scope = cls(OpticalTruth(column=None, voltages_v=(ht_v,), whole_column=whole_column), electronics,
                    magnification_table={r["mag"][0]: {n: r[n] for n in PROJECTOR_LENSES} for r in rows["MAG"]},
                    diffraction_table={r["mag"][0] / 1000: {n: r[n] for n in PROJECTOR_LENSES} for r in rows["DIFF"]},
                    magnification=magnification, diffraction_camera_length_m=diffraction_camera_length_m,
                    spot_table=spots, incident_beam=None, camera=camera, beam_current_a=beam_current_a,
                    ray_sample_power=ray_sample_power, detector_samples_per_pixel=detector_samples_per_pixel, seed=seed)
        if spot is not None:
            scope.select_spot(spot)
            scope.settle()
        return scope

    def pyjem(self):
        """``TEM3, detector``: this microscope behind PyJEM's names, used exactly like PyJEM's
        (the same pair on every call). See :func:`~temgym_core.pyjem_simulator.pyjem.pyjem_interface`."""
        if self._pyjem is None:
            from .pyjem import pyjem_interface

            self._pyjem = pyjem_interface(self)
        return self._pyjem

    # ------------------------------------------------------------------ state
    def reset(self):
        """Return every control to its power-on value (the MAG and DIFF rows nearest ``magnification`` and
        ``diffraction_camera_length_m``, else the first rows; neutral DAC values)."""
        self.ht_v = self.truth.voltages_v[0]
        self.tem_stem = 0
        self.function_mode = 0
        self.probe_mode = 0
        self.alpha = 0
        self.selector = 0
        if self.start_magnification is not None:
            mags = np.array([row[0] for row in self.magnification_table])
            self.selector = int(np.argmin(np.abs(mags - self.start_magnification)))
        self.camera_length_selector = 0
        if self.diffraction_table and self.diffraction_camera_length_m is not None:
            lengths = np.array([row[0] for row in self.diffraction_table])
            self.camera_length_selector = int(np.argmin(np.abs(lengths - self.diffraction_camera_length_m)))
        self.spot = 2                        # PyJEM index of "spot 3"; not modelled
        self.direct = {}
        self.flc_switch = {name: 0 for name in LENS_IDS}
        self.flc_value = {name: None for name in LENS_IDS}
        self.deflectors = {name: [NEUTRAL_DAC, NEUTRAL_DAC] for name in DEFLECTORS}
        self.select_spot(self.spot)          # the spot table's spot alignment, if it has one
        self.beam_blank = 0
        # GUN3.SetHtWobbler switches this. While it is on, the high tension alternates between
        # ht_v + ht_wobbler_v and ht_v - ht_wobbler_v from one camera frame to the next. A real wobbler swings it
        # continuously.
        self.ht_wobbler = False
        self.ht_wobble_sign = 1
        self.aperture_kind = CONDENSER_APERTURE
        self.exp_kind = 0
        self.apertures = {kind: {"size": 0, "position": list(APERTURE_HOME)} for kind in APERTURE_KINDS}
        self.apertures[CONDENSER_APERTURE]["size"] = 1
        self.specimen = None                 # vacuum; or a TemGym component at the specimen plane
        # With a specimen, None images it with rays, as its shadow. A number of beamlets across images it with
        # one coherent Gaussian beam at the specimen. The beam is decomposed into that number squared of beamlets.
        self.wave_beamlets = None
        self.stage = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.lens_memory = {}                # per lens with hysteresis: (the DAC value it has, its backlash elements)
        self.settle()
        self.screen = 1                      # PyJEM detect3 screen: 0 lower screen, 1 the first camera, 2 the camera below
        self.detectors = {}                  # per camera name: the detector settings a script can change
        for name, model in self.camera_models.items():
            binning = max(model.binnings)
            shape, _, _ = model.geometry(binning)
            self.detectors[name] = {"inserted": True, "binning": binning, "exposure_us": 10_000.0, "frames": 1,
                                    "area": [0, 0, shape[1], shape[0]], "gain_index": 0, "offset_index": 0}

    @property
    def camera_models(self):
        """Every camera's :class:`CameraModel` by name, from the top of the column down."""
        return {self.camera.name: self.camera, **{name: c.model for name, c in self.lower_cameras.items()}}

    @property
    def detector(self):
        """The first camera's detector settings."""
        return self.detectors[self.camera.name]

    def blocked_by(self, camera):
        """The inserted cameras above ``camera`` (by name): while any is in, ``camera`` sees no beam."""
        names = list(self.camera_models)
        return [name for name in names[:names.index(camera)] if self.detectors[name]["inserted"]]

    @property
    def magnification(self):
        """Nominal magnification of the selected TEM MAG row."""
        return self.magnification_table[self.selector][0]

    @property
    def diffraction(self):
        """True in TEM DIFF mode."""
        return self.tem_stem == 0 and self.function_mode == 4

    @property
    def camera_length(self):
        """Camera length (m) of the selected TEM DIFF row (``None`` without a diffraction table)."""
        if not self.diffraction_table:
            return None
        return self.diffraction_table[self.camera_length_selector][0]

    def table_dac(self, name):
        """DAC value a lens takes from the selector or power-on tables (``None`` if absent).

        Parameters
        ----------
        name : str
            Lens name (a key of :data:`LENS_IDS`).

        Returns
        -------
        int or None
        """
        if name in PROJECTOR_LENSES:
            if self.diffraction:
                if not self.diffraction_table:
                    return None
                return self.diffraction_table[self.camera_length_selector][1].get(name)
            return self.magnification_table[self.selector][1].get(name)
        if name in OBJECTIVE_LENSES:
            return self.electronics.objective_dac[OBJECTIVE_LENSES.index(name)]
        if name in CONDENSER_LENSES:
            if self.truth.condenser is not None and name != "CL2":
                return self.truth.condenser.table_dac(self.spot, name)
            return self.spot_table.get(self.spot, {}).get(name, self.condenser_dac.get(name, NEUTRAL_DAC))
        return NEUTRAL_DAC

    def lens_dac(self, name):
        """Effective DAC value: free lens control, else a direct setting, else the tables.

        Parameters
        ----------
        name : str
            Lens name.

        Returns
        -------
        int or None
            ``None`` when no table provides the lens (DIFF without a diffraction table).
        """
        if self.flc_switch[name]:
            return self.flc_value[name]
        if name in self.direct:
            return self.direct[name]
        return self.table_dac(name)

    def settle(self):
        """Let every lens with hysteresis go to the DAC value it now has. The PyJEM calls do this after each change,
        so a lens's field depends on the order of the values it was given."""
        for name, law in self.electronics.hysteresis.items():
            dac = self.lens_dac(name)
            if dac is None:
                continue
            if name not in self.lens_memory:
                self.lens_memory[name] = (dac, law.at_rest(dac))
            elif self.lens_memory[name][0] != dac:
                self.lens_memory[name] = (dac, law.move(self.lens_memory[name][1], dac))

    def effective_dac(self, name):
        """The DAC value a lens acts as if it had: its DAC value, with its hysteresis lag if it has one."""
        dac = self.lens_dac(name)
        law = self.electronics.hysteresis.get(name)
        if law is None or dac is None:
            return dac
        self.settle()
        return law.effective_dac(self.lens_memory[name][1], dac)

    def select_spot(self, spot):
        """Select a spot-size index, as the console does: the tables reload, and a deflector that the spot table
        holds for this spot (the spot alignment ``SpotA``) takes its value.

        Parameters
        ----------
        spot : int
            PyJEM's spot-size index.
        """
        self.spot = int(spot)
        for name, value in self.spot_deflectors.get(self.spot, {}).items():
            self.deflectors[name] = list(value)
        self.reload_tables()

    def reload_tables(self):
        """A selector, function-mode or spot change reloads table values and clears direct settings."""
        self.direct = {}

    # ---------------------------------------------------------------- physics
    def beam_voltage_v(self):
        """Return the high tension of the next frame: ht_v, moved by the HT wobbler while it is on."""
        if not self.ht_wobbler:
            return float(self.ht_v)
        return float(self.ht_v) + self.ht_wobble_sign * self.ht_wobbler_v

    def physical_setting(self):
        """Translate the current DAC values into relative lens strengths and the kick at the specimen.

        Returns
        -------
        PhysicalSetting

        Raises
        ------
        UnmodelledStateError
            Listing every aspect of the state the optics cannot render.
        """
        reasons = []
        if self.ht_v not in self.truth.voltages_v:
            reasons.append(f"high tension {self.ht_v:g} V is not one of the modelled {self.truth.voltages_v}")
        if self.tem_stem != 0:
            reasons.append("STEM mode is not modelled")
        elif self.diffraction and not self.diffraction_table:
            reasons.append("TEM function mode DIFF is not modelled (no diffraction table)")
        elif self.function_mode not in (0, 4):
            reasons.append(f"TEM function mode {TEM_FUNCTION_MODES.get(self.function_mode)} is not modelled "
                           "(MAG and DIFF only)")
        if self.probe_mode != 0:
            reasons.append("only the TEM probe mode is modelled")
        if self.alpha != 0:
            reasons.append("only alpha selector 0 is modelled")
        if self.truth.whole_column is not None:  # the hidden optics take the whole console: nothing else to refuse
            if reasons:
                raise UnmodelledStateError(reasons)
            lenses = tuple((n, float(NEUTRAL_DAC if self.lens_dac(n) is None else self.effective_dac(n)))
                           for n in LENS_IDS)
            deflectors = tuple((n, tuple(float(v) for v in pair)) for n, pair in self.deflectors.items())
            # The condenser aperture's hole (0 = none) and its Apt3 position, for hidden optics that have one.
            aperture = self.apertures[CONDENSER_APERTURE]
            condenser_aperture = (("CondenserAperture", (float(aperture["size"]), *map(float, aperture["position"]))),)
            return PhysicalSetting(voltage_v=self.beam_voltage_v(), projector_controls=(), specimen_kick=(),
                                   console=lenses + deflectors + condenser_aperture)
        for name in UNMODELLED_LENSES:
            if self.lens_dac(name) != NEUTRAL_DAC:
                reasons.append(f"{name} is not modelled and must stay at {NEUTRAL_DAC}")
        objective = tuple(self.lens_dac(name) for name in OBJECTIVE_LENSES)
        if objective != self.electronics.objective_dac:
            reasons.append(f"objective (OLc, OLf) = {objective} differs from the modelled "
                           f"{self.electronics.objective_dac}")
        modelled = ("CLA1", "CLA2", "PLA") if self.truth.pla_kick is not None else ("CLA1", "CLA2")
        for name in DEFLECTORS:
            if name not in modelled and self.deflectors[name] != [NEUTRAL_DAC, NEUTRAL_DAC]:
                reasons.append(f"deflector {name} is not modelled and must stay neutral")
        for kind, state in self.apertures.items():
            if kind != CONDENSER_APERTURE and state["size"] != 0:
                reasons.append(f"aperture {APERTURE_KINDS[kind]} must stay open")
        condenser = self.truth.condenser
        if condenser is not None:
            if self.spot not in condenser.spot_table:
                reasons.append(f"spot size index {self.spot} is not in the spot table {sorted(condenser.spot_table)}")
            if self.lens_dac("CL2") != self.table_dac("CL2"):
                reasons.append(f"CL2 is not modelled and must stay at {self.table_dac('CL2')}")
        if reasons:
            raise UnmodelledStateError(reasons)
        drives = self.electronics.lenses
        projector_controls = (self.electronics.objective_control,) + tuple(
            drives[n].control_at(self.effective_dac(n)) if n in self.electronics.hysteresis
            else drives[n].control(self.lens_dac(n)) for n in PROJECTOR_LENSES)
        dac_values = (*self.deflectors["CLA1"], *self.deflectors["CLA2"])
        kick = tuple(float(k) for k in self.electronics.deflector.kick(dac_values, self.beam_voltage_v()))
        pla_kick = (0.0, 0.0)
        if self.truth.pla_kick is not None:
            pla_kick = tuple(float(k) for k in self.truth.pla_kick(*self.deflectors["PLA"], self.beam_voltage_v(), self.diffraction))
        if condenser is not None:
            aperture = self.apertures[CONDENSER_APERTURE]
            return PhysicalSetting(voltage_v=self.beam_voltage_v(), projector_controls=projector_controls,
                                   specimen_kick=kick, pla_kick=pla_kick,
                                   condenser_dac=tuple((n, self.effective_dac(n)) for n in ("CL1", "CL3", "CM")),
                                   aperture=(int(aperture["size"]), *map(int, aperture["position"])))
        cl3_dac = int(self.lens_dac("CL3")) if self.truth.beam_size is not None else None
        return PhysicalSetting(voltage_v=self.beam_voltage_v(), projector_controls=projector_controls, specimen_kick=kick,
                               pla_kick=pla_kick, cl3_dac=cl3_dac)

    def expected_image(self, setting, binning, camera=None):
        """Mean transmitted fraction per output pixel for a setting (cached).

        Parameters
        ----------
        setting : PhysicalSetting
            Physical controls.
        binning : int
            Camera binning.
        camera : str, optional
            The camera's name; the first camera by default.

        Returns
        -------
        numpy.ndarray
            Read-only image per incident electron, full binned frame.
        """
        specimen = self.specimen
        camera = camera or self.camera.name
        model = self.camera_models[camera]
        key = (setting, binning, camera, self.ray_sample_power, self.detector_samples_per_pixel, id(specimen),
               self.incident_beam, self.wave_beamlets)
        if key in self._images:
            self._images.move_to_end(key)
            return self._images[key]
        if setting.console is not None:          # the whole console in, the column from the source and its beam out
            projector, start, mean, covariance = self.truth.whole_column(dict(setting.console), setting.voltage_v)
            projector = tuple(projector)
        else:
            projector = tuple(self.truth.column(np.asarray(setting.projector_controls), setting.voltage_v))
        if self.truth.pla_z_m is not None:       # PLA's deflector, in front of the first component at or after its plane
            pla = Deflector(z=self.truth.pla_z_m, def_x=setting.pla_kick[0], def_y=setting.pla_kick[1])
            position = next((i for i, part in enumerate(projector) if part.z >= self.truth.pla_z_m), len(projector))
            projector = projector[:position] + (pla,) + projector[position:]
        if camera in self.lower_cameras:         # a straight flight on, to where that camera is and how it is turned
            lower = self.lower_cameras[camera]
            z = projector[-1].z + lower.below_m
            projector = projector + (Plane(z=z), Scanner(z=z, scan_pos_x=-lower.centre_m[0], scan_pos_y=-lower.centre_m[1]),
                                     Rotator(z=z, angle=float(np.degrees(lower.rotation_rad))))
        column = tuple(projector) if specimen is None else (specimen, *projector)
        condenser = self.truth.condenser
        if setting.console is not None:          # the beam and the whole column are already there
            column = projector if specimen is None else tuple(
                sorted((*projector, specimen), key=lambda part: float(part.z)))
        elif condenser is not None:              # from the source, through the condenser
            hole, *position = setting.aperture
            column = condenser.components(dict(setting.condenser_dac), hole=hole, aperture_position_dac=position,
                                          specimen_kick=setting.specimen_kick,
                                          objective_control=setting.projector_controls[0],
                                          voltage_v=setting.voltage_v) + column
            start, mean, covariance = condenser.source()
        else:                                    # from the specimen plane
            b = self.incident_beam
            size = b.rms_size_m if setting.cl3_dac is None else self.truth.beam_size(setting.cl3_dac)
            start, mean = 0.0, np.asarray(setting.specimen_kick, float)
            covariance = np.diag([size ** 2] * 2 + [b.rms_angle_rad ** 2] * 2)
        shape, pitch, psf = model.geometry(binning)
        detector = Detector(z=0.0, shape=shape, pixel_size=(pitch, pitch), flip_y=not model.mirrored)
        rays = sample_rays(mean, covariance, z=start, voltage=setting.voltage_v, sample_power=self.ray_sample_power)
        if specimen is not None and self.wave_beamlets is not None:
            image = self._wave_image(rays, column, specimen, detector, psf * pitch)
        else:
            image = ray_image(rays, column, detector, samples_per_pixel=self.detector_samples_per_pixel,
                              psf_sigma_pixels=psf)
        image.setflags(write=False)
        self._images[key] = image
        if len(self._images) > 64:
            self._images.popitem(last=False)
        return image

    def _wave_image(self, rays, column, specimen, detector, psf_sigma_m):
        """Image of the specimen with one coherent Gaussian beam: flux per pixel, per incident electron.

        The function traces the rays to the specimen plane. It weights each ray by what the apertures above let
        through. One round Gaussian takes the rays' weighted centre, tilt, rms size and curvature there. That
        Gaussian is the coherent part of the beam, and it leaves out the rest of the rays' spread. It carries the
        fraction of the rays that got through. The function decomposes it into ``wave_beamlets**2`` beamlets
        over +-3 rms. The beamlets go through the specimen and the column below it to the camera, where their
        fields add.
        """
        import jax.numpy as jnp

        from ..constants import energy2wavelength
        from ..evaluate import gaussian_image
        from ..gaussian import GaussianBeam
        from ..components import GaussianActionComponent
        from ..components._base import _uses_default_log_transmission
        from ..run import run_iter
        from ..source import decompose_gaussian

        z = float(specimen.z)
        above = [part for part in column if float(part.z) < z]
        below = [part for part in column if float(part.z) > z]
        # Rays carry no amplitude: the apertures above weight them by their intensity transmission, as in ray_image.
        weights = jnp.ones_like(rays.x)
        for component, at in run_iter(rays, [*above, Plane(z=z)]):
            if isinstance(component, GaussianActionComponent) and not _uses_default_log_transmission(component):
                weights *= jnp.exp(2 * component.log_transmission(at.r_xy))
        weights = np.asarray(weights, float)
        transmitted = weights.mean()
        if transmitted == 0:                                # the apertures stop the whole beam
            return np.zeros(detector.shape)
        x, y, dx, dy = (np.asarray(v, float) for v in (at.x, at.y, at.dx, at.dy))
        mean = {name: np.average(v, weights=weights) for name, v in (("x", x), ("y", y), ("dx", dx), ("dy", dy))}
        vx = np.average((x - mean["x"]) ** 2, weights=weights)
        vy = np.average((y - mean["y"]) ** 2, weights=weights)
        variance = 0.5 * (vx + vy)
        curvature = np.average((x - mean["x"]) * (dx - mean["dx"]) + (y - mean["y"]) * (dy - mean["dy"]),
                               weights=weights) / (vx + vy)
        width = 2 * np.sqrt(variance)                       # 1/e amplitude radius of the Gaussian of that rms size
        voltage = float(np.ravel(at.voltage)[0])
        k = 2 * np.pi / float(energy2wavelength(voltage))
        beam = GaussianBeam(
            x=jnp.asarray(mean["x"]), y=jnp.asarray(mean["y"]), dx=jnp.asarray(mean["dx"]), dy=jnp.asarray(mean["dy"]),
            z=z, voltage=voltage, amplitude=jnp.sqrt(2 * transmitted / (jnp.pi * width**2)) + 0j,
            Q_inv=(curvature + 2j / (k * width**2)) * jnp.eye(2), pathlength=0.0,
        )
        bundle = decompose_gaussian(beam, num_across=self.wave_beamlets, extent_rms=3.0)
        end = float(column[-1].z)
        camera = Detector(z=end, shape=detector.shape, pixel_size=detector.pixel_size, flip_y=detector.flip_y)
        out = _run_compiled(bundle, [specimen, *below, camera])
        image = gaussian_image(out, camera, samples_per_pixel=1, blur_sigma_m=psf_sigma_m if psf_sigma_m > 0 else None,
                               method="auto")
        return np.asarray(image)

    def acquire(self, camera=None):
        """Record one camera exposure with the current state and detector settings.

        Parameters
        ----------
        camera : str, optional
            The camera's name; the first camera by default.

        Returns
        -------
        numpy.ndarray
            Integer ADU frame (rows, columns) of the imaging area, in the camera's
            output type. A camera below an inserted one sees no beam: its frame is dark.

        Raises
        ------
        RuntimeError
            If the camera is retracted.
        UnmodelledStateError
            If the beam is on and the state is not modelled.
        """
        camera = camera or self.camera.name
        model, detector = self.camera_models[camera], self.detectors[camera]
        if not detector["inserted"]:
            raise RuntimeError(f"The {camera} detector is retracted.")
        binning = detector["binning"]
        shape, _, _ = model.geometry(binning)
        x, y, width, height = detector["area"]
        if self.beam_blank or self.blocked_by(camera) or self.screen == 0:
            expected = np.zeros(shape)
        else:
            incident = self.beam_current_a * detector["exposure_us"] * 1e-6 / elementary_charge
            expected = self.expected_image(self.physical_setting(), binning, camera) * incident
        if self.ht_wobbler:                      # the next frame sees the other side of the wobble
            self.ht_wobble_sign = -self.ht_wobble_sign
        return model.expose(expected[y:y + height, x:x + width], self.rng, detector["frames"])
