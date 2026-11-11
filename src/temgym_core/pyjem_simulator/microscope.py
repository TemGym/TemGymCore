"""The simulated microscope: DAC values in, camera frames out, truth hidden.

:class:`SimulatedMicroscope` keeps the state of a JEOL-like TEM column in the
units its control interface uses (16-bit DAC values, volts, nanometres, indices). A
frame is rendered only when the optical model can justify the state: anything
unmodelled (another function mode, a moved unmodelled lens or deflector, an
untabulated high tension, an inserted objective aperture, ...) raises
:class:`UnmodelledStateError` listing every reason, instead of returning an
image the model cannot produce.

The beam starts at the specimen plane (:class:`SpecimenBeam`); the condenser is not modelled. If
the optics give a law for it (:attr:`OpticalTruth.beam_size`), CL3 (brightness) sets the beam's rms
size there; CL1, CL2, CM, spot size and the condenser aperture are accepted but change nothing.
The hidden optics are any TemGym column after the specimen (:class:`OpticalTruth`); images
come from TemGym's ray images (:func:`temgym_core.evaluate.ray_image`).
"""

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from scipy.constants import elementary_charge

from ..components import Deflector, Detector
from ..config import require_x64
from ..evaluate import ray_image
from ..source import sample_rays
from .camera import CameraModel
from .electronics import NEUTRAL_DAC, Electronics

#: PyJEM free-lens-control identifiers.
LENS_IDS = {"CL1": 0, "CL2": 1, "CL3": 2, "CM": 3, "OLc": 6, "OLf": 7, "OM": 8, "OM2": 9,
            "IL1": 10, "IL2": 11, "IL3": 12, "IL4": 13, "PL1": 14, "PL2": 15, "PL3": 16}
PROJECTOR_LENSES = ("IL1", "IL2", "IL3", "PL1")
#: Accepted but not modelled (the beam is given at the specimen plane), except CL3 when the optics give a beam-size law.
CONDENSER_LENSES = ("CL1", "CL2", "CL3", "CM")
OBJECTIVE_LENSES = ("OLc", "OLf")
UNMODELLED_LENSES = ("OM", "OM2", "IL4", "PL2", "PL3")
#: PyJEM Def3 deflector pairs; CLA1 (beam shift) and CLA2 (beam tilt) are modelled, PLA only if the optics give it a plane.
DEFLECTORS = ("GunA1", "GunA2", "CLA1", "CLA2", "CLs", "IS1", "IS2", "PLA", "OLs", "ILs",
              "SpotA", "ShifBal", "TiltBal", "AngBal")
TEM_FUNCTION_MODES = {0: "MAG", 1: "MAG2", 2: "LowMAG", 3: "SAMAG", 4: "DIFF"}
STEM_FUNCTION_MODES = {0: "Align", 1: "SM-LMAG", 2: "SM-MAG", 3: "AMAG", 4: "uuDIFF", 5: "Rocking"}
#: Apt3 kinds. The condenser aperture (kind 1) is accepted but not modelled; the others
#: (after the specimen, or analytical) must stay open.
APERTURE_KINDS = {1: "CLA", 2: "OLA", 3: "HCA", 4: "SAA", 5: "ENTA", 6: "EDS"}
CONDENSER_APERTURE = 1
APERTURE_HOME = (2048, 2048)


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
        angle spread stays the ``incident_beam``'s. Without it CL3 changes nothing.
    """

    column: object
    voltages_v: tuple
    parameters: dict = field(default_factory=dict)
    pla_z_m: float = None
    pla_kick: object = None
    beam_size: object = None

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
    """

    voltage_v: float
    projector_controls: tuple
    specimen_kick: tuple
    pla_kick: tuple = (0.0, 0.0)
    cl3_dac: int = None


@dataclass(frozen=True)
class SpecimenBeam:
    """The incident beam at the specimen plane: round and Gaussian, of rms size ``rms_size_m``
    and rms angle spread ``rms_angle_rad``, centred where the beam-shift and beam-tilt
    deflectors put it."""

    rms_size_m: float
    rms_angle_rad: float


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
    camera : CameraModel, optional
        The camera; ideal by default.
    beam_current_a : float, optional
        Beam current at the specimen; with the exposure it sets the dose.
    ray_sample_power : int, optional
        Trace ``2**ray_sample_power`` rays per image.
    detector_samples_per_pixel : int, optional
        Internal detector sub-samples per output pixel axis.
    seed : int, optional
        Seed of the camera noise.
    """

    def __init__(self, truth, electronics, *, magnification_table, incident_beam, diffraction_table=None,
                 diffraction_camera_length_m=None, magnification=None, condenser_dac=None, camera=None, beam_current_a=1.602176634e-9,
                 ray_sample_power=18, detector_samples_per_pixel=1, seed=0):
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
        self.camera = camera or CameraModel()
        self.beam_current_a = float(beam_current_a)
        self.ray_sample_power = int(ray_sample_power)
        self.detector_samples_per_pixel = int(detector_samples_per_pixel)
        self.rng = np.random.default_rng(seed)
        self.incident_beam = incident_beam
        self._images = OrderedDict()
        self._pyjem = None
        self.reset()

    @classmethod
    def from_file(cls, path, *, ht_v, second_ht_v=None, diffraction_camera_length_m=None, magnification=None,
                  ray_sample_power=22,
                  detector_samples_per_pixel=8, seed=0):
        """A simulated microscope from a ground-truth file, at its power-on state at ``ht_v``.

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
        from .electronics import DeflectorDrive, LensDrive
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

        ht_v = float(ht_v)
        second_ht_v = float(second_ht_v or ht_v)
        scale = 1 / float(magnetic_strength_ratio(ht_v, v_ref))          # p(HT) / p(reference)
        at_ht = lambda row: {n: round(row[n] * scale) for n in PROJECTOR_LENSES}
        d = g["deflectors"]
        kick_per_dac = d.get("kick_per_dac") or np.diag([d["shift_m_per_dac"]] * 2 + [d["tilt_rad_per_dac"]] * 2)
        electronics = Electronics({n: LensDrive(dac_per_unit=1.0) for n in PROJECTOR_LENSES},
                                  DeflectorDrive(kick_per_dac, reference_voltage_v=v_ref),
                                  objective_dac=tuple(round(g["objective_dac"][n] * scale) for n in OBJECTIVE_LENSES),
                                  objective_control=scale)
        truth = OpticalTruth(column=column, voltages_v=(ht_v, second_ht_v), parameters=g,
                             pla_z_m=pla_z_m, pla_kick=pla_kick, beam_size=beam_size)
        beam = g["beam"]
        return cls(truth, electronics,
                   magnification_table={float(m): at_ht(row) for m, row in g["lens_table"]["MAG"].items()},
                   diffraction_table={float(length): at_ht(row) for length, row in g["lens_table"]["DIFF"].items()},
                   diffraction_camera_length_m=diffraction_camera_length_m, magnification=magnification,
                   condenser_dac=g.get("condenser_dac"), incident_beam=SpecimenBeam(rms_size_m=beam["rms_size_m"], rms_angle_rad=beam["rms_angle_rad"]),
                   camera=CameraModel(**{k: tuple(v) if isinstance(v, list) else v for k, v in g["camera"].items()})
                   if "camera" in g else None,
                   beam_current_a=beam["current_a"], ray_sample_power=ray_sample_power,
                   detector_samples_per_pixel=detector_samples_per_pixel, seed=seed)

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
        self.beam_blank = 0
        self.aperture_kind = CONDENSER_APERTURE
        self.exp_kind = 0
        self.apertures = {kind: {"size": 0, "position": list(APERTURE_HOME)} for kind in APERTURE_KINDS}
        self.apertures[CONDENSER_APERTURE]["size"] = 1
        self.specimen = None                 # vacuum; or a TemGym component at the specimen plane
        self.stage = [0.0, 0.0, 0.0, 0.0, 0.0]
        binning = max(self.camera.binnings)
        shape, _, _ = self.camera.geometry(binning)
        self.detector = {"inserted": True, "binning": binning, "exposure_us": 10_000.0, "frames": 1,
                         "area": [0, 0, shape[1], shape[0]], "gain_index": 0, "offset_index": 0}

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
            return self.condenser_dac.get(name, NEUTRAL_DAC)
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

    def reload_tables(self):
        """A selector, function-mode or spot change reloads table values and clears direct settings."""
        self.direct = {}

    # ---------------------------------------------------------------- physics
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
        if reasons:
            raise UnmodelledStateError(reasons)
        drives = self.electronics.lenses
        projector_controls = (self.electronics.objective_control,) + tuple(
            drives[n].control(self.lens_dac(n)) for n in PROJECTOR_LENSES)
        dac_values = (*self.deflectors["CLA1"], *self.deflectors["CLA2"])
        kick = tuple(float(k) for k in self.electronics.deflector.kick(dac_values, self.ht_v))
        pla_kick = (0.0, 0.0)
        if self.truth.pla_kick is not None:
            pla_kick = tuple(float(k) for k in self.truth.pla_kick(*self.deflectors["PLA"], self.ht_v, self.diffraction))
        cl3_dac = int(self.lens_dac("CL3")) if self.truth.beam_size is not None else None
        return PhysicalSetting(voltage_v=float(self.ht_v), projector_controls=projector_controls, specimen_kick=kick,
                               pla_kick=pla_kick, cl3_dac=cl3_dac)

    def expected_image(self, setting, binning):
        """Mean transmitted fraction per output pixel for a setting (cached).

        Parameters
        ----------
        setting : PhysicalSetting
            Physical controls.
        binning : int
            Camera binning.

        Returns
        -------
        numpy.ndarray
            Read-only image per incident electron, full binned frame.
        """
        specimen = self.specimen
        key = (setting, binning, self.ray_sample_power, self.detector_samples_per_pixel, id(specimen),
               self.incident_beam)
        if key in self._images:
            self._images.move_to_end(key)
            return self._images[key]
        projector = tuple(self.truth.column(np.asarray(setting.projector_controls), setting.voltage_v))
        if self.truth.pla_z_m is not None:       # PLA's deflector, in front of the first component at or after its plane
            pla = Deflector(z=self.truth.pla_z_m, def_x=setting.pla_kick[0], def_y=setting.pla_kick[1])
            position = next((i for i, part in enumerate(projector) if part.z >= self.truth.pla_z_m), len(projector))
            projector = projector[:position] + (pla,) + projector[position:]
        b = self.incident_beam
        size = b.rms_size_m if setting.cl3_dac is None else self.truth.beam_size(setting.cl3_dac)
        mean = np.asarray(setting.specimen_kick, float)
        covariance = np.diag([size ** 2] * 2 + [b.rms_angle_rad ** 2] * 2)
        column = tuple(projector) if specimen is None else (specimen, *projector)
        shape, pitch, psf = self.camera.geometry(binning)
        detector = Detector(z=0.0, shape=shape, pixel_size=(pitch, pitch), flip_y=not self.camera.mirrored)
        rays = sample_rays(mean, covariance, z=0.0, voltage=setting.voltage_v, sample_power=self.ray_sample_power)
        image = ray_image(rays, column, detector, samples_per_pixel=self.detector_samples_per_pixel, psf_sigma_pixels=psf)
        image.setflags(write=False)
        self._images[key] = image
        if len(self._images) > 64:
            self._images.popitem(last=False)
        return image

    def acquire(self):
        """Record one camera exposure with the current state and detector settings.

        Returns
        -------
        numpy.ndarray
            Integer ADU frame (rows, columns) of the imaging area, in the camera's
            output type.

        Raises
        ------
        RuntimeError
            If the camera is retracted.
        UnmodelledStateError
            If the beam is on and the state is not modelled.
        """
        detector = self.detector
        if not detector["inserted"]:
            raise RuntimeError(f"The {self.camera.name} detector is retracted.")
        binning = detector["binning"]
        shape, _, _ = self.camera.geometry(binning)
        x, y, width, height = detector["area"]
        if self.beam_blank:
            expected = np.zeros(shape)
        else:
            incident = self.beam_current_a * detector["exposure_us"] * 1e-6 / elementary_charge
            expected = self.expected_image(self.physical_setting(), binning) * incident
        return self.camera.expose(expected[y:y + height, x:x + width], self.rng, detector["frames"])
