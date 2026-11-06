"""The physical constants are explicit and match the values previously taken from ase.units."""
import json
from pathlib import Path

import pytest

from temgym_core import constants

RAW = json.loads((Path(__file__).parent / "reference" / "baseline.json").read_text())["raw_constants"]


def test_explicit_constants_match_the_frozen_snapshot():
    assert constants.ELEMENTARY_CHARGE == RAW["_e"]
    assert constants.ELECTRON_MASS == RAW["_me"]
    assert constants.SPEED_OF_LIGHT == RAW["_c"]
    assert constants.PLANCK == RAW["_hplanck"]
    assert constants.HBAR == RAW["_hbar"]
    # Atomic-unit conversion factors (length A, energy eV, mass u, charge e).
    assert constants._KG == RAW["kg"]
    assert constants._COULOMB == RAW["C"]
    assert constants._JOULE == RAW["J"]
    assert constants._SECOND == RAW["s"]


def test_explicit_constants_agree_with_ase_when_it_is_installed():
    units = pytest.importorskip("ase").units
    assert constants.ELEMENTARY_CHARGE == units._e
    assert constants.ELECTRON_MASS == units._me
    assert constants.SPEED_OF_LIGHT == units._c
    assert constants.PLANCK == units._hplanck
    assert constants.HBAR == units._hbar
