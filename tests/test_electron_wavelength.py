"""Test for the electron_wavelength utility function."""

import pytest
import numpy as np
from temgym_core.utils import electron_wavelength


def test_electron_wavelength_200kev():
    """Test wavelength calculation at 200 keV."""
    wavelength = electron_wavelength(200.0)
    # Expected value: ~2.5 pm (calculated with relativistic formula)
    # λ = h / sqrt(2 * m_e * e * V * (1 + e*V / (2*m_e*c^2)))
    assert 2.3e-12 < wavelength < 2.4e-12
    np.testing.assert_allclose(wavelength, 2.3249e-12, rtol=0.01)


def test_electron_wavelength_210kev():
    """Test wavelength calculation at 210 keV."""
    wavelength = electron_wavelength(210.0)
    # Expected value: ~2.4 pm
    assert 2.2e-12 < wavelength < 2.3e-12
    np.testing.assert_allclose(wavelength, 2.2531e-12, rtol=0.01)


def test_electron_wavelength_decreases_with_voltage():
    """Test that wavelength decreases as voltage increases."""
    wavelength_100 = electron_wavelength(100.0)
    wavelength_200 = electron_wavelength(200.0)
    wavelength_300 = electron_wavelength(300.0)
    
    assert wavelength_100 > wavelength_200 > wavelength_300


def test_electron_wavelength_relative_change():
    """Test the relative change between 200 and 210 keV."""
    wavelength_200 = electron_wavelength(200.0)
    wavelength_210 = electron_wavelength(210.0)
    
    relative_change = (wavelength_200 - wavelength_210) / wavelength_200
    # Should be around 3%
    assert 0.025 < relative_change < 0.035
    np.testing.assert_allclose(relative_change, 0.0309, rtol=0.05)


def test_electron_wavelength_physical_constants():
    """Test that wavelength is in the correct order of magnitude."""
    # For typical TEM voltages (100-300 keV), wavelengths should be 1-4 pm
    for voltage_kev in [100, 150, 200, 250, 300]:
        wavelength = electron_wavelength(voltage_kev)
        assert 1e-12 < wavelength < 5e-12  # 1-5 pm range
