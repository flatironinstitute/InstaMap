"""Utilities for working with physical units.
"""

from __future__ import annotations

import math

import ase.units


def energy_to_mass(energy: float) -> float:
    """Computes the relativistic mass of the electron from its energy.

    Parameters
    ----------
    energy : float
        The energy of the electron in eV

    Returns
    -------
    float
        The relativistic mass of the electron in kg
    """
    relativistic_factor = (1 + ase.units._e * energy) / (ase.units._me * ase.units._c**2)
    return ase.units._me * relativistic_factor


def energy_to_wavelength(energy: float) -> float:
    """Computes the relativistic de Broglie wavelength of the electron from its energy.

    Parameters
    ----------
    energy : float
        The energy of the electron in eV

    Returns
    -------
    float
        The wavelength of the electron in A
    """
    return ase.units._hplanck * ase.units._c / math.sqrt(
        energy * (2 * ase.units._me * ase.units._c**2 / ase.units._e + energy)
    ) / ase.units._e * 1e10


def energy_to_interaction(energy: float) -> float:
    """Computes the elastic scattering interaction parameter of the electron from its energy.

    Parameters
    ----------
    energy : float
        The energy of the electron in eV

    Returns
    -------
    float
        The interaction parameter of the electron in 1 / (A eV)
    """
    return 2 * math.pi * energy_to_mass(
        energy
    ) * ase.units.kg * ase.units._e * ase.units.C * energy_to_wavelength(energy) / (
        ase.units._hplanck * ase.units.s * ase.units.J
    )**2
