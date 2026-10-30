"""From single-flake gaps to ensemble photoluminescence line shapes.

A real graphene-quantum-dot sample is an *ensemble*: the emission of the
whole sample is the size law E(d) averaged over the size distribution P(d)
(measured e.g. by TEM/AFM/DLS) and broadened by inhomogeneous + homogeneous
effects. This module builds that average:

    S(E) = integral P(d) G(E - E(d)) dd

with G a Gaussian of width ``broaden_ev``. ``stokes_ev`` optionally shifts
emission below absorption (a phenomenological Stokes shift).

E(d) is the zigzag edge-state law measured by exact diagonalization:

    E(d) = E1 * exp(-d / xi)

with xi ~ 1.2 nm and E1 ~ 2t ~ 5.4 eV (the benzene/molecular limit). The
confrontation with measured PL is the point: sub-1.5 nm dots emit in the
visible under this law, while larger dots emitting visible light require
edge/functional (sp3, C=O) states -- a testable statement for the lab.
"""

from __future__ import annotations

import numpy as np


def emission_energy_ev(
    diameter_nm: np.ndarray, amplitude_ev: float, xi_nm: float
) -> np.ndarray:
    """Edge-state emission energy E(d) = E1 exp(-d/xi)."""
    return amplitude_ev * np.exp(-np.asarray(diameter_nm) / xi_nm)


def lognormal_sizes(
    d0_nm: float, sigma_ln: float, n_samples: int = 40_000, seed: int = 5
) -> np.ndarray:
    """Lognormal size distribution with median ``d0_nm`` and width ``sigma_ln``."""
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=np.log(d0_nm), sigma=sigma_ln, size=n_samples)


def model_pl_spectrum(
    d0_nm: float,
    sigma_ln: float,
    amplitude_ev: float,
    xi_nm: float,
    broaden_ev: float = 0.15,
    stokes_ev: float = 0.0,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulated PL spectrum of a GQD ensemble.

    Returns (energy_grid, spectrum_normalized, peak_energy_ev). The spectrum
    integrates to 1; the peak energy is where it is maximal.
    """
    if grid is None:
        grid = np.linspace(0.5, 6.5, 1200)

    sizes = lognormal_sizes(d0_nm, sigma_ln)
    energies = emission_energy_ev(sizes, amplitude_ev, xi_nm) - stokes_ev

    spectrum = np.zeros_like(grid)
    step = max(1, len(energies) // 6000)
    sampled = energies[::step]
    prefactor = 1.0 / (broaden_ev * np.sqrt(2.0 * np.pi) * len(sampled))
    for energy in sampled:
        spectrum += np.exp(-0.5 * ((grid - energy) / broaden_ev) ** 2)
    spectrum *= prefactor

    integral = np.trapezoid(spectrum, grid)
    spectrum = spectrum / integral
    peak = float(grid[np.argmax(spectrum)])
    return grid, spectrum, peak
