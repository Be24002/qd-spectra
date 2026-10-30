"""Gap-versus-size scans and confinement-law fits for graphene flakes."""

from __future__ import annotations

import numpy as np
from scipy.stats import linregress

from .lattices import Flake, hexagonal_flake, triangular_flake
from .tightbinding import HOPPING_EV, Spectrum, solve


def hexagonal_gap_series(max_n: int = 8, t: float = HOPPING_EV) -> list[dict]:
    """Gap of the coronene series C_{6n^2} for n = 1..max_n."""
    rows = []
    for n in range(1, max_n + 1):
        flake = hexagonal_flake(n)
        spectrum = solve(flake, t)
        rows.append(
            {
                "shape": "hexagon",
                "n": n,
                "atoms": flake.n_atoms,
                "diameter_nm": flake.diameter_nm,
                "gap_ev": spectrum.homo_lumo_gap,
                "imbalance": flake.imbalance,
                "zero_modes": spectrum.zero_modes(),
            }
        )
    return rows


def triangular_gap_series(max_k: int = 4, t: float = HOPPING_EV) -> list[dict]:
    """Zero-mode census of triangular non-Kekule flakes (Lieb's theorem)."""
    rows = []
    for k in range(1, max_k + 1):
        flake = triangular_flake(k)
        spectrum = solve(flake, t)
        positive = spectrum.eigenvalues[spectrum.eigenvalues > 1e-8]
        rows.append(
            {
                "shape": "triangle",
                "k": k,
                "atoms": flake.n_atoms,
                "diameter_nm": flake.diameter_nm,
                "imbalance": flake.imbalance,
                "zero_modes": spectrum.zero_modes(),
                "first_excited_ev": float(positive[0]),
            }
        )
    return rows


def fit_exponential_decay(diameters_nm, gaps_ev) -> tuple[float, float, float]:
    """Fit gap = E1 * exp(-d/xi) -- the zigzag edge-state law.

    The frontier levels of zigzag flakes are edge-localized states whose
    splitting is the tunneling coupling between opposite edges, so it decays
    exponentially with size (not as a power law). Returns (E1_eV, xi_nm, r2).
    """
    fit = linregress(np.asarray(diameters_nm), np.log(np.asarray(gaps_ev)))
    e1 = float(np.exp(fit.intercept))
    xi = float(-1.0 / fit.slope)
    return e1, xi, float(fit.rvalue**2)


def fit_power_law(diameters_nm, gaps_ev) -> tuple[float, float]:
    """Fit gap = C * d^(-alpha) in log-log; returns (alpha, r_value).

    For zigzag flakes the effective alpha drifts upward with the size window
    -- the signature that the true law is exponential, not a power law.
    """
    fit = linregress(np.log(diameters_nm), np.log(gaps_ev))
    return float(-fit.slope), float(fit.rvalue**2)
