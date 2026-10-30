"""Nearest-neighbour tight-binding (Huckel) for pi-electron systems.

    H = -t * sum_<ij> (c_i^dag c_j + h.c.)

with t ~ 2.7 eV for graphene. For a bipartite system the spectrum is
particle-hole symmetric; if the two sublattices are imbalanced by
S = |N_A - N_B|, Lieb's theorem guarantees at least S exact zero modes
(degenerate states at E = 0 carrying S unpaired spins for t > 0, U -> 0+).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattices import Flake

HOPPING_EV = 2.7


def build_hamiltonian(flake: Flake, t: float = HOPPING_EV) -> np.ndarray:
    """Dense nearest-neighbour Hamiltonian (bonds = pairs at nn distance)."""
    n = flake.n_atoms
    hamiltonian = np.zeros((n, n))
    diff = flake.positions[:, None, :] - flake.positions[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    bonds = np.abs(distances - 1.0) < 1e-6
    hamiltonian[bonds] = -t
    np.fill_diagonal(hamiltonian, 0.0)
    return hamiltonian


@dataclass
class Spectrum:
    """Sorted eigenvalues of a Huckel Hamiltonian at half filling."""

    eigenvalues: np.ndarray

    @property
    def n(self) -> int:
        return len(self.eigenvalues)

    @property
    def homo_lumo_gap(self) -> float:
        """Gap between the highest occupied and lowest unoccupied levels
        (valid for even N; for odd N use ``zero_modes`` instead)."""
        if self.n % 2 != 0:
            raise ValueError("odd atom count: half filling puts an electron in a zero mode; use zero_modes()")
        occupied = self.eigenvalues[self.n // 2 - 1]
        empty = self.eigenvalues[self.n // 2]
        return float(empty - occupied)

    def zero_modes(self, tol: float = 1e-8) -> int:
        """Number of eigenvalues at E = 0 within tolerance."""
        return int((np.abs(self.eigenvalues) < tol).sum())


def solve(flake: Flake, t: float = HOPPING_EV) -> Spectrum:
    """Diagonalize the flake Hamiltonian (exact, dense LAPACK)."""
    eigenvalues = np.linalg.eigvalsh(build_hamiltonian(flake, t))
    return Spectrum(eigenvalues=eigenvalues)
