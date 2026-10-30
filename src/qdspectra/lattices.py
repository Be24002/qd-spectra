"""Atomistic builders for graphene nanoflakes and molecular chains.

Conventions
-----------
- Lengths are in units of the nearest-neighbour C-C distance (nn = 1);
  the physical scale is a_CC = 0.142 nm.
- Primitive vectors of the graphene Bravais lattice (nn units):
  a1 = (3/2, sqrt(3)/2), a2 = (3/2, -sqrt(3)/2).
- Sublattice A sites at n1*a1 + n2*a2; sublattice B at (1,0) + n1*a1 + n2*a2.
- Hexagon (ring) centers form a triangular lattice at
  (1/2, sqrt(3)/2) + m1*a1 + m2*a2.

Builders
--------
- Hexagonal patches of rings give the zigzag-edge *coronene series*
  C_{6n^2}: benzene C6, coronene C24, circumcoronene C54, ... with balanced
  sublattices.
- Triangular patches give non-Kekule flakes (phenalenyl C13, triangulene
  C22, ...) with sublattice imbalance and the Lieb zero modes.
- Polyene chains give the textbook 1D limit with an analytic HOMO-LUMO gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A_CC_NM = 0.142
A1 = np.array([1.5, np.sqrt(3) / 2])
A2 = np.array([1.5, -np.sqrt(3) / 2])
_RING_OFFSET = np.array([0.5, np.sqrt(3) / 2])
_ANGLES = np.radians(60.0 * np.arange(6))


@dataclass
class Flake:
    """A nanoflake: atom positions (nn units) and sublattice labels."""

    positions: np.ndarray  # (N, 2)
    sublattice: np.ndarray  # 0 = A, 1 = B
    shape: str
    n: int  # size index of the series

    @property
    def n_atoms(self) -> int:
        return len(self.positions)

    @property
    def imbalance(self) -> int:
        n_a = int((self.sublattice == 0).sum())
        n_b = int((self.sublattice == 1).sum())
        return abs(n_a - n_b)

    @property
    def diameter_nm(self) -> float:
        center = self.positions.mean(axis=0)
        radius = np.max(np.linalg.norm(self.positions - center, axis=1))
        return float(2.0 * radius * A_CC_NM)


def _rings_hexagon(k: int) -> list[np.ndarray]:
    """Hexagonal patch of rings: |m1|, |m2|, |m1+m2| <= k -> 3k(k+1)+1 rings."""
    rings = []
    for m1 in range(-k, k + 1):
        for m2 in range(-k, k + 1):
            if abs(m1 + m2) <= k:
                rings.append(_RING_OFFSET + m1 * A1 + m2 * A2)
    return rings


def _rings_triangle(k: int) -> list[np.ndarray]:
    """Triangular patch of rings: m1 >= 0, m2 >= 0, m1 + m2 <= k."""
    rings = []
    for m1 in range(0, k + 1):
        for m2 in range(0, k + 1 - m1):
            rings.append(_RING_OFFSET + m1 * A1 + m2 * A2)
    return rings


def _atoms_from_rings(rings: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Deduped atom positions from ring centers, with sublattice labels."""
    unique: list[np.ndarray] = []
    for center in rings:
        for theta in _ANGLES:
            pos = center + np.array([np.cos(theta), np.sin(theta)])
            if not any(np.linalg.norm(pos - u) < 1e-6 for u in unique):
                unique.append(pos)

    basis = np.array([A1, A2]).T
    labels = []
    for pos in unique:
        coeffs_a = np.linalg.solve(basis, pos)
        coeffs_b = np.linalg.solve(basis, pos - np.array([1.0, 0.0]))
        on_a = np.allclose(coeffs_a, np.round(coeffs_a), atol=1e-6)
        on_b = np.allclose(coeffs_b, np.round(coeffs_b), atol=1e-6)
        labels.append(0 if on_a else 1)

    return np.array(unique), np.array(labels)


def hexagonal_flake(n: int) -> Flake:
    """Zigzag hexagonal flake of the coronene series: C_{6 n^2} atoms.

    n = 1 -> benzene (C6), n = 2 -> coronene (C24), n = 3 -> circumcoronene
    (C54), ...
    """
    positions, sublattice = _atoms_from_rings(_rings_hexagon(n - 1))
    return Flake(positions=positions, sublattice=sublattice, shape="hexagon", n=n)


def triangular_flake(k: int) -> Flake:
    """Triangular non-Kekule flake: k = 1 -> phenalenyl (C13, imbalance 1),
    k = 2 -> triangulene (C22, imbalance 2), k = 3 -> C33 (imbalance 3)."""
    positions, sublattice = _atoms_from_rings(_rings_triangle(k))
    return Flake(positions=positions, sublattice=sublattice, shape="triangle", n=k)


def polyene_chain(n_sites: int) -> Flake:
    """Linear polyene chain of ``n_sites`` carbons (1D reference system)."""
    positions = np.column_stack([np.arange(n_sites, dtype=float), np.zeros(n_sites)])
    sublattice = np.arange(n_sites) % 2
    return Flake(positions=positions, sublattice=sublattice, shape="chain", n=n_sites)
