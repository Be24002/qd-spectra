"""Physics regression tests: textbook Huckel limits, Lieb's theorem,
coronene-series sizes, confinement scaling, and emission behaviour."""

import numpy as np
import pytest

from qdspectra.lattices import hexagonal_flake, polyene_chain, triangular_flake
from qdspectra.observables import (
    fit_exponential_decay,
    hexagonal_gap_series,
    triangular_gap_series,
)
from qdspectra.spectra import model_pl_spectrum
from qdspectra.tightbinding import HOPPING_EV, solve

T = HOPPING_EV


# ------------------------------------------------------------- textbooks


def test_benzene_hueckel_levels():
    # Benzene: energies {-2t, -t, -t, +t, +t, +2t} -- the original Huckel result.
    e = solve(hexagonal_flake(1)).eigenvalues
    expected = np.array([-2 * T, -T, -T, T, T, 2 * T])
    assert np.allclose(e, expected, atol=1e-10)


def test_benzene_gap_is_2t():
    assert solve(hexagonal_flake(1)).homo_lumo_gap == pytest.approx(2 * T, abs=1e-10)


def test_polyene_gap_matches_analytic():
    # N-site chain: gap = 4t sin(pi / (2(N+1))) -- analytic Huckel result.
    for n_sites in (8, 10):
        gap = solve(polyene_chain(n_sites)).homo_lumo_gap
        assert gap == pytest.approx(4 * T * np.sin(np.pi / (2 * (n_sites + 1))), rel=1e-10)


def test_spectrum_is_particle_hole_symmetric():
    # Bipartite lattice: eigenvalues come in +/- pairs.
    spectrum = solve(hexagonal_flake(3))
    assert np.allclose(spectrum.eigenvalues, -spectrum.eigenvalues[::-1], atol=1e-10)


# -------------------------------------------------------------- builders


def test_coronene_series_sizes():
    sizes = [hexagonal_flake(n).n_atoms for n in (1, 2, 3, 4)]
    assert sizes == [6, 24, 54, 96]


def test_hexagonal_flakes_are_sublattice_balanced():
    for n in (2, 3, 4):
        assert hexagonal_flake(n).imbalance == 0


def test_phenalenyl_and_triangulene_geometry():
    assert triangular_flake(1).n_atoms == 13   # phenalenyl radical
    assert triangular_flake(2).n_atoms == 22   # triangulene
    assert triangular_flake(1).imbalance == 1
    assert triangular_flake(2).imbalance == 2


def test_lieb_theorem_zero_modes():
    # Lieb (1989): a bipartite pi system with imbalance S has >= S zero modes.
    for k in (1, 2, 3):
        flake = triangular_flake(k)
        spectrum = solve(flake)
        assert spectrum.zero_modes() >= flake.imbalance


# --------------------------------------------------------------- scaling


def test_gap_decreases_with_size():
    gaps = [row["gap_ev"] for row in hexagonal_gap_series(max_n=6)]
    assert all(g2 < g1 for g1, g2 in zip(gaps, gaps[1:]))


def test_gap_decays_exponentially():
    # Zigzag flakes: the frontier levels are edge states; their splitting is
    # the edge-to-edge tunneling coupling and decays exponentially. A power
    # law would show a drifting effective exponent (measured: 1.1 -> 3.4 as
    # the window grows), so we assert the exponential law directly.
    rows = hexagonal_gap_series(max_n=8)[1:]  # skip the molecular benzene
    e1, xi, r2 = fit_exponential_decay(
        [r["diameter_nm"] for r in rows], [r["gap_ev"] for r in rows]
    )
    assert r2 > 0.995
    assert 0.9 < xi < 1.5          # edge-state hybridization length (nm)
    assert 3.5 < e1 < 7.0          # anchors near the benzene gap 2t = 5.4 eV


# --------------------------------------------------------------- emission


def test_emission_redshifts_with_growing_dots():
    # E(d) = E1 exp(-d/xi): bigger dots emit redder.
    _, _, peak_small = model_pl_spectrum(0.5, 0.3, 5.4, 1.2)
    _, _, peak_large = model_pl_spectrum(1.5, 0.3, 5.4, 1.2)
    assert peak_small > peak_large


def test_emission_spectrum_is_normalized():
    grid, spectrum, _ = model_pl_spectrum(0.8, 0.3, 5.4, 1.2)
    assert np.trapezoid(spectrum, grid) == pytest.approx(1.0, rel=1e-3)
