"""Flake survey: coronene-series spectra, Lieb zero modes, edge-state law.

Produces figures/flakes_gallery.png, figures/gap_vs_size.png and
results/flakes.csv.

Headline result: the HOMO-LUMO gap of zigzag hexagonal flakes decays
EXPONENTIALLY with diameter, gap = E1 exp(-d/xi), because the frontier
levels are edge-localized states whose splitting is the tunneling coupling
between opposite zigzag edges (an effective power-law exponent drifts
upward with the fit window -- the fingerprint of the exponential law).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qdspectra.lattices import hexagonal_flake
from qdspectra.observables import (
    fit_exponential_decay,
    fit_power_law,
    hexagonal_gap_series,
    triangular_gap_series,
)
from qdspectra.tightbinding import HOPPING_EV

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

print("Solving hexagonal (coronene) flakes n = 1..8 ...")
hexa = hexagonal_gap_series(max_n=8)
for row in hexa:
    print(
        f"  C_{row['atoms']:<4d} d = {row['diameter_nm']:.3f} nm  "
        f"gap = {row['gap_ev']:.3f} eV  zero modes = {row['zero_modes']}"
    )

print("Solving triangular non-Kekule flakes ...")
tri = triangular_gap_series(max_k=4)
for row in tri:
    print(
        f"  k={row['k']}  C_{row['atoms']:<3d} imbalance = {row['imbalance']}  "
        f"zero modes = {row['zero_modes']}  (Lieb: >= {row['imbalance']})"
    )

diameters = np.array([r["diameter_nm"] for r in hexa])
gaps = np.array([r["gap_ev"] for r in hexa])
e1, xi, r2_exp = fit_exponential_decay(diameters[1:], gaps[1:])
alpha_eff, r2_pow = fit_power_law(diameters[1:], gaps[1:])
print(f"exponential fit gap = E1 exp(-d/xi): E1 = {e1:.2f} eV, xi = {xi:.3f} nm, R2 = {r2_exp:.5f}")
print(f"  (compare E1 with 2t = {2 * HOPPING_EV:.1f} eV, the benzene gap)")
print(f"power-law fit (for contrast): alpha_eff = {alpha_eff:.2f}, R2 = {r2_pow:.5f}")

# ---- Figure 1: flakes gallery -------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
for ax, n in zip(axes, (1, 2, 3, 5)):
    flake = hexagonal_flake(n)
    colors = np.where(flake.sublattice == 0, "C0", "C3")
    ax.scatter(flake.positions[:, 0], flake.positions[:, 1], c=colors, s=42 * 6 / n)
    ax.set_title(f"C$_{{{flake.n_atoms}}}$  ($d$ = {flake.diameter_nm:.2f} nm)")
    ax.set_aspect("equal")
    ax.axis("off")
fig.suptitle("Zigzag hexagonal graphene flakes (coronene series): blue/red = A/B sublattices")
fig.tight_layout()
fig.savefig(FIGURES / "flakes_gallery.png", dpi=150)
plt.close(fig)

# ---- Figure 2: gap vs size ----------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.semilogy(diameters, gaps, "o", color="C0", label="exact Hückel spectra")
ref_d = np.linspace(diameters[1] * 0.9, diameters[-1] * 1.15, 100)
ax.semilogy(ref_d, e1 * np.exp(-ref_d / xi), "--", color="C3",
            label=f"$E_1\\,e^{{-d/\\xi}}$, $E_1$ = {e1:.2f} eV, $\\xi$ = {xi:.2f} nm")
ax.semilogy(diameters[0], gaps[0], "s", color="C2",
            label=f"benzene gap = $2t$ = {2 * HOPPING_EV:.1f} eV")
ax.set_xlabel("flake diameter $d$ (nm)")
ax.set_ylabel("HOMO–LUMO gap (eV)")
ax.set_title(f"Zigzag edge-state law: exponential fit, $R^2$ = {r2_exp:.4f}")
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "gap_vs_size.png", dpi=150)
plt.close(fig)

# ---- CSV ------------------------------------------------------------------
with (RESULTS / "flakes.csv").open("w") as handle:
    handle.write("shape,size_index,atoms,diameter_nm,gap_ev,imbalance,zero_modes\n")
    for row in hexa:
        handle.write(
            f"hexagon,{row['n']},{row['atoms']},{row['diameter_nm']:.4f},"
            f"{row['gap_ev']:.4f},{row['imbalance']},{row['zero_modes']}\n"
        )
    for row in tri:
        handle.write(
            f"triangle,{row['k']},{row['atoms']},{row['diameter_nm']:.4f},"
            f"first_excited={row['first_excited_ev']:.4f},{row['imbalance']},{row['zero_modes']}\n"
        )
print("Wrote figures/flakes_gallery.png, figures/gap_vs_size.png, results/flakes.csv")
