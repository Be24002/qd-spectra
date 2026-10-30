"""Simulated emission spectra of GQD ensembles vs the visible window.

Produces figures/emission_ensembles.png and results/emission.csv.

The scientific point: under the zigzag edge-state law E(d) = E1 exp(-d/xi)
measured by exact diagonalization, sub-1.5 nm dots emit across the visible
range, while dots of several nm fall into the infrared. Dots measured at
several nm that still emit visible light therefore reveal emission from
edge/functional states (sp3 defects, C=O groups) -- a testable statement
for the lab's PL data.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qdspectra.observables import fit_exponential_decay, hexagonal_gap_series
from qdspectra.spectra import model_pl_spectrum

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

rows = hexagonal_gap_series(max_n=8)[1:]
e1, xi, r2 = fit_exponential_decay(
    [r["diameter_nm"] for r in rows], [r["gap_ev"] for r in rows]
)
print(f"Edge-state law from the coronene series: E1 = {e1:.2f} eV, xi = {xi:.3f} nm (R2 = {r2:.5f})")

ensembles = [
    {"d0_nm": 0.50, "sigma_ln": 0.25, "label": "$d_0$ = 0.5 nm"},
    {"d0_nm": 0.80, "sigma_ln": 0.25, "label": "$d_0$ = 0.8 nm"},
    {"d0_nm": 1.10, "sigma_ln": 0.25, "label": "$d_0$ = 1.1 nm"},
    {"d0_nm": 1.60, "sigma_ln": 0.25, "label": "$d_0$ = 1.6 nm"},
]

fig, ax = plt.subplots(figsize=(8, 4.8))
peaks = []
for spec in ensembles:
    grid, spectrum, peak = model_pl_spectrum(
        spec["d0_nm"], spec["sigma_ln"], e1, xi, broaden_ev=0.15
    )
    ax.plot(grid, spectrum, label=f'{spec["label"]}, peak = {peak:.2f} eV')
    peaks.append((spec["d0_nm"], spec["sigma_ln"], peak))

ax.axvspan(1.60, 3.10, color="gold", alpha=0.18, label="visible window (1.6–3.1 eV)")
ax.set_xlabel("emission energy (eV)")
ax.set_ylabel("normalized PL intensity")
ax.set_title("Simulated GQD ensemble emission (edge-state law, lognormal sizes)")
ax.set_xlim(0.5, 6.5)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "emission_ensembles.png", dpi=150)
plt.close(fig)

for d0, sigma, peak in peaks:
    wavelength = 1239.84 / peak
    band = "visible" if 1.6 <= peak <= 3.1 else ("UV" if peak > 3.1 else "IR")
    print(f"d0 = {d0:.2f} nm -> emission peak {peak:.2f} eV ({wavelength:.0f} nm, {band})")

d_visible = xi * np.log(e1 / 3.1)
print(
    f"Largest diameter that still emits visible light (E >= 3.1 eV red edge... "
    f"or <= 3.1): d(E=3.1 eV) = {d_visible:.2f} nm"
)
print("=> dots of several nm emitting visible light require edge/functional states.")

with (RESULTS / "emission.csv").open("w") as handle:
    handle.write("median_diameter_nm,sigma_ln,peak_energy_ev,peak_wavelength_nm\n")
    for d0, sigma, peak in peaks:
        handle.write(f"{d0:.2f},{sigma:.2f},{peak:.4f},{1239.84 / peak:.1f}\n")
print("Wrote figures/emission_ensembles.png, results/emission.csv")
