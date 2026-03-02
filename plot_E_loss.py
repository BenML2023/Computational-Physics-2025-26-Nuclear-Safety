"""
Plot: Energy Loss vs Incoming Neutron Energy (log x-axis)
         for U235 and U238 inelastic scattering

Each discrete inelastic level (MT=51–90) contributes a curve showing how much
energy the neutron loses at each incident energy above that level's threshold.

Energy loss = E_in - E_out
where E_out = (A/(A+1)) * ((A/(A+1)) * E_in + Q_eV)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os

# ── make sure cs_getter.py is importable ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from cs_getter import NuclearData

DATA_DIR = "/Users/benweihrauch/Desktop/Nuclear Safety/Code/master/NuclearData"

# ── helper ────────────────────────────────────────────────────────────────────


def compute_energy_loss_curves(db, isotope):
    """
    Returns a list of dicts, one per open inelastic level:
        {level, Q_eV, E_thresh, E_in_arr, delta_E_arr}
    """
    mass_factor, levels = db.get_inelastic_levels(isotope)
    A = db.cache[isotope].atomic_weight_ratio

    curves = []
    for lv in levels:
        E_thresh = lv["E_thresh"]
        # Energy grid: from threshold up to 20 MeV, 500 log-spaced points
        E_in_arr = np.logspace(
            np.log10(max(E_thresh * 1.001, 1e3)),  # just above threshold (eV)
            np.log10(20e6),  # 20 MeV (eV)
            500,
        )

        E_out_arr = (A / (A + 1)) * ((A / (A + 1)) * E_in_arr + lv["Q_eV"])
        # Only keep physically valid points (E_out > 0)
        mask = E_out_arr > 0
        if mask.sum() < 2:
            continue

        delta_E = (E_in_arr[mask] - E_out_arr[mask]) / 1e6  # → MeV

        curves.append(
            dict(
                level=lv["level"],
                Q_eV=lv["Q_eV"],
                E_thresh=E_thresh,
                E_in_MeV=E_in_arr[mask] / 1e6,
                delta_E_MeV=delta_E,
            )
        )
    return curves


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    db = NuclearData(data_dir=DATA_DIR)

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharey=False)
    fig.suptitle(
        "Inelastic Scattering — Energy Loss per Discrete Level",
        fontsize=16,
        fontweight="bold",
    )

    for ax, isotope in zip(axes, ["U235", "U238"]):
        curves = compute_energy_loss_curves(db, isotope)

        # Colour-map: one colour per level
        cmap = cm.get_cmap("viridis", max(len(curves), 1))

        for i, c in enumerate(curves):
            ax.plot(
                c["E_in_MeV"],
                c["delta_E_MeV"],
                color=cmap(i),
                linewidth=0.9,
                label=f"Level {c['level']}  (Q={c['Q_eV']/1e3:.1f} keV)",
            )

        ax.set_xscale("log")
        ax.set_xlabel("Incoming Neutron Energy  [MeV]", fontsize=16)
        ax.set_ylabel("Energy Loss  ΔE = E_in − E_out  [MeV]", fontsize=16)
        ax.set_title(f"{isotope} Inelastic Scattering", fontsize=16)
        ax.set_xlim(1e-3, 20)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.tick_params(axis="both", which="minor", labelsize=12)

        # Show legend only for a manageable number of levels
        n_levels = len(curves)
        if n_levels <= 20:
            ax.legend(fontsize=8, ncol=2, loc="upper left")
        else:
            # Annotate colourbar instead
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, n_levels))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label("Level index", fontsize=14)
            cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "inelastic_energy_loss.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
