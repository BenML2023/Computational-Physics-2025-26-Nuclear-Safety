"""
Shannon Entropy vs Cycle — minimal script
==========================================
Uses the existing `entropy_samples` already returned by run_keff_accel
(2nd return value, one value per cycle covering inactive + active).

Steps
-----
1. Pre-tabulate fuel once.
2. Find R_crit via the parallel scan (same seed = BASELINE_SEED).
3. Run ONE simulation at R_crit with the SAME seed → capture entropy & keff.
4. Plot and save.

Nothing in fast_reactor_accel.py needs to be changed.
"""

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(__file__))

from fast_reactor_optimized import get_fresh_u235_fuel, Sphere, Cube, Cylinder
from fast_reactor_accel import run_keff_accel, PreTabulatedFuel

from scipy.interpolate import interp1d
from scipy.optimize import bisect
from multiprocessing import Pool

# ── tuneable parameters ────────────────────────────────────────────────────────
GEOMETRY_TYPE = "sphere"  # "sphere" | "cube" | "cylinder"
BASELINE_SEED = 0

N_NEUTRONS = 200
N_INACTIVE = 25
N_ACTIVE = 150

N_SCAN_POINTS = 7
N_WORKERS = 12
CYLINDER_H_RATIO = 2.0

PARAM_RANGES = {
    "sphere": (7.0, 9.5),
    "cube": (12.0, 19.0),
    "cylinder": (6.0, 10.0),
}
# ──────────────────────────────────────────────────────────────────────────────


# ── geometry factory ──────────────────────────────────────────────────────────
def make_geometry(gtype, param):
    if gtype == "sphere":
        return Sphere(radius=param)
    elif gtype == "cube":
        return Cube(side=param)
    elif gtype == "cylinder":
        return Cylinder(radius=param, height=CYLINDER_H_RATIO * param)
    raise ValueError(gtype)


# ── pool worker (module-level required by multiprocessing) ────────────────────
_FAST_FUEL = None


def _init(ff):
    global _FAST_FUEL
    _FAST_FUEL = ff


def _worker(task):
    gtype, param, seed = task
    geom = make_geometry(gtype, param)
    try:
        keff_cycles, _, _ = run_keff_accel(
            geom,
            _FAST_FUEL.fuel,
            n_neutrons=N_NEUTRONS,
            n_inactive=N_INACTIVE,
            n_active=N_ACTIVE,
            seed=seed,
            verbose=False,
            fast_fuel=_FAST_FUEL,
        )
        if len(keff_cycles) > 0:
            return param, float(np.mean(keff_cycles))
    except Exception as e:
        print(f"  worker error param={param:.3f}: {e}")
    return param, None


# ── criticality scan ──────────────────────────────────────────────────────────
def find_r_crit(fast_fuel, gtype, seed_base):
    lo, hi = PARAM_RANGES[gtype]
    params = np.linspace(lo, hi, N_SCAN_POINTS)

    tasks = [(gtype, float(p), seed_base + int(p * 1000)) for p in params]

    print(f"  Scanning {N_SCAN_POINTS} points with {N_WORKERS} workers …")
    with Pool(
        processes=min(N_WORKERS, len(tasks)), initializer=_init, initargs=(fast_fuel,)
    ) as pool:
        raw = pool.map(_worker, tasks)

    pts, ks = [], []
    for p, k in raw:
        if k is not None:
            pts.append(p)
            ks.append(k)
            print(f"    param={p:.3f} cm  k={k:.4f}")

    if len(pts) < 3:
        raise RuntimeError("Too few valid k_eff points — widen PARAM_RANGES.")

    pv, kv = np.array(pts), np.array(ks)
    try:
        poly = np.poly1d(np.polyfit(pv, kv, 2))
        root = bisect(lambda x: poly(x) - 1.0, pv.min(), pv.max())
        print(f"  R_crit = {root:.5f} cm  (k ~ {poly(root):.4f})")
        return float(root)
    except Exception:
        fn = interp1d(kv, pv, kind="linear", fill_value="extrapolate")
        root = float(fn(1.0))
        print(f"  R_crit = {root:.5f} cm  (linear fallback)")
        return root


# ── single simulation — captures entropy directly ────────────────────────────
def run_with_entropy(fast_fuel, gtype, param, seed):
    """
    Calls run_keff_accel once and returns:
        keff_cycles  — np.ndarray, shape (n_active,)
        entropy_all  — np.ndarray, shape (n_inactive + n_active,)
                       one Shannon entropy value per cycle, starting from
                       cycle 0 (before any neutrons have moved).
    """
    geom = make_geometry(gtype, param)
    keff_cycles, entropy_all, _ = run_keff_accel(
        geom,
        fast_fuel.fuel,
        n_neutrons=N_NEUTRONS,
        n_inactive=N_INACTIVE,
        n_active=N_ACTIVE,
        seed=seed,
        verbose=True,
        fast_fuel=fast_fuel,
    )
    return np.array(keff_cycles), np.array(entropy_all)


# ── plot ──────────────────────────────────────────────────────────────────────
def plot_entropy(keff_cycles, entropy_all, gtype, R_crit, seed):
    n_inact = N_INACTIVE
    n_tot = len(entropy_all)
    cycles = np.arange(1, n_tot + 1)

    ent_inactive = entropy_all[:n_inact]
    ent_active = entropy_all[n_inact:]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"Shannon Entropy of Fission Source  —  {gtype.capitalize()}\n"
        f"R_crit = {R_crit:.4f} cm  |  albedo = 0  |  seed = {seed}",
        fontsize=16,
        fontweight="bold",
    )

    ax.axvspan(1, n_inact, alpha=0.12, color="#95A5A6", label="Inactive")
    ax.axvspan(n_inact + 1, n_tot, alpha=0.10, color="#27AE60", label="Active")
    ax.axvline(n_inact + 0.5, color="#E74C3C", lw=1.8, ls="--", label="Active start")

    ax.plot(
        cycles,
        entropy_all,
        "o-",
        color="#2980B9",
        ms=4,
        lw=1.5,
        alpha=0.85,
        label="Shannon H",
    )

    # running mean over active cycles
    if len(ent_active) > 0:
        rm = np.cumsum(ent_active) / np.arange(1, len(ent_active) + 1)
        ax.plot(
            np.arange(n_inact + 1, n_inact + len(rm) + 1),
            rm,
            "-",
            color="#E67E22",
            lw=2.5,
            label="Running mean (active)",
        )

    ax.set_xlabel("Cycle", fontsize=16, fontweight="bold")
    ax.set_ylabel("Shannon Entropy H (bits)", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(fontsize=16)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    path = os.path.join("plots", f"{gtype}_entropy_vs_cycle.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved → {os.path.abspath(path)}")
    return path


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    from multiprocessing import freeze_support

    freeze_support()

    print("=" * 60)
    print("  Shannon Entropy vs Cycle")
    print(f"  Geometry : {GEOMETRY_TYPE.upper()}  |  Seed : {BASELINE_SEED}")
    print("=" * 60)

    # 1. Tabulate once
    print("\nPre-tabulating nuclear data …")
    t0 = time.time()
    fuel = get_fresh_u235_fuel(reflector_albedo=0.0)
    ff = PreTabulatedFuel(fuel)
    print(f"  Done in {time.time()-t0:.1f} s")

    # 2. Find R_crit (same seed as the entropy run)
    print(f"\nLocating R_crit  (seed_base = {BASELINE_SEED}) …")
    R_crit = find_r_crit(ff, GEOMETRY_TYPE, seed_base=BASELINE_SEED)

    # 3. Single simulation at R_crit — SAME seed
    print(
        f"\nRunning single simulation at R = {R_crit:.4f} cm  (seed = {BASELINE_SEED}) …"
    )
    keff_cycles, entropy_all = run_with_entropy(
        ff, GEOMETRY_TYPE, R_crit, BASELINE_SEED
    )

    print(f"\n  k_eff (mean active) : {np.mean(keff_cycles):.5f}")
    print(f"  Entropy cycles      : {len(entropy_all)}")
    print(
        f"  Entropy range       : [{entropy_all.min():.4f}, {entropy_all.max():.4f}] bits"
    )

    # 4. Plot
    plot_entropy(keff_cycles, entropy_all, GEOMETRY_TYPE, R_crit, BASELINE_SEED)


if __name__ == "__main__":
    main()
