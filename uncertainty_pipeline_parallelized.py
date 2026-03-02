"""
Optimized Criticality Uncertainty Analysis Pipeline  (Parallel Edition)
========================================================================
KEY SPEEDUP vs original:
    Old: Every scan_parallel() → Pool(initializer) → 12 workers each
         tabulate all XS from HDF5 (~5-10s each).
         10 replicas + 13 perturbations = 23 scans × 12 workers = 276 tabulations
         → ~30-45 minutes just tabulating.

    New: Tabulate ONCE in the main process (~10s).
         Perturbations: copy numpy arrays + multiply one slice + recompute
         macros (~1 ms each).  Pool initializer stores pre-built arrays,
         no tabulation in workers.
         → ~10s total tabulation + ~12 ms for all perturbations.

Architecture
------------
1. Main process builds PreTabulatedFuel ONCE from baseline fuel.
2. Phase 1: ALL replica tasks dispatched to ONE pool.
   Workers receive the pre-built fast_fuel via initializer (just stores it).
3. Phase 2 baseline: same pool, same fast_fuel.
4. Phase 2 perturbations: perturb_fast_fuel() copies arrays + applies factor.
   One pool per perturbation with the modified fast_fuel.
   Workers store the pre-perturbed arrays — zero tabulation.

Perturbation correctness
------------------------
perturb_fast_fuel() modifies micro_xs[iso_idx, :, rxn_idx] *= factor (or
nu_bar for ν̄), then recomputes macro_total, isotope weights, and reaction
CDFs from the modified partials.  Scattering angular distributions are
unaffected — only cross-section magnitudes and ν̄ change.
"""

import copy
import os
import sys
import time
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import bisect

sys.path.append(os.path.dirname(__file__))

from fast_reactor_optimized import (
    Sphere,
    Cube,
    Cylinder,
    FastReactorFuel,
    get_fresh_u235_fuel,
)
from fast_reactor_accel import run_keff_accel, PreTabulatedFuel, perturb_fast_fuel


# =============================================================================
# SINGLE PARAMETER TO CHANGE GEOMETRY
GEOMETRY_TYPE: str = "sphere"  # "sphere" | "cube" | "cylinder"
# =============================================================================


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

N_NEUTRONS = 200
N_INACTIVE = 25
N_ACTIVE = 150

N_REPLICAS = 10
BASELINE_SEED = 0
PERTURBATION = 0.01  # +1%

N_SCAN_POINTS = 7
N_SIMS_PER_PT = 1
N_WORKERS = 12
REFLECTOR_ALBEDO = 0.0

PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "sphere": (7.0, 9.5),
    "cube": (12.0, 19.0),
    "cylinder": (6.0, 10.0),
}
CYLINDER_H_RATIO = 2.0

# Phase 3: Albedo reflector scan
ALBEDO_STEP = 0.2  # scan from 0.0 to MAX_ALBEDO in steps of this size
MAX_ALBEDO = 0.1  # maximum albedo to test
N_REPLICAS_P3 = 10  # replicas per albedo point for statistical uncertainty

# =============================================================================
# PHASE SELECTION — set to False to skip
# =============================================================================
RUN_PHASE1 = False  # statistical uncertainty (replicas)
RUN_PHASE2 = False  # systematic uncertainty (perturbations)
RUN_PHASE3 = True  # reflector albedo scan


# =============================================================================
# LITERATURE CROSS-SECTION UNCERTAINTIES  (fractional, 1σ)
# =============================================================================
# These are the evaluated nuclear data covariance uncertainties (ε_i) from
# ENDF/B-VIII.0 and related evaluations for the fast energy range.
# The systematic uncertainty on R_crit from each channel is:
#     ΔR_i = S_i × ε_i
# where S_i is the sensitivity (dR/dσ per unit fractional change).

LITERATURE_UNCERTAINTIES: Dict[str, float] = {
    "U-235 fission": 0.014,  # ~1.4%  — well-measured standard
    "U-235 capture": 0.050,  # ~5.0%  — harder to measure in fast spectrum
    "U-235 elastic": 0.030,  # ~3.0%
    "U-235 inelastic": 0.200,  # ~20%   — large uncertainty, few measurements
    "U-238 fission": 0.012,  # ~1.2%  — threshold reaction, well-constrained
    "U-238 capture": 0.015,  # ~1.5%  — important for breeding ratio
    "U-238 elastic": 0.030,  # ~3.0%
    "U-238 inelastic": 0.150,  # ~15%   — dominant spectrum shaper, poorly known
    "nu U-235": 0.005,  # ~0.5%  — standards measurement
    "nu U-238": 0.015,  # ~1.5%
}

# Cross-correlation between U-235 fission and U-235 capture.
# These share evaluation constraints (sum rules, resolved resonance fits),
# so a positive correlation exists — increasing fission typically means
# decreasing capture in the evaluation, but the covariance is positive
# because measurement systematics affect both.
RHO_U235_FISS_CAPT: float = 0.5


# =============================================================================
# GEOMETRY FACTORY
# =============================================================================


def make_geometry(geometry_type: str, param: float):
    if geometry_type == "sphere":
        return Sphere(radius=param)
    elif geometry_type == "cube":
        return Cube(side=param)
    elif geometry_type == "cylinder":
        return Cylinder(radius=param, height=CYLINDER_H_RATIO * param)
    else:
        raise ValueError(f"Unknown GEOMETRY_TYPE: {geometry_type!r}")


# =============================================================================
# POOL INITIALIZER + WORKER  (module-level for multiprocessing)
# =============================================================================

_WORKER_FAST_FUEL: Optional[PreTabulatedFuel] = None


def _init_worker_prebuilt(fast_fuel: PreTabulatedFuel):
    """
    Called once per worker process.  Stores the PRE-BUILT fast_fuel.
    No tabulation — just a reference assignment.
    """
    global _WORKER_FAST_FUEL
    _WORKER_FAST_FUEL = fast_fuel


def _single_simulation_accel(task):
    """
    Worker for one (geometry_type, param, seed) simulation.
    Uses the pre-built _WORKER_FAST_FUEL — zero tabulation cost.
    """
    global _WORKER_FAST_FUEL
    geometry_type, param, seed = task

    geom = make_geometry(geometry_type, param)
    fuel = _WORKER_FAST_FUEL.fuel

    try:
        keff_cycles, _, _ = run_keff_accel(
            geom,
            fuel,
            n_neutrons=N_NEUTRONS,
            n_inactive=N_INACTIVE,
            n_active=N_ACTIVE,
            seed=seed,
            verbose=False,
            fast_fuel=_WORKER_FAST_FUEL,
        )
        if len(keff_cycles) > 0:
            return (param, float(np.mean(keff_cycles)))
    except Exception as exc:
        print(f"    worker exception at param={param:.3f}: {exc}")

    return (param, None)


# =============================================================================
# PARALLEL CRITICALITY SCAN
# =============================================================================


def scan_parallel(
    fast_fuel: PreTabulatedFuel,
    geometry_type: str,
    seed_base: int = 0,
    label: str = "",
) -> float:
    """
    Parallel criticality search using a PRE-BUILT fast_fuel.
    Workers receive the tabulated arrays via initializer — no re-tabulation.
    """
    lo, hi = PARAM_RANGES[geometry_type]
    params = np.linspace(lo, hi, N_SCAN_POINTS)

    tasks = []
    for p in params:
        for sim_id in range(N_SIMS_PER_PT):
            seed = seed_base + int(p * 1000) * (sim_id + 1)
            tasks.append((geometry_type, float(p), seed))

    n_workers = min(N_WORKERS, len(tasks))
    tag = f" [{label}]" if label else ""
    print(f"  Pool: {n_workers} workers, {len(tasks)} tasks{tag}")

    with Pool(
        processes=n_workers,
        initializer=_init_worker_prebuilt,
        initargs=(fast_fuel,),
    ) as pool:
        raw = pool.map(_single_simulation_accel, tasks)

    # Aggregate
    keff_dict: Dict[float, List[float]] = {float(p): [] for p in params}
    for p, k in raw:
        if k is not None:
            keff_dict[p].append(k)

    valid_params, valid_keffs = [], []
    for p in params:
        vals = keff_dict[float(p)]
        if vals:
            valid_params.append(p)
            valid_keffs.append(float(np.mean(vals)))
            print(f"    param={p:.3f} cm  k={valid_keffs[-1]:.4f}")
        else:
            print(f"    param={p:.3f} cm  FAILED")

    if len(valid_params) < 3:
        raise RuntimeError(
            f"Only {len(valid_params)} valid k_eff points — widen PARAM_RANGES."
        )

    pv = np.array(valid_params)
    kv = np.array(valid_keffs)

    try:
        coeffs = np.polyfit(pv, kv, deg=2)
        poly = np.poly1d(coeffs)
        root = bisect(lambda x: poly(x) - 1.0, pv.min(), pv.max())
        print(f"  Critical dim: {root:.5f} cm  (k ~ {poly(root):.4f})")
        return float(root)
    except Exception:
        pass

    try:
        interp_fn = interp1d(kv, pv, kind="linear", fill_value="extrapolate")
        root = float(interp_fn(1.0))
        print(f"  Critical dim (linear fallback): {root:.5f} cm")
        return root
    except Exception as exc:
        raise RuntimeError(f"Could not locate critical dimension. Error: {exc}")


# =============================================================================
# PERTURBATION DEFINITIONS
# =============================================================================

MAJOR_PERTURBATIONS: List[Tuple[str, str, str]] = [
    ("U-235 fission", "U235", "fission"),
    ("U-235 capture", "U235", "capture"),
    ("U-235 elastic", "U235", "elastic"),
    ("U-235 inelastic", "U235", "inelastic"),
    ("U-238 fission", "U238", "fission"),
    ("U-238 capture", "U238", "capture"),
    ("U-238 elastic", "U238", "elastic"),
    ("U-238 inelastic", "U238", "inelastic"),
    ("nu U-235", "U235", "nu"),
    ("nu U-238", "U238", "nu"),
]

MINOR_PERTURBATIONS: List[Tuple[str, str, str]] = []


# =============================================================================
# SANDWICH-RULE SYSTEMATIC UNCERTAINTY
# =============================================================================


def compute_sandwich_uncertainty(
    sensitivity_table: List[Dict],
) -> Dict:
    """
    Compute the systematic uncertainty on R_crit using the sandwich rule:

        ΔR_i  = S_i × ε_i           (sensitivity × literature uncertainty)
        σ²_syst = Σ ΔR_i²  +  2ρ × ΔR(U235_fiss) × ΔR(U235_capt)

    The cross-correlation term accounts for the covariance between the
    U-235 fission and capture evaluations (shared resonance parameters).

    Returns dict with:
        delta_R_phys : {param: ΔR_i}   per-channel physical uncertainty
        sigma_syst   : float            total systematic σ
        corr_term    : float            the 2ρ cross-correlation contribution
        dominant     : str              channel with largest |ΔR_i|
    """
    delta_R_phys: Dict[str, float] = {}

    for row in sensitivity_table:
        param = row["parameter"]
        S_i = row["sensitivity"]  # dR / d(fractional change)
        eps_i = LITERATURE_UNCERTAINTIES.get(param, 0.0)
        delta_R_phys[param] = S_i * eps_i  # ΔR_i in cm

    # Sum of squares (diagonal terms)
    sum_sq = sum(dr**2 for dr in delta_R_phys.values())

    # Off-diagonal: U-235 fission ↔ capture cross-correlation
    dr_fiss = delta_R_phys.get("U-235 fission", 0.0)
    dr_capt = delta_R_phys.get("U-235 capture", 0.0)
    corr_term = 0  # 2.0 * RHO_U235_FISS_CAPT * dr_fiss * dr_capt

    variance = sum_sq  # + corr_term
    # Guard against negative variance from anti-correlated terms
    sigma_syst = float(np.sqrt(max(0.0, variance)))

    # Identify dominant channel
    dominant = (
        max(delta_R_phys, key=lambda k: abs(delta_R_phys[k])) if delta_R_phys else ""
    )

    return {
        "delta_R_phys": delta_R_phys,
        "sigma_syst": sigma_syst,
        "corr_term": corr_term,
        "dominant": dominant,
    }


# =============================================================================
# PHASE 1 — STATISTICAL UNCERTAINTY
# =============================================================================


def phase1_statistical(
    baseline_ff: PreTabulatedFuel,
) -> Tuple[List[float], float, float]:
    """
    Run N_REPLICAS independent criticality searches.
    All replicas use the SAME baseline fast_fuel — workers never re-tabulate.
    """
    sep = "=" * 72
    print(f"\n{sep}")
    print("  PHASE 1 - Statistical Uncertainty (Independent Replicas)")
    print(f"  {N_REPLICAS} replicas  |  {N_SCAN_POINTS} scan points each (parallel)")
    print(sep)

    replica_results: List[float] = []

    for i in range(N_REPLICAS):
        print(f"\n{'─'*60}")
        print(f"  Replica {i+1}/{N_REPLICAS}  |  seed_base = {i * 100_000}")
        print(f"{'─'*60}")
        t0 = time.time()
        try:
            R_crit = scan_parallel(
                baseline_ff,
                GEOMETRY_TYPE,
                seed_base=i * 100_000,
                label=f"replica {i+1}",
            )
            elapsed = time.time() - t0
            replica_results.append(R_crit)
            print(f"  OK  R_crit = {R_crit:.5f} cm   ({elapsed:.1f} s)")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  FAIL  Replica {i+1} ({elapsed:.1f} s): {exc}")

    R_bar = float(np.mean(replica_results)) if replica_results else float("nan")
    N_ok = len(replica_results)
    # Sample standard deviation: spread of individual replicas
    s_R = float(np.std(replica_results, ddof=1)) if N_ok > 1 else 0.0
    # Standard error of the mean: uncertainty on R̄
    SE_R = s_R / np.sqrt(N_ok) if N_ok > 1 else 0.0

    print(f"\n{'─'*60}")
    print(f"  Phase 1 complete  ({N_ok}/{N_REPLICAS} replicas ok)")
    print(f"  R-bar   = {R_bar:.5f} cm")
    print(f"  s       = {s_R:.5f} cm  (single-run spread)")
    print(f"  SE      = {SE_R:.5f} cm  (uncertainty on R̄ = s/√{N_ok})")
    print(f"{'─'*60}")

    return replica_results, R_bar, s_R, SE_R


# =============================================================================
# PHASE 2 — SYSTEMATIC UNCERTAINTY
# =============================================================================


def phase2_systematic(
    fuel: FastReactorFuel,
    baseline_ff: PreTabulatedFuel,
) -> Tuple[float, List[Dict]]:
    """
    Baseline + all perturbation runs.

    Key speedup: perturbations use perturb_fast_fuel() which copies the
    numpy arrays and multiplies one slice (~1 ms), instead of building a
    PerturbedNuclearData wrapper and re-tabulating from HDF5 (~10 s per
    worker process).
    """
    sep = "=" * 72
    print(f"\n{sep}")
    print("  PHASE 2 - Systematic Uncertainty (Sensitivity Analysis)")
    print(
        f"  Perturbation: +{PERTURBATION*100:.1f}%  |  Baseline seed: {BASELINE_SEED}"
    )
    print(sep)

    # Baseline — uses the same pre-built fast_fuel
    print(f"\n{'─'*60}")
    print("  Baseline run")
    print(f"{'─'*60}")
    t0 = time.time()
    R_baseline = scan_parallel(
        baseline_ff, GEOMETRY_TYPE, seed_base=BASELINE_SEED, label="baseline"
    )
    print(f"  OK  R_baseline = {R_baseline:.5f} cm   ({time.time()-t0:.1f} s)")

    sensitivity_table: List[Dict] = []
    factor = 1.0 + PERTURBATION

    def _run_one(
        label: str, isotope: str, reaction: str, pert_idx: int
    ) -> Optional[Dict]:
        pert_seed_base = BASELINE_SEED  # SAME seed as baseline — isolates the XS effect
        print(f"\n{'─'*60}")
        print(
            f"  Perturbation: {label}  [+{PERTURBATION*100:.1f}%]  seed_base={pert_seed_base} (= baseline)"
        )
        print(f"{'─'*60}")

        t0 = time.time()

        # FAST: perturb arrays (~1 ms) instead of re-tabulating (~10 s/worker)
        try:
            pert_ff = perturb_fast_fuel(baseline_ff, isotope, reaction, factor)
        except ValueError as exc:
            print(f"  SKIP  Cannot perturb '{label}': {exc}")
            return None

        t_pert = time.time() - t0
        print(f"  Array perturbation: {t_pert*1000:.1f} ms")

        R_pert = scan_parallel(
            pert_ff, GEOMETRY_TYPE, seed_base=pert_seed_base, label=label
        )
        elapsed = time.time() - t0
        delta_R = R_pert - R_baseline
        S = delta_R / PERTURBATION
        print(
            f"  OK  R_pert={R_pert:.5f}  dR={delta_R:+.5f} cm  "
            f"S={S:+.5f} cm/unit  ({elapsed:.1f} s)"
        )
        return {
            "parameter": label,
            "R_perturbed": R_pert,
            "delta_R": delta_R,
            "sensitivity": S,
        }

    # Major perturbations (all perturbations are in MAJOR now)
    print(
        f"\n--- Cross-Section Perturbations ({len(MAJOR_PERTURBATIONS)} channels) ---"
    )
    for pert_idx, (label, isotope, reaction) in enumerate(MAJOR_PERTURBATIONS):
        result = _run_one(label, isotope, reaction, pert_idx)
        if result:
            sensitivity_table.append(result)

    # Minor perturbations (empty list, kept for structural compatibility)
    n_major = len(MAJOR_PERTURBATIONS)
    for pert_idx, (label, isotope, reaction) in enumerate(MINOR_PERTURBATIONS):
        try:
            result = _run_one(label, isotope, reaction, n_major + pert_idx)
            if result:
                sensitivity_table.append(result)
        except Exception as exc:
            warnings.warn(f"Perturbation '{label}' SKIPPED: {exc}")
            print(f"  SKIP  '{label}': {exc}")

    # ---- Sandwich-rule systematic uncertainty ----
    sandwich = compute_sandwich_uncertainty(sensitivity_table)

    print(f"\n{'═'*60}")
    print("  SANDWICH-RULE SYSTEMATIC UNCERTAINTY")
    print(f"{'═'*60}")
    print(f"  {'Channel':<22s}  {'S_i':>10s}  {'ε_i':>8s}  {'ΔR_i (cm)':>12s}")
    print(f"  {'-'*56}")
    for row in sensitivity_table:
        p = row["parameter"]
        S_i = row["sensitivity"]
        eps_i = LITERATURE_UNCERTAINTIES.get(p, 0.0)
        dR_phys = sandwich["delta_R_phys"].get(p, 0.0)
        print(f"  {p:<22s}  {S_i:>+10.5f}  {eps_i:>8.3f}  {dR_phys:>+12.6f}")
    print(f"  {'-'*56}")
    print(f"  Cross-correlation (2ρ·ΔR_f·ΔR_c) : {sandwich['corr_term']:+.6f} cm²")
    print(f"  σ_syst (sandwich rule)             : {sandwich['sigma_syst']:.6f} cm")
    print(f"  Dominant channel                   : {sandwich['dominant']}")

    return R_baseline, sensitivity_table, sandwich


# =============================================================================
# PHASE 3 — REFLECTOR ALBEDO SCAN
# =============================================================================

# Approximate param ranges per albedo (for sphere geometry, scale for others)
# Higher albedo → more reflection → smaller critical radius
ALBEDO_PARAM_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "sphere": {
        "bare": (7.0, 9.5),
        "reflected": (3.0, 9.5),
    },
    "cube": {
        "bare": (12.0, 19.0),
        "reflected": (3.0, 16.0),
    },
    "cylinder": {
        "bare": (6.0, 10.0),
        "reflected": (1.0, 5.0),
    },
}


def _get_albedo_param_range(geometry_type: str, albedo: float) -> Tuple[float, float]:
    """
    Return scan range for a given albedo.
    albedo == 0.0 → bare range;  albedo > 0 → fixed reflected range.
    """
    ranges = ALBEDO_PARAM_RANGES.get(geometry_type, ALBEDO_PARAM_RANGES["sphere"])
    if albedo == 0.0:
        return ranges["bare"]
    return ranges["reflected"]


def phase3_albedo_scan(
    fuel: FastReactorFuel,
    baseline_ff: PreTabulatedFuel,
    geometry_type: str,
    albedo_step: float = 0.2,
    max_albedo: float = 0.9,
    seed_base: int = 0,
    n_replicas: int = 5,
) -> List[Dict]:
    """
    Scan R_crit as a function of reflector albedo, with full uncertainty
    characterization at each point:

      - Statistical: N replicas with different seeds → SE = s/√N
      - Systematic:  major perturbations with correlated seeds → σ_syst

    Nuclear data arrays are reused; only fuel.reflector_albedo changes.

    Parameters
    ----------
    fuel           : baseline FastReactorFuel (albedo=0)
    baseline_ff    : baseline PreTabulatedFuel (arrays to reuse)
    geometry_type  : "sphere", "cube", or "cylinder"
    albedo_step    : step between albedo values (default 0.2)
    max_albedo     : maximum albedo to scan (default 0.9)
    seed_base      : seed for reproducibility
    n_replicas     : number of replicas per albedo for statistical uncertainty

    Returns
    -------
    List of dicts per albedo with keys: albedo, R_crit, mass_kg, SE_R,
    s_R, replica_results, sensitivities, sandwich, syst_sandwich, sigma_tot
    """
    sep = "=" * 72
    print(f"\n{sep}")
    print("  PHASE 3 - Reflector Albedo Scan + Full Uncertainty (Sandwich Rule)")
    albedos = np.arange(0.0, max_albedo + albedo_step * 0.5, albedo_step)
    albedos = np.round(albedos, 4)
    print(f"  Albedo values   : {[f'{a:.2f}' for a in albedos]}")
    print(f"  Replicas/point  : {n_replicas} (statistical)")
    print(f"  Perturbations   : {len(MAJOR_PERTURBATIONS)} channels (sandwich rule)")
    print(sep)

    factor = 1.0 + PERTURBATION
    results: List[Dict] = []

    for albedo in albedos:
        print(f"\n{'═'*60}")
        print(f"  ALBEDO = {albedo:.2f}")
        print(f"{'═'*60}")

        t0 = time.time()

        # Create fuel with this albedo
        a_fuel = copy.copy(fuel)
        a_fuel.reflector_albedo = albedo

        # Reuse baseline arrays, swap fuel reference for albedo
        a_ff = copy.copy(baseline_ff)
        a_ff.fuel = a_fuel

        # Set scan range for this albedo
        old_range = PARAM_RANGES[geometry_type]
        new_range = _get_albedo_param_range(geometry_type, albedo)
        PARAM_RANGES[geometry_type] = new_range

        try:
            # ========================================================
            # STATISTICAL: N replicas with different seeds
            # ========================================================
            print(f"\n  --- Statistical ({n_replicas} replicas) ---")
            replica_R: List[float] = []
            for rep in range(n_replicas):
                rep_seed = rep * 100_000
                try:
                    R = scan_parallel(
                        a_ff,
                        geometry_type,
                        seed_base=rep_seed,
                        label=f"albedo={albedo:.2f} rep {rep+1}/{n_replicas}",
                    )
                    replica_R.append(R)
                    print(f"    replica {rep+1}: R_crit = {R:.5f} cm")
                except Exception as exc:
                    print(f"    replica {rep+1}: FAILED ({exc})")

            N_ok = len(replica_R)
            if N_ok > 0:
                R_bar = float(np.mean(replica_R))
                s_R = float(np.std(replica_R, ddof=1)) if N_ok > 1 else 0.0
                SE_R = s_R / np.sqrt(N_ok) if N_ok > 1 else 0.0
            else:
                R_bar, s_R, SE_R = float("nan"), 0.0, 0.0

            print(f"  R̄ = {R_bar:.5f} cm  ±  {SE_R:.5f} cm (SE, N={N_ok})")

            # ========================================================
            # SYSTEMATIC: major perturbations with correlated seeds
            # ========================================================
            print(f"\n  --- Systematic ({len(MAJOR_PERTURBATIONS)} perturbations) ---")
            # Use the FIRST replica seed as the baseline for sensitivity
            sens_seed = 0
            R_baseline_sens = replica_R[0] if replica_R else R_bar

            sens_list: List[Dict] = []
            for label, isotope, reaction in MAJOR_PERTURBATIONS:
                try:
                    pert_ff = perturb_fast_fuel(a_ff, isotope, reaction, factor)
                    R_pert = scan_parallel(
                        pert_ff,
                        geometry_type,
                        seed_base=sens_seed,  # SAME seed as first replica
                        label=f"albedo={albedo:.2f} {label}",
                    )
                    dR = R_pert - R_baseline_sens
                    S = dR / PERTURBATION
                    sens_list.append(
                        {
                            "parameter": label,
                            "R_perturbed": float(R_pert),
                            "delta_R": float(dR),
                            "sensitivity": float(S),
                        }
                    )
                    print(f"    {label:<20s}  dR={dR:+.5f} cm  S={S:+.5f}")
                except Exception as exc:
                    print(f"    {label:<20s}  FAILED: {exc}")

            if sens_list:
                sandwich_p3 = compute_sandwich_uncertainty(sens_list)
                syst_sandwich = sandwich_p3["sigma_syst"]
                # Also keep raw quadrature for comparison
                delta_Rs = [s["delta_R"] for s in sens_list]
                syst_quad = float(np.sqrt(sum(d**2 for d in delta_Rs)))
                syst_max = float(max(abs(d) for d in delta_Rs))
            else:
                sandwich_p3 = {
                    "delta_R_phys": {},
                    "sigma_syst": 0.0,
                    "corr_term": 0.0,
                    "dominant": "",
                }
                syst_sandwich, syst_quad, syst_max = 0.0, 0.0, 0.0

            # Combined total uncertainty
            sigma_tot = float(np.sqrt(s_R**2 + syst_sandwich**2))

            # ========================================================
            # Combine results
            # ========================================================
            if geometry_type == "sphere":
                vol = (4.0 / 3.0) * np.pi * R_bar**3
            elif geometry_type == "cube":
                vol = R_bar**3
            else:
                vol = np.pi * R_bar**2 * (CYLINDER_H_RATIO * R_bar)
            mass_kg = vol * fuel.total_density_g_cm3 / 1000.0

            elapsed = time.time() - t0
            results.append(
                {
                    "albedo": float(albedo),
                    "R_crit": float(R_bar),
                    "mass_kg": float(mass_kg),
                    "replica_results": replica_R,
                    "s_R": float(s_R),
                    "SE_R": float(SE_R),
                    "sensitivities": sens_list,
                    "sandwich": sandwich_p3,
                    "syst_sandwich": syst_sandwich,
                    "syst_quadrature": syst_quad,
                    "syst_max": syst_max,
                    "sigma_tot": sigma_tot,
                    "time_s": elapsed,
                }
            )

            print(
                f"\n  RESULT: R̄ = {R_bar:.4f} ± {s_R:.4f} (stat) ± {syst_sandwich:.4f} (syst) cm"
                f"  →  σ_tot = {sigma_tot:.4f} cm  |  mass = {mass_kg:.2f} kg  ({elapsed:.1f} s)"
            )

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  FAIL  albedo={albedo:.2f} ({elapsed:.1f} s): {exc}")
        finally:
            PARAM_RANGES[geometry_type] = old_range

    # --- Summary table ---
    print(f"\n{'═'*72}")
    print("  ALBEDO SCAN — FULL SUMMARY (sandwich rule)")
    print(f"{'═'*72}")
    print(
        f"  {'Albedo':>8}  {'R̄ (cm)':>10}  {'σ_stat':>8}  {'σ_syst':>8}  {'σ_tot':>8}  {'Mass (kg)':>10}"
    )
    print(f"  {'-'*58}")
    for r in results:
        print(
            f"  {r['albedo']:>8.2f}  {r['R_crit']:>10.4f}  {r['s_R']:>8.4f}"
            f"  {r['syst_sandwich']:>8.4f}  {r['sigma_tot']:>8.4f}  {r['mass_kg']:>10.2f}"
        )

    if len(results) >= 2:
        ratio = results[-1]["R_crit"] / results[0]["R_crit"]
        print(f"\n  R_crit ratio (max/bare): {ratio:.3f}")
        print(f"  Mass ratio  (max/bare): {(ratio**3):.3f}")

    # --- Plots ---
    _plot_albedo_scan(results, geometry_type)

    return results


def _plot_albedo_scan(results: List[Dict], geometry_type: str):
    """Create plots of R_crit, mass, and sensitivities vs albedo with error bars."""
    import matplotlib.pyplot as plt

    if len(results) < 2:
        return

    plot_dir = os.path.join("plots", "albedo_scan")
    os.makedirs(plot_dir, exist_ok=True)

    albedos = [r["albedo"] for r in results]
    R_crits = [r["R_crit"] for r in results]
    masses = [r["mass_kg"] for r in results]
    s_Rs = [r.get("s_R", 0) for r in results]
    syst_sandwiches = [r.get("syst_sandwich", 0) for r in results]

    # Total uncertainty: σ_tot = √(σ²_stat + σ²_syst)
    total_err = [
        r.get("sigma_tot", np.sqrt(s**2 + sy**2))
        for r, s, sy in zip(results, s_Rs, syst_sandwiches)
    ]

    has_sens = any(r.get("sensitivities") for r in results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ax1, ax2, ax3, ax4 = axes.flat

    # Panel 1: R_crit vs albedo with error bars
    ax1.errorbar(
        albedos,
        R_crits,
        yerr=total_err,
        fmt="o-",
        ms=10,
        lw=2.5,
        color="#FF6B6B",
        capsize=6,
        capthick=2,
        elinewidth=2,
        label="R̄ ± σ_total",
    )
    ax1.errorbar(
        albedos,
        R_crits,
        yerr=s_Rs,
        fmt="none",
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
        color="#2C3E50",
        label="± σ_stat",
    )
    ax1.set_xlabel("Reflector Albedo", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Critical Radius (cm)", fontsize=13, fontweight="bold")
    ax1.set_title(
        f"R_crit vs Albedo — {geometry_type.capitalize()}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3, ls="--")
    ax1.legend(fontsize=10)

    # Panel 2: Mass vs albedo
    ax2.plot(albedos, masses, "s-", ms=10, lw=2.5, color="#4ECDC4", label="Monte Carlo")
    ax2.set_xlabel("Reflector Albedo", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Critical Mass (kg)", fontsize=13, fontweight="bold")
    ax2.set_title(
        f"Critical Mass vs Albedo — {geometry_type.capitalize()}",
        fontsize=14,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3, ls="--")
    ax2.legend(fontsize=11)

    # Panel 3: Uncertainty components vs albedo
    ax3.plot(albedos, s_Rs, "o-", ms=8, lw=2, color="#3498DB", label="σ_stat (s)")
    ax3.plot(
        albedos,
        syst_sandwiches,
        "D-",
        ms=8,
        lw=2,
        color="#9B59B6",
        label="σ_syst (sandwich)",
    )
    ax3.plot(albedos, total_err, "^-", ms=8, lw=2, color="#E74C3C", label="σ_total")
    ax3.set_xlabel("Reflector Albedo", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Uncertainty (cm)", fontsize=13, fontweight="bold")
    ax3.set_title("Uncertainty Components vs Albedo", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3, ls="--")
    ax3.legend(fontsize=10)

    # Panel 4: Individual sensitivities vs albedo
    if has_sens:
        param_names = [s["parameter"] for s in results[0].get("sensitivities", [])]
        colors = [
            "#E74C3C",
            "#3498DB",
            "#2ECC71",
            "#F39C12",
            "#9B59B6",
            "#1ABC9C",
            "#E67E22",
            "#8E44AD",
            "#2C3E50",
            "#D35400",
            "#16A085",
            "#C0392B",
        ]
        for p_idx, pname in enumerate(param_names):
            S_vals = []
            for r in results:
                sens = r.get("sensitivities", [])
                match = [s for s in sens if s["parameter"] == pname]
                S_vals.append(match[0]["sensitivity"] if match else float("nan"))
            color = colors[p_idx % len(colors)]
            ax4.plot(albedos, S_vals, "o-", ms=7, lw=2, color=color, label=pname)
        ax4.axhline(0, color="black", ls="--", lw=1, alpha=0.4)
        ax4.legend(fontsize=9, loc="best")
    ax4.set_xlabel("Reflector Albedo", fontsize=13, fontweight="bold")
    ax4.set_ylabel("Sensitivity S (cm / unit)", fontsize=13, fontweight="bold")
    ax4.set_title("Per-Channel Sensitivity vs Albedo", fontsize=14, fontweight="bold")
    ax4.grid(True, alpha=0.3, ls="--")

    plt.tight_layout()
    path = os.path.join(plot_dir, f"{geometry_type}_albedo_scan.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {path}")


# =============================================================================
# REPORT WRITER
# =============================================================================


def save_report(
    path: str,
    geometry_type: str,
    replica_results: List[float],
    R_bar: float,
    s_R: float,
    SE_R: float,
    R_baseline: float,
    sensitivity_table: List[Dict],
    sandwich: Optional[Dict],
    albedo_results: Optional[List[Dict]],
    total_elapsed: float,
):
    sep = "=" * 72
    thin = "-" * 72

    lines: List[str] = [
        sep,
        "  CRITICALITY UNCERTAINTY ANALYSIS - RESULTS REPORT",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        sep,
        "",
        f"  Geometry         : {geometry_type.upper()}",
        f"  Fuel             : Fresh U-235 (93 wt% HEU)",
        f"  Neutrons/cycle   : {N_NEUTRONS}",
        f"  Inactive cycles  : {N_INACTIVE}",
        f"  Active cycles    : {N_ACTIVE}",
        f"  Replicas (P1)    : {N_REPLICAS}",
        f"  Scan points      : {N_SCAN_POINTS}",
        f"  Reflector albedo : {REFLECTOR_ALBEDO}  (baseline)",
        f"  Total wall time  : {total_elapsed/60:.1f} min",
        "",
    ]

    # ================================================================
    # Phase 1 — STATISTICAL
    # ================================================================
    N_ok = len(replica_results)
    lines += [sep, "  PHASE 1 - STATISTICAL UNCERTAINTY", sep, ""]
    if replica_results:
        lines.append(f"  {'Replica':>8}  {'Seed base':>10}  {'R_crit (cm)':>14}")
        lines.append(f"  {thin[:36]}")
        for i, R in enumerate(replica_results):
            lines.append(f"  {i+1:>8}  {i*100_000:>10}  {R:>14.6f}")
        lines += [
            "",
            f"  Mean R-bar           : {R_bar:.6f} cm",
            f"  Sample std dev  s    : {s_R:.6f} cm  (σ_stat — single-run spread)",
            f"  Std error  SE=s/√N   : {SE_R:.6f} cm  (uncertainty on R̄, N={N_ok})",
        ]
        if R_bar == R_bar:
            lines.append(f"  Rel. uncert. (s/R̄)   : {abs(s_R / R_bar) * 100:.4f} %")
    else:
        lines.append("  No replica results recorded.")
    lines.append("")

    # ================================================================
    # Phase 2 — SYSTEMATIC (Sandwich Rule)
    # ================================================================
    lines += [sep, "  PHASE 2 - SYSTEMATIC UNCERTAINTY (SANDWICH RULE)", sep, ""]
    lines += [
        f"  Perturbation δ       : {PERTURBATION*100:.1f} %",
        f"  R_baseline           : {R_baseline:.6f} cm",
        f"  Correlation ρ(f,c)   : {RHO_U235_FISS_CAPT}  (U-235 fission ↔ capture)",
        "",
    ]

    if sensitivity_table and sandwich:
        # Sensitivity + literature uncertainty table
        lines.append("  Sensitivity Table:")
        lines.append("")
        c = [22, 11, 8, 12, 12]
        header = (
            f"  {'Channel':<{c[0]}} "
            f"{'S_i':>{c[1]}} "
            f"{'ε_i':>{c[2]}} "
            f"{'ΔR_i (cm)':>{c[3]}} "
            f"{'ΔR_i² (cm²)':>{c[4]}}"
        )
        lines.append(header)
        lines.append(f"  {thin[:sum(c)+6]}")

        dr_phys = sandwich["delta_R_phys"]
        for row in sensitivity_table:
            p = row["parameter"]
            S_i = row["sensitivity"]
            eps_i = LITERATURE_UNCERTAINTIES.get(p, 0.0)
            dR_i = dr_phys.get(p, 0.0)
            lines.append(
                f"  {p:<{c[0]}} "
                f"{S_i:>{c[1]}+.5f} "
                f"{eps_i:>{c[2]}.3f} "
                f"{dR_i:>{c[3]}+.6f} "
                f"{dR_i**2:>{c[4]}.8f}"
            )

        lines.append(f"  {thin[:sum(c)+6]}")

        # Sandwich-rule calculation breakdown
        sum_sq = sum(dr**2 for dr in dr_phys.values())
        corr = sandwich["corr_term"]
        sigma_syst = sandwich["sigma_syst"]

        lines += [
            "",
            "  Sandwich-Rule Calculation:",
            "",
            f"    Σ(ΔR_i²)                       = {sum_sq:.8f} cm²",
            f"    2ρ · ΔR(U235_f) · ΔR(U235_c)   = {corr:+.8f} cm²",
            f"    ─────────────────────────────────────────────",
            f"    σ²_syst = Σ(ΔR_i²) + corr       = {sum_sq + corr:.8f} cm²",
            f"    σ_syst  = √(σ²_syst)             = {sigma_syst:.6f} cm",
            "",
            f"  Dominant channel : {sandwich['dominant']}",
        ]
    else:
        lines.append("  No perturbation results recorded.")
    lines.append("")

    # ================================================================
    # COMBINED UNCERTAINTY
    # ================================================================
    lines += [sep, "  COMBINED UNCERTAINTY BUDGET", sep, ""]
    if replica_results and sandwich:
        sigma_syst = sandwich["sigma_syst"]
        sigma_tot = float(np.sqrt(s_R**2 + sigma_syst**2))

        lines += [
            f"  σ_stat  (Phase 1, s)      : {s_R:.6f} cm",
            f"  σ_syst  (Phase 2, sandwich): {sigma_syst:.6f} cm",
            f"  ─────────────────────────────────────────────",
            f"  σ_total = √(σ²_stat + σ²_syst)",
            f"          = √({s_R:.6f}² + {sigma_syst:.6f}²)",
            f"          = {sigma_tot:.6f} cm",
            "",
            f"  R_crit  = {R_bar:.4f}  ±  {sigma_tot:.4f} cm  (total 1σ)",
        ]
        if R_bar == R_bar and R_bar != 0:
            lines.append(
                f"  Rel. total uncertainty : {abs(sigma_tot / R_bar) * 100:.4f} %"
            )
            lines.append("")
            # Budget fraction
            frac_stat = s_R**2 / (sigma_tot**2) * 100 if sigma_tot > 0 else 0
            frac_syst = sigma_syst**2 / (sigma_tot**2) * 100 if sigma_tot > 0 else 0
            lines += [
                "  Variance budget:",
                f"    Statistical : {frac_stat:5.1f}%  of total variance",
                f"    Systematic  : {frac_syst:5.1f}%  of total variance",
            ]
    else:
        lines.append("  Incomplete data — cannot compute combined uncertainty.")
    lines.append("")

    lines += [sep, "  END OF REPORT", sep, ""]

    # ================================================================
    # Phase 3 — Albedo scan (if present)
    # ================================================================
    if albedo_results:
        # Insert before END
        end_block = lines[-4:]
        lines = lines[:-4]

        lines += [sep, "  PHASE 3 - REFLECTOR ALBEDO SCAN + FULL UNCERTAINTY", sep, ""]
        lines.append(f"  Replicas/point  : {N_REPLICAS_P3}")
        lines.append(
            f"  Perturbation    : +{PERTURBATION*100:.1f}% on {len(MAJOR_PERTURBATIONS)} channels (sandwich rule)"
        )
        lines.append("")

        # Summary table
        c3 = [8, 10, 8, 8, 8, 10]
        hdr = (
            f"  {'Albedo':<{c3[0]}} "
            f"{'R̄ (cm)':>{c3[1]}} "
            f"{'σ_stat':>{c3[2]}} "
            f"{'σ_syst':>{c3[3]}} "
            f"{'σ_tot':>{c3[4]}} "
            f"{'Mass (kg)':>{c3[5]}}"
        )
        lines.append(hdr)
        lines.append(f"  {thin[:sum(c3)+7]}")
        for row in albedo_results:
            s_stat = row.get("s_R", 0)
            s_syst = row.get("syst_sandwich", 0)
            s_tot = row.get("sigma_tot", float(np.sqrt(s_stat**2 + s_syst**2)))
            lines.append(
                f"  {row['albedo']:<{c3[0]}.2f} "
                f"{row['R_crit']:>{c3[1]}.4f} "
                f"{s_stat:>{c3[2]}.4f} "
                f"{s_syst:>{c3[3]}.4f} "
                f"{s_tot:>{c3[4]}.4f} "
                f"{row['mass_kg']:>{c3[5]}.2f}"
            )
        lines.append("")

        # Detailed breakdown per albedo
        lines += [f"  {thin}", "  DETAILED BREAKDOWN BY ALBEDO", f"  {thin}", ""]
        for row in albedo_results:
            s_stat = row.get("s_R", 0)
            s_syst = row.get("syst_sandwich", 0)
            s_tot = row.get("sigma_tot", 0)
            lines.append(
                f"  Albedo = {row['albedo']:.2f}  |  R̄ = {row['R_crit']:.4f} "
                f"± {s_tot:.4f} cm (total)"
            )

            # Replicas
            reps = row.get("replica_results", [])
            if reps:
                for j, r in enumerate(reps):
                    lines.append(f"    replica {j+1}: {r:.5f} cm")
                lines.append(
                    f"    → σ_stat = {s_stat:.5f} cm  " f"(s, N = {len(reps)})"
                )

            # Sensitivities with sandwich
            sens = row.get("sensitivities", [])
            p3_sandwich = row.get("sandwich", {})
            p3_dr_phys = p3_sandwich.get("delta_R_phys", {}) if p3_sandwich else {}
            if sens:
                for s in sens:
                    p = s["parameter"]
                    eps_i = LITERATURE_UNCERTAINTIES.get(p, 0.0)
                    dR_phys = p3_dr_phys.get(p, 0.0)
                    lines.append(
                        f"    {p:<22s}  S={s['sensitivity']:+.5f}  "
                        f"ε={eps_i:.3f}  ΔR={dR_phys:+.6f}"
                    )
                lines.append(f"    → σ_syst(sandwich) = {s_syst:.5f} cm")
                lines.append(
                    f"    → σ_tot = √({s_stat:.5f}² + {s_syst:.5f}²) = {s_tot:.5f} cm"
                )
            lines.append("")

        lines += end_block

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n{'='*72}")
    print(f"  Report saved -> {os.path.abspath(path)}")
    print(f"{'='*72}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================


def main():
    from multiprocessing import freeze_support

    freeze_support()

    t_start = time.time()

    n_perturbations = len(MAJOR_PERTURBATIONS) + len(MINOR_PERTURBATIONS)

    print("=" * 72)
    print("  OPTIMIZED CRITICALITY UNCERTAINTY ANALYSIS PIPELINE  [PARALLEL]")
    print(f"  Geometry      : {GEOMETRY_TYPE.upper()}")
    print(
        f"  MC params     : {N_NEUTRONS} neutrons | {N_INACTIVE} inactive | {N_ACTIVE} active"
    )
    print(
        f"  Phase 1       : {N_REPLICAS} replicas x {N_SCAN_POINTS} pts"
        + ("" if RUN_PHASE1 else "  [SKIPPED]")
    )
    print(
        f"  Phase 2       : {len(MAJOR_PERTURBATIONS)} perturbations + sandwich rule"
        + ("" if RUN_PHASE2 else "  [SKIPPED]")
    )
    n_albedo_pts = len(np.arange(0.0, MAX_ALBEDO + ALBEDO_STEP * 0.5, ALBEDO_STEP))
    print(
        f"  Phase 3       : albedo scan 0.0–{MAX_ALBEDO:.1f} step {ALBEDO_STEP:.1f} ({n_albedo_pts} points)"
        + ("" if RUN_PHASE3 else "  [SKIPPED]")
    )
    print(f"  Workers/scan  : {N_WORKERS} (of {cpu_count()} available cores)")
    print(f"  Started       : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("=" * 72)

    if GEOMETRY_TYPE not in PARAM_RANGES:
        raise ValueError(
            f"GEOMETRY_TYPE must be one of {list(PARAM_RANGES.keys())}, "
            f"got {GEOMETRY_TYPE!r}."
        )

    # =================================================================
    # ONE-TIME TABULATION  (the key optimization)
    # =================================================================
    print("\nLoading nuclear data and pre-tabulating (one-time cost)...")
    t_tab = time.time()
    fuel = get_fresh_u235_fuel(reflector_albedo=REFLECTOR_ALBEDO)
    baseline_ff = PreTabulatedFuel(fuel)
    t_tab = time.time() - t_tab
    print(f"OK  Tabulation complete in {t_tab:.1f}s")
    print(
        f"    (Old pipeline would have tabulated ~{(N_REPLICAS + 1 + n_perturbations) * N_WORKERS} times)"
    )

    # =================================================================
    # PHASES
    # =================================================================
    replica_results: List[float] = []
    R_bar, s_R, SE_R = float("nan"), 0.0, 0.0
    R_baseline = float("nan")
    sensitivity_table: List[Dict] = []
    sandwich: Optional[Dict] = None
    albedo_results: List[Dict] = []

    if RUN_PHASE1:
        replica_results, R_bar, s_R, SE_R = phase1_statistical(baseline_ff)
    else:
        print("\n  Phase 1 SKIPPED (RUN_PHASE1 = False)")

    if RUN_PHASE2:
        R_baseline, sensitivity_table, sandwich = phase2_systematic(fuel, baseline_ff)
    else:
        print("  Phase 2 SKIPPED (RUN_PHASE2 = False)")

    if RUN_PHASE3:
        albedo_results = phase3_albedo_scan(
            fuel,
            baseline_ff,
            GEOMETRY_TYPE,
            albedo_step=ALBEDO_STEP,
            max_albedo=MAX_ALBEDO,
            seed_base=BASELINE_SEED,
            n_replicas=N_REPLICAS_P3,
        )
    else:
        print("  Phase 3 SKIPPED (RUN_PHASE3 = False)")

    total_elapsed = time.time() - t_start

    report_name = f"uncertainty_results_{GEOMETRY_TYPE}.txt"
    save_report(
        report_name,
        geometry_type=GEOMETRY_TYPE,
        replica_results=replica_results,
        R_bar=R_bar,
        s_R=s_R,
        SE_R=SE_R,
        R_baseline=R_baseline,
        sensitivity_table=sensitivity_table,
        sandwich=sandwich,
        albedo_results=albedo_results,
        total_elapsed=total_elapsed,
    )

    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    print("FINAL SUMMARY")
    print(f"  Geometry        : {GEOMETRY_TYPE.upper()}")
    print(
        f"  Phases run      : {' + '.join(p for p, r in [('P1', RUN_PHASE1), ('P2', RUN_PHASE2), ('P3', RUN_PHASE3)] if r)}"
    )
    if RUN_PHASE1:
        print(
            f"  σ_stat (s)      : {s_R:.5f} cm  (sample std dev, N={len(replica_results)})"
        )
    if RUN_PHASE2 and sandwich:
        sigma_syst = sandwich["sigma_syst"]
        print(
            f"  σ_syst (sandwich): {sigma_syst:.5f} cm  ({sandwich['dominant']} dominant)"
        )
    if RUN_PHASE1 and RUN_PHASE2 and sandwich:
        sigma_syst = sandwich["sigma_syst"]
        sigma_tot = float(np.sqrt(s_R**2 + sigma_syst**2))
        print(f"  ─────────────────────────────────")
        print(f"  σ_total          : {sigma_tot:.5f} cm")
        print(f"  R_crit           : {R_bar:.4f}  ±  {sigma_tot:.4f} cm")
        if R_bar == R_bar and R_bar != 0:
            print(f"  Rel. uncertainty : {abs(sigma_tot / R_bar) * 100:.3f} %")
    if RUN_PHASE3:
        print(f"  Albedo points   : {len(albedo_results)}")
    print(f"  Total wall time : {total_elapsed/60:.1f} min")
    print(f"\n  -> {report_name}")


if __name__ == "__main__":
    main()
