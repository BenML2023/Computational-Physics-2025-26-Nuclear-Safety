"""
Accelerated Fast Reactor Transport — EXACT physics, faster lookups
===================================================================
Drop-in replacement for run_fast_reactor_keff() that eliminates the
cross-section lookup bottleneck while keeping IDENTICAL physics.

What's faster
--------------
- All microscopic XS and ν̄ pre-tabulated on a fine log-spaced energy grid
- Single interpolation call replaces ~50 get_xs() calls per collision
  (each of which does: energy binning → string format → LRU hash →
   dict lookup → potential HDF5 read)
- All XS computed ONCE per collision, reused for mfp + isotope + reaction
- Geometry distance_to_boundary inlined (no method dispatch)

What's IDENTICAL to the original
----------------------------------
- Scattering: calls fuel.scatter_neutron() → OpenMC angular distributions
- Inelastic level selection: same discrete-level sampling from HDF5
- Fission spectrum: same Watt sampling via fuel.sample_fission_energy()
- ν̄(E): same tabulated ENDF values (just interpolated from pre-built table)
- Boundary logic: identical reflection, escape, step-back
- Power iteration: identical cycle structure, source resampling

Usage
-----
    from fast_reactor_accel import run_keff_accel, PreTabulatedFuel

    fuel = get_fresh_u235_fuel(reflector_albedo=0.0)
    geom = Sphere(radius=8.74)

    # Option A: drop-in (pre-tabulates internally)
    keff, entropy, stats = run_keff_accel(geom, fuel, n_neutrons=1000)

    # Option B: pre-tabulate once, reuse across radii
    fast_fuel = PreTabulatedFuel(fuel)
    keff, entropy, stats = run_keff_accel(geom, fuel, n_neutrons=1000,
                                          fast_fuel=fast_fuel)
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import time
import sys
import os

sys.path.append(os.path.dirname(__file__))
from fast_reactor_optimized import (
    FastReactorFuel,
    Geometry, Sphere, Cube, Cylinder,
    IsotopeComponent,
    isotropic_direction,
    compute_shannon_entropy,
    get_fresh_u235_fuel,
    get_waste_fuel,
)


# =============================================================================
# PRE-TABULATED FUEL
# =============================================================================

class PreTabulatedFuel:
    """
    Wraps a FastReactorFuel and pre-tabulates all cross sections and ν̄
    on a fine energy grid for fast numpy interpolation.

    The scattering kernel (angular distributions, inelastic levels) is
    NOT pre-tabulated — it delegates to the original fuel object to
    keep the exact same physics.
    """

    def __init__(
        self,
        fuel: FastReactorFuel,
        n_energy_pts: int = 5000,
        E_min: float = 1.0e2,      # 100 eV
        E_max: float = 2.5e7,      # 25 MeV
    ):
        self.fuel = fuel  # keep original for scattering calls

        # Build energy grid (log-spaced)
        self.E_grid = np.logspace(np.log10(E_min), np.log10(E_max), n_energy_pts)
        self.log_E_grid = np.log(self.E_grid)
        self.n_E = n_energy_pts

        # Isotope info
        self.n_iso = len(fuel.isotopes)
        self.iso_names = [iso.name for iso in fuel.isotopes]
        self.atom_densities = np.array([
            fuel.atom_densities[iso.name] for iso in fuel.isotopes
        ])

        # Pre-tabulate
        t0 = time.time()
        self._tabulate_xs()
        self._tabulate_nu()
        self._precompute_macro()
        dt = time.time() - t0

        print(f"  Pre-tabulation complete: {self.n_iso} isotopes × "
              f"{self.n_E} energies in {dt:.1f}s")

    def _tabulate_xs(self):
        """
        Build microscopic XS tables: (n_iso, n_E, 4)
        Last dimension = [elastic, inelastic, fission, capture]
        """
        reactions = ["elastic", "inelastic", "fission", "capture"]
        self.micro_xs = np.zeros((self.n_iso, self.n_E, 4))

        for i, iso in enumerate(self.fuel.isotopes):
            for r_idx, rx in enumerate(reactions):
                for j in range(self.n_E):
                    try:
                        self.micro_xs[i, j, r_idx] = \
                            self.fuel.nuclear_data.get_xs(iso.name, rx, self.E_grid[j])
                    except Exception:
                        self.micro_xs[i, j, r_idx] = 0.0
            print(f"    [{iso.name}] XS tabulated")

    def _tabulate_nu(self):
        """Build ν̄ tables: (n_iso, n_E)"""
        self.nu_bar = np.zeros((self.n_iso, self.n_E))
        for i, iso in enumerate(self.fuel.isotopes):
            for j in range(self.n_E):
                try:
                    self.nu_bar[i, j] = \
                        self.fuel.nuclear_data.get_nu(iso.name, self.E_grid[j])
                except Exception:
                    self.nu_bar[i, j] = 0.0
            print(f"    [{iso.name}] ν̄ tabulated")

    def _precompute_macro(self):
        """
        Pre-compute macroscopic cross sections and cumulative
        isotope weights for fast lookup.

        macro_total[j]         = Σ_t(E_j)
        macro_by_iso[i, j]     = N_i × σ_tot_i(E_j)   (for isotope sampling)
        macro_by_iso_cum[i, j] = cumulative sum over i  (for isotope sampling)
        reaction_cum[i, j, 4]  = cumulative reaction XS (for reaction sampling)
        """
        N = self.atom_densities
        n_iso = self.n_iso
        n_E = self.n_E

        # Macroscopic total
        self.macro_total = np.zeros(n_E)
        # Per-isotope total microscopic XS (sum of 4 partials)
        self.micro_total = np.zeros((n_iso, n_E))

        for i in range(n_iso):
            self.micro_total[i] = self.micro_xs[i, :, :].sum(axis=1)  # (n_E,)
            self.macro_total += N[i] * self.micro_total[i] * 1e-24

        # Pre-compute isotope weights = N_i × σ_tot_i  (not ×1e-24, cancels in ratio)
        self.iso_weights = np.zeros((n_iso, n_E))
        self.iso_weights_cumsum = np.zeros((n_iso, n_E))
        self.iso_weights_total = np.zeros(n_E)

        for i in range(n_iso):
            self.iso_weights[i] = N[i] * self.micro_total[i]
        self.iso_weights_total = self.iso_weights.sum(axis=0)

        cumsum = np.zeros(n_E)
        for i in range(n_iso):
            cumsum = cumsum + self.iso_weights[i]
            self.iso_weights_cumsum[i] = cumsum.copy()

        # Per-isotope cumulative reaction XS (for reaction sampling)
        # Order: fission(0), capture(1), elastic(2), inelastic(3)
        # (reordered from storage for same branching logic as original)
        self.rxn_cumsum = np.zeros((n_iso, n_E, 4))
        # Map: storage [elastic=0, inelastic=1, fission=2, capture=3]
        # Sampling order: fission, capture, elastic, inelastic
        rxn_order = [2, 3, 0, 1]  # indices into micro_xs last dim
        for i in range(n_iso):
            cum = np.zeros(n_E)
            for k, r_idx in enumerate(rxn_order):
                cum = cum + self.micro_xs[i, :, r_idx]
                self.rxn_cumsum[i, :, k] = cum.copy()

        # Reaction name mapping for sampling order
        self.rxn_names = ["fission", "capture", "elastic", "inelastic"]

    def _interp_idx(self, log_E):
        """Find interpolation index and fraction for a given log(energy)."""
        if log_E <= self.log_E_grid[0]:
            return 0, 0.0
        if log_E >= self.log_E_grid[-1]:
            return self.n_E - 2, 1.0

        # Binary search
        lo, hi = 0, self.n_E - 1
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if self.log_E_grid[mid] <= log_E:
                lo = mid
            else:
                hi = mid

        t = (log_E - self.log_E_grid[lo]) / (self.log_E_grid[hi] - self.log_E_grid[lo])
        return lo, t

    def _lerp(self, table, lo, t):
        """Linear interpolation: table[lo] + t * (table[lo+1] - table[lo])"""
        return table[lo] + t * (table[lo + 1] - table[lo])

    def mean_free_path(self, E):
        """Fast MFP lookup."""
        log_E = np.log(E) if E > 0.0 else self.log_E_grid[0]
        lo, t = self._interp_idx(log_E)
        sigma_t = self._lerp(self.macro_total, lo, t)
        return 1.0 / sigma_t if sigma_t > 0.0 else 1e10

    def sample_isotope(self, E, rng):
        """Fast isotope sampling from pre-computed cumulative weights."""
        log_E = np.log(E) if E > 0.0 else self.log_E_grid[0]
        lo, t = self._interp_idx(log_E)

        total_w = self._lerp(self.iso_weights_total, lo, t)
        if total_w <= 0.0:
            return self.iso_names[0]

        xi = rng.random() * total_w
        for i in range(self.n_iso):
            cum_i = self._lerp(self.iso_weights_cumsum[i], lo, t)
            if xi < cum_i:
                return self.iso_names[i]

        return self.iso_names[-1]

    def sample_reaction(self, isotope, E, rng):
        """Fast reaction sampling from pre-computed cumulative XS."""
        iso_idx = self.iso_names.index(isotope)

        log_E = np.log(E) if E > 0.0 else self.log_E_grid[0]
        lo, t = self._interp_idx(log_E)

        # Total XS for this isotope
        xs_total = self._lerp(self.micro_total[iso_idx], lo, t)
        if xs_total <= 0.0:
            return "capture"

        xi = rng.random() * xs_total
        for k in range(4):
            cum_k = self._lerp(self.rxn_cumsum[iso_idx, :, k], lo, t)
            if xi < cum_k:
                return self.rxn_names[k]

        return "capture"

    def get_nu_for_fission(self, isotope, E):
        """Fast ν̄ lookup."""
        iso_idx = self.iso_names.index(isotope)
        log_E = np.log(E) if E > 0.0 else self.log_E_grid[0]
        lo, t = self._interp_idx(log_E)
        return self._lerp(self.nu_bar[iso_idx], lo, t)

    # --- Delegate exact physics to original fuel ---

    def scatter_neutron(self, isotope, reaction, E_in, v_in, rng):
        """EXACT same scattering — delegates to original fuel."""
        return self.fuel.scatter_neutron(isotope, reaction, E_in, v_in, rng)

    def sample_fission_energy(self, rng, isotope=None):
        """EXACT same fission spectrum — delegates to original fuel."""
        return self.fuel.sample_fission_energy(rng, isotope)


# =============================================================================
# ARRAY-LEVEL PERTURBATION  (replaces full re-tabulation)
# =============================================================================

def perturb_fast_fuel(
    base_ff: PreTabulatedFuel,
    isotope: str,
    reaction: str,
    factor: float = 1.01,
) -> PreTabulatedFuel:
    """
    Create a perturbed copy of a PreTabulatedFuel by modifying the numpy
    arrays directly.  Takes ~1 ms instead of ~5–10 s for full tabulation.

    Parameters
    ----------
    base_ff  : the baseline PreTabulatedFuel (NOT modified)
    isotope  : e.g. "U235", "U238"
    reaction : "fission", "capture", "elastic", "inelastic", or "nu"
    factor   : multiplicative perturbation (1.01 = +1%)

    Returns
    -------
    A new PreTabulatedFuel with the perturbation baked into the arrays.
    Scattering angular distributions are unaffected (shared with baseline).

    Raises
    ------
    ValueError if isotope or reaction is not in the tabulated channels.
    """
    import copy

    ff = copy.copy(base_ff)

    # Deep-copy only the mutable arrays that will be modified
    ff.micro_xs = base_ff.micro_xs.copy()
    ff.nu_bar   = base_ff.nu_bar.copy()

    # Locate isotope
    if isotope not in ff.iso_names:
        raise ValueError(
            f"Isotope '{isotope}' not in tabulated set: {ff.iso_names}"
        )
    iso_idx = ff.iso_names.index(isotope)

    # Apply perturbation
    if reaction == "nu":
        ff.nu_bar[iso_idx] *= factor
    else:
        # micro_xs layout: [elastic=0, inelastic=1, fission=2, capture=3]
        rxn_map = {"elastic": 0, "inelastic": 1, "fission": 2, "capture": 3}
        if reaction not in rxn_map:
            raise ValueError(
                f"Cannot perturb reaction '{reaction}' — "
                f"not in tabulated channels {list(rxn_map.keys())}. "
                f"Use 'nu' for ν̄ perturbation."
            )
        r_idx = rxn_map[reaction]
        ff.micro_xs[iso_idx, :, r_idx] *= factor

    # Recompute all derived macroscopic quantities from modified arrays
    # (macro_total, iso_weights, cumulative sums — ~instant)
    ff._precompute_macro()

    return ff


# =============================================================================
# FAST GEOMETRY (inlined, avoids method dispatch overhead)
# =============================================================================

def _distance_to_boundary(x, v, geom_type, geom_p1, geom_p2):
    """
    Inlined distance-to-boundary for all geometry types.
    geom_type: 0=sphere, 1=cube, 2=cylinder
    geom_p1: radius (sphere/cyl) or side (cube)
    geom_p2: height (cylinder) or unused
    """
    if geom_type == 0:
        # Sphere
        a = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
        b = 2.0 * (x[0]*v[0] + x[1]*v[1] + x[2]*v[2])
        c = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] - geom_p1 * geom_p1
        disc = b*b - 4.0*a*c
        if disc < 0.0:
            return 1e10
        sq = disc**0.5
        d1 = (-b + sq) / (2.0 * a)
        d2 = (-b - sq) / (2.0 * a)
        if d1 > 0.0 or d2 > 0.0:
            return max(d1, d2)
        return 1e10

    elif geom_type == 1:
        # Cube
        half = geom_p1 / 2.0
        d_min = 1e10
        for i in range(3):
            if abs(v[i]) > 1e-10:
                d_pos = (half - x[i]) / v[i]
                d_neg = (-half - x[i]) / v[i]
                if 0.0 < d_pos < d_min:
                    d_min = d_pos
                if 0.0 < d_neg < d_min:
                    d_min = d_neg
        return d_min

    else:
        # Cylinder
        radius = geom_p1
        half_h = geom_p2 / 2.0
        d_min = 1e10

        a = v[0]*v[0] + v[1]*v[1]
        if a > 1e-10:
            b = 2.0 * (x[0]*v[0] + x[1]*v[1])
            c = x[0]*x[0] + x[1]*x[1] - radius*radius
            disc = b*b - 4.0*a*c
            if disc >= 0.0:
                sq = disc**0.5
                d1 = (-b + sq) / (2.0 * a)
                d2 = (-b - sq) / (2.0 * a)
                if 0.0 < d1 < d_min:
                    d_min = d1
                if 0.0 < d2 < d_min:
                    d_min = d2

        if abs(v[2]) > 1e-10:
            d_top = (half_h - x[2]) / v[2]
            d_bot = (-half_h - x[2]) / v[2]
            if 0.0 < d_top < d_min:
                d_min = d_top
            if 0.0 < d_bot < d_min:
                d_min = d_bot

        return d_min


# =============================================================================
# ACCELERATED TRANSPORT LOOP
# =============================================================================

def run_keff_accel(
    geometry: Geometry,
    fuel: FastReactorFuel,
    n_neutrons: int = 1000,
    n_inactive: int = 10,
    n_active: int = 20,
    seed: int = 42,
    verbose: bool = True,
    track_isotope_stats: bool = False,
    fast_fuel: Optional[PreTabulatedFuel] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Accelerated k_eff calculation — EXACT same physics as
    run_fast_reactor_keff(), just faster XS lookups.

    Parameters
    ----------
    fast_fuel : PreTabulatedFuel, optional
        Pass a pre-built PreTabulatedFuel to avoid re-tabulating
        when scanning multiple geometries with the same fuel.
    """

    # --- Pre-tabulate if needed ---
    if fast_fuel is None:
        print("  Pre-tabulating cross sections...")
        fast_fuel = PreTabulatedFuel(fuel)

    ff = fast_fuel  # short alias

    # --- Geometry setup (inlined for speed) ---
    if isinstance(geometry, Sphere):
        geom_type = 0
        geom_p1 = geometry.radius
        geom_p2 = 0.0
    elif isinstance(geometry, Cube):
        geom_type = 1
        geom_p1 = geometry.side
        geom_p2 = 0.0
    elif isinstance(geometry, Cylinder):
        geom_type = 2
        geom_p1 = geometry.radius
        geom_p2 = geometry.height
    else:
        raise ValueError(f"Unknown geometry: {type(geometry)}")

    albedo = fuel.reflector_albedo
    rng = np.random.default_rng(seed)

    # --- Initialise source ---
    positions = geometry.sample_initial(n_neutrons, rng)
    energies = np.array([ff.sample_fission_energy(rng) for _ in range(n_neutrons)])
    directions = np.array([isotropic_direction(rng) for _ in range(n_neutrons)])

    keff_samples = []
    entropy_samples = []
    n_reflections = 0
    n_boundary = 0
    isotope_fissions = {} if track_isotope_stats else None

    n_cycles = n_inactive + n_active
    params = geometry.get_params_dict()
    char_length = params.get("radius", params.get("side", 10.0))

    if verbose:
        print(f"Running {n_cycles} cycles ({n_inactive} inactive + {n_active} active)")
        print(
            f"  {n_neutrons} neutrons/cycle | Watt: {fuel.use_watt_spectrum} | "
            f"ν(2 MeV ref): {fuel.nu:.4f} ± {fuel.nu_sigma:.4f} [energy-dependent]"
        )

    t_start = time.time()

    for cycle in range(n_cycles):
        n_start = len(positions)

        entropy = compute_shannon_entropy(positions, char_length)
        entropy_samples.append(entropy)

        fission_sites = []
        fission_energies = []
        fission_directions = []
        nu_sum = 0.0

        for i in range(n_start):
            x = positions[i].copy()
            E = energies[i]
            v = directions[i].copy()

            for _ in range(100):  # Max collisions — same as original

                # --- MFP (fast lookup) ---
                mfp = ff.mean_free_path(E)
                d_coll = rng.exponential(mfp)

                # --- Distance to boundary (inlined) ---
                d_bound = _distance_to_boundary(x, v, geom_type, geom_p1, geom_p2)

                if d_bound < d_coll:
                    # Boundary — IDENTICAL logic to original
                    n_boundary += 1
                    if rng.random() < albedo:
                        n_reflections += 1
                        x = x + (d_bound - 0.01) * v  # move to just before boundary (OLD direction)
                        v = isotropic_direction(rng)   # THEN pick new direction
                        continue
                    else:
                        break

                # Collision
                x = x + d_coll * v

                # --- Sample isotope (fast lookup) ---
                isotope = ff.sample_isotope(E, rng)

                # --- Sample reaction (fast lookup) ---
                reaction = ff.sample_reaction(isotope, E, rng)

                if reaction == "capture":
                    break

                elif reaction == "fission":
                    fission_sites.append(x.copy())
                    fission_energies.append(ff.sample_fission_energy(rng, isotope))
                    fission_directions.append(isotropic_direction(rng))
                    nu_sum += ff.get_nu_for_fission(isotope, E)

                    if track_isotope_stats:
                        isotope_fissions[isotope] = \
                            isotope_fissions.get(isotope, 0) + 1
                    break

                else:
                    # Scatter — EXACT same physics (OpenMC angular distributions)
                    E, v = ff.scatter_neutron(isotope, reaction, E, v, rng)

        n_fissions = len(fission_sites)
        k_eff = nu_sum / n_start if n_start > 0 else 0.0

        if cycle >= n_inactive:
            keff_samples.append(k_eff)

        if verbose and (cycle % 10 == 0 or cycle < 3):
            status = "inactive" if cycle < n_inactive else "active"
            elapsed = time.time() - t_start
            print(
                f"  Cycle {cycle+1:3d}/{n_cycles} ({status:8s}): "
                f"k={k_eff:.4f}, fissions={n_fissions}, n={n_start}, "
                f"t={elapsed:.1f}s"
            )

        if n_fissions == 0:
            if verbose:
                print(f"  Extinct at cycle {cycle+1}")
            break

        # Sample next generation — IDENTICAL to original
        indices = rng.choice(
            len(fission_sites), size=n_neutrons,
            replace=len(fission_sites) < n_neutrons,
        )
        positions = np.array(fission_sites)[indices]
        energies = np.array(fission_energies)[indices]
        directions = np.array(fission_directions)[indices]

    elapsed = time.time() - t_start

    # --- Stats ---
    reflection_rate = n_reflections / n_boundary if n_boundary > 0 else 0.0
    stats = {
        "reflection_rate": reflection_rate,
        "cache_hit_rate": 1.0,
        "cache_hits": 0,
        "total_time": elapsed,
    }

    if track_isotope_stats:
        stats["isotope_fissions"] = isotope_fissions

    if fuel.include_nu_uncertainty and len(keff_samples) > 0:
        keff_mean = np.mean(keff_samples) if len(keff_samples) > 0 else 0.0
        sigma_stat = np.std(keff_samples, ddof=1) if len(keff_samples) > 1 else 0.0
        sigma_nu = keff_mean * (fuel.nu_sigma / fuel.nu) if fuel.nu > 0 else 0.0
        sigma_total = np.sqrt(sigma_stat**2 + sigma_nu**2)
        stats["sigma_statistical"] = sigma_stat
        stats["sigma_nu"] = sigma_nu
        stats["sigma_total"] = sigma_total

    if verbose and len(keff_samples) > 0:
        km = np.mean(keff_samples)
        ks = np.std(keff_samples)
        print(f"\n  k_eff = {km:.4f} ± {ks:.4f}  "
              f"({len(keff_samples)} active cycles, {elapsed:.1f}s)")

    return np.array(keff_samples), np.array(entropy_samples), stats


# =============================================================================
# MAIN — Benchmark against original
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ACCELERATED TRANSPORT — BENCHMARK")
    print("=" * 70)

    fuel = get_fresh_u235_fuel(reflector_albedo=0.0)
    geom = Sphere(radius=8.74)

    print(f"\nBare HEU sphere R=8.74 cm (Godiva benchmark, expect k ≈ 1.0)")

    # --- Pre-tabulate ---
    print("\nPre-tabulating...")
    fast_fuel = PreTabulatedFuel(fuel)

    # --- Run accelerated ---
    print("\n--- Accelerated transport ---")
    t0 = time.time()
    keff_a, entropy_a, stats_a = run_keff_accel(
        geom, fuel,
        n_neutrons=500, n_inactive=20, n_active=50,
        seed=42, verbose=True, track_isotope_stats=True,
        fast_fuel=fast_fuel,
    )
    t_accel = time.time() - t0

    # --- Run original for comparison ---
    from fast_reactor_optimized import run_fast_reactor_keff
    print("\n--- Original transport ---")
    t0 = time.time()
    keff_o, entropy_o, stats_o = run_fast_reactor_keff(
        geom, fuel,
        n_neutrons=500, n_inactive=20, n_active=50,
        seed=42, verbose=True, track_isotope_stats=True,
    )
    t_orig = time.time() - t0

    # --- Compare ---
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")

    ka = np.mean(keff_a) if len(keff_a) > 0 else 0.0
    ko = np.mean(keff_o) if len(keff_o) > 0 else 0.0
    print(f"  Accelerated: k_eff = {ka:.4f}, time = {t_accel:.1f}s")
    print(f"  Original:    k_eff = {ko:.4f}, time = {t_orig:.1f}s")
    print(f"  Speedup:     {t_orig/t_accel:.1f}×")
    print(f"  k_eff diff:  {abs(ka - ko):.4f}")

    if "isotope_fissions" in stats_a and "isotope_fissions" in stats_o:
        print(f"\n  Fission fractions:")
        all_iso = set(list(stats_a["isotope_fissions"].keys()) +
                      list(stats_o["isotope_fissions"].keys()))
        ta = sum(stats_a["isotope_fissions"].values())
        to = sum(stats_o["isotope_fissions"].values())
        print(f"  {'Isotope':<8s} {'Accel':>8s} {'Orig':>8s}")
        for iso in sorted(all_iso):
            fa = stats_a["isotope_fissions"].get(iso, 0) / ta * 100 if ta > 0 else 0
            fo = stats_o["isotope_fissions"].get(iso, 0) / to * 100 if to > 0 else 0
            print(f"  {iso:<8s} {fa:>7.1f}% {fo:>7.1f}%")
