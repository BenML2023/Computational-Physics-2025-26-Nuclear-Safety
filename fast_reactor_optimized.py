"""
PROPERLY FIXED Fast Reactor Optimized
Follows original logic:
1. Store ONE fission site per fission
2. Energy-dependent ν̄(E) per fission event from ENDF/HDF5 tabulated data
3. Handle ALL reactions (elastic, inelastic, capture, fission)
4. Includes U-234 and U-236 isotope support
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import sys
import os

sys.path.append(os.path.dirname(__file__))
from cs_getter import NuclearData


# =============================================================================
# WATT SPECTRUM
# =============================================================================

WATT_PARAMS = {
    "U234": {"a": 0.988, "b": 2.249},  # MeV (approx same as U-235)
    "U235": {"a": 0.988, "b": 2.249},
    "U236": {"a": 0.884, "b": 3.300},  # MeV (approx same as U-238)
    "U238": {"a": 0.880, "b": 3.400},
    "Pu239": {"a": 0.966, "b": 2.383},
    "Pu240": {"a": 0.960, "b": 2.400},
}


def sample_watt_spectrum(a: float, b: float, rng) -> float:
    """
    Robust Watt sampling via Maxwell transform
    Returns energy in eV
    """

    # 1. Maxwellian sample (temperature = a)
    # Using relation: W = a * chi-square(3)
    W = a * rng.chisquare(df=3) / 2.0

    # 2. Transformation
    xi = rng.random()
    term = np.sqrt(a * a * b * W)

    E_MeV = W + (a * a * b) / 4.0 + (2.0 * xi - 1.0) * term
    return max(E_MeV, 0.0) * 1e6


# =============================================================================
# NU UNCERTAINTY — only used for error propagation on k_eff
# The actual ν̄(E) values come from the HDF5 files via NuclearData.get_nu()
# =============================================================================

NU_SIGMA = {
    "U234": 0.0100,
    "U235": 0.0023,
    "U236": 0.0100,
    "U238": 0.0100,
    "Pu239": 0.0065,
    "Pu240": 0.0100,
}


# =============================================================================
# FIX 2: TOTAL CROSS SECTION (compute from partials)
# =============================================================================


class CachedNuclearData:
    """Nuclear data with LRU caching and auto-computed total XS"""

    def __init__(self, data_dir: str):
        from cs_getter import NuclearData

        self.db = NuclearData(data_dir=data_dir)
        self.cache_hits = 0
        self.cache_misses = 0

    @lru_cache(maxsize=20000)
    def _get_xs_cached(self, isotope: str, reaction: str, energy_key: str) -> float:
        """Cached with string key"""
        energy_eV = float(energy_key)

        # Special handling for 'total' which may not be in ENDF
        if reaction == "total":
            return self._compute_total_xs(isotope, energy_eV)

        return self.db.get_xs(isotope, reaction, energy_eV)

    def _compute_total_xs(self, isotope: str, energy_eV: float) -> float:
        """
        Compute total XS from partial reactions
        σ_total = σ_elastic + σ_fission + σ_capture + σ_inelastic + ...
        """

        # First try direct lookup (some ENDF files have MT=1)
        try:
            return self.db.get_xs(isotope, "total", energy_eV)
        except:
            pass

        # Compute from partials
        xs_sum = 0.0

        # Main reactions
        for reaction in ["elastic", "fission", "capture"]:
            try:
                xs_sum += self.db.get_xs(isotope, reaction, energy_eV)
            except:
                pass

        # Inelastic (MT=51-91)
        # Try a few common inelastic levels
        for mt in range(51, 92):
            try:
                xs_sum += self.db.get_xs(isotope, f"inelastic_MT{mt}", energy_eV)
            except:
                pass

        # Return sum or small value if all failed
        return xs_sum if xs_sum > 0 else 1e-10

    def get_xs(self, isotope: str, reaction: str, energy_eV: float) -> float:
        """Get cross section with caching"""

        # Round energy to bins
        if energy_eV < 1e4:
            energy_rounded = round(energy_eV, 0)
        elif energy_eV < 1e5:
            energy_rounded = round(energy_eV, -1)
        elif energy_eV < 1e6:
            energy_rounded = round(energy_eV, -2)
        else:
            energy_rounded = round(energy_eV, -3)

        energy_key = f"{energy_rounded:.0f}"

        try:
            result = self._get_xs_cached(isotope, reaction, energy_key)
            self.cache_hits += 1
            return result
        except Exception as e:
            self.cache_misses += 1
            # For total, try computing from partials
            if reaction == "total":
                return self._compute_total_xs(isotope, energy_eV)
            # Otherwise reraise
            return self.db.get_xs(isotope, reaction, energy_eV)

    def get_cache_stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        info = self._get_xs_cached.cache_info()
        return {
            "hit_rate": hit_rate,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "cache_size": info.currsize,
        }

    def _load(self, isotope: str):
        return self.db._load(isotope)

    def sample_scattering(self, *args, **kwargs):
        return self.db.sample_scattering(*args, **kwargs)

    def get_scattered_direction(self, *args, **kwargs):
        return self.db.get_scattered_direction(*args, **kwargs)

    def get_inelastic_scattered_energy(self, *args, **kwargs):
        return self.db.get_inelastic_scattered_energy(*args, **kwargs)

    def get_nu(self, isotope: str, energy_eV: float) -> float:
        """Return tabulated ν̄(E) from HDF5 data."""
        return self.db.get_nu(isotope, energy_eV)


# =============================================================================
# GEOMETRY
# =============================================================================


class Geometry:
    def sample_initial(self, n: int, rng) -> np.ndarray:
        raise NotImplementedError

    def distance_to_boundary(self, x: np.ndarray, v: np.ndarray) -> float:
        raise NotImplementedError

    def get_volume(self) -> float:
        raise NotImplementedError

    def get_params_dict(self) -> dict:
        raise NotImplementedError


class Sphere(Geometry):
    def __init__(self, radius: float):
        self.radius = radius

    def sample_initial(self, n: int, rng) -> np.ndarray:
        positions = []
        for _ in range(n):
            while True:
                x = rng.uniform(-self.radius, self.radius, 3)
                if np.linalg.norm(x) < self.radius:
                    positions.append(x)
                    break
        return np.array(positions)

    def distance_to_boundary(self, x: np.ndarray, v: np.ndarray) -> float:
        a = np.dot(v, v)
        b = 2 * np.dot(x, v)
        c = np.dot(x, x) - self.radius**2
        disc = b**2 - 4 * a * c
        if disc < 0:
            return 1e10
        d1 = (-b + np.sqrt(disc)) / (2 * a)
        d2 = (-b - np.sqrt(disc)) / (2 * a)
        return max(0, max(d1, d2) if d1 > 0 or d2 > 0 else 1e10)

    def get_volume(self) -> float:
        return (4 / 3) * np.pi * self.radius**3

    def get_params_dict(self) -> dict:
        return {"radius": self.radius}

    def __str__(self):
        return f"Sphere(R={self.radius:.2f} cm)"


class Cube(Geometry):
    def __init__(self, side: float):
        self.side = side

    def sample_initial(self, n: int, rng) -> np.ndarray:
        half = self.side / 2
        return rng.uniform(-half, half, (n, 3))

    def distance_to_boundary(self, x: np.ndarray, v: np.ndarray) -> float:
        half = self.side / 2
        distances = []
        for i in range(3):
            if abs(v[i]) > 1e-10:
                d_pos = (half - x[i]) / v[i]
                d_neg = (-half - x[i]) / v[i]
                if d_pos > 0:
                    distances.append(d_pos)
                if d_neg > 0:
                    distances.append(d_neg)
        return min(distances) if distances else 1e10

    def get_volume(self) -> float:
        return self.side**3

    def get_params_dict(self) -> dict:
        return {"side": self.side}

    def __str__(self):
        return f"Cube(side={self.side:.2f} cm)"


class Cylinder(Geometry):
    def __init__(self, radius: float, height: float):
        self.radius = radius
        self.height = height

    def sample_initial(self, n: int, rng) -> np.ndarray:
        positions = []
        for _ in range(n):
            r = self.radius * np.sqrt(rng.random())
            theta = rng.uniform(0, 2 * np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = rng.uniform(-self.height / 2, self.height / 2)
            positions.append([x, y, z])
        return np.array(positions)

    def distance_to_boundary(self, x: np.ndarray, v: np.ndarray) -> float:
        a = v[0] ** 2 + v[1] ** 2
        b = 2 * (x[0] * v[0] + x[1] * v[1])
        c = x[0] ** 2 + x[1] ** 2 - self.radius**2

        distances = []
        if a > 1e-10:
            disc = b**2 - 4 * a * c
            if disc >= 0:
                d1 = (-b + np.sqrt(disc)) / (2 * a)
                d2 = (-b - np.sqrt(disc)) / (2 * a)
                if d1 > 0:
                    distances.append(d1)
                if d2 > 0:
                    distances.append(d2)

        if abs(v[2]) > 1e-10:
            d_top = (self.height / 2 - x[2]) / v[2]
            d_bot = (-self.height / 2 - x[2]) / v[2]
            if d_top > 0:
                distances.append(d_top)
            if d_bot > 0:
                distances.append(d_bot)

        return min(distances) if distances else 1e10

    def get_volume(self) -> float:
        return np.pi * self.radius**2 * self.height

    def get_params_dict(self) -> dict:
        return {"radius": self.radius, "height": self.height}

    def __str__(self):
        return f"Cylinder(R={self.radius:.2f} cm, H={self.height:.2f} cm)"


def isotropic_direction(rng) -> np.ndarray:
    mu = rng.uniform(-1, 1)
    phi = rng.uniform(0, 2 * np.pi)
    sin_theta = np.sqrt(1 - mu**2)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), mu])


def compute_shannon_entropy(
    positions: np.ndarray, characteristic_length: float
) -> float:
    bins_per_dim = 4  # 4×4×4 = 64 bins total, sensible for 1000 neutrons

    hist, _ = np.histogramdd(positions, bins=bins_per_dim)
    hist = hist.flatten()
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    return -np.sum(probs * np.log(probs))


# =============================================================================
# FUEL
# =============================================================================


@dataclass
class IsotopeComponent:
    name: str
    atom_fraction: float
    density_g_cm3: float


class FastReactorFuel:

    def __init__(
        self,
        isotopes: List[IsotopeComponent],
        reflector_albedo: float = 0.90,
        use_watt_spectrum: bool = True,
        include_nu_uncertainty: bool = True,
        data_dir: str = "/Users/benweihrauch/Desktop/Nuclear Safety/Code/master/NuclearData",
    ):
        self.isotopes = isotopes
        self.reflector_albedo = reflector_albedo
        self.use_watt_spectrum = use_watt_spectrum
        self.include_nu_uncertainty = include_nu_uncertainty

        print("Loading nuclear data...")
        self.nuclear_data = CachedNuclearData(data_dir=data_dir)

        for iso in isotopes:
            print(f"[{iso.name}] Loading from HDF5...")
            self.nuclear_data._load(iso.name)
            print(f"[{iso.name}] Ready.")
        print("✓ All isotopes loaded\n")

        self.total_density_g_cm3 = sum(iso.density_g_cm3 for iso in isotopes)

        self.atom_densities = {}
        for iso in isotopes:
            A = {
                "U234": 234,
                "U235": 235,
                "U236": 236,
                "U238": 238,
                "Pu239": 239,
                "Pu240": 240,
            }[iso.name]
            N_avogadro = 6.022e23
            self.atom_densities[iso.name] = (iso.density_g_cm3 * N_avogadro) / A

        self.total_atom_density = sum(self.atom_densities.values())

        # Calculate average nu with uncertainty
        self.nu = self._calc_average_nu()
        self.nu_sigma = self._calc_nu_uncertainty()

    def _calc_average_nu(self) -> float:
        """Calculate mixture-averaged nu at a reference energy (2 MeV) for display"""
        E_ref = 2.0e6
        return self.get_nu_at_energy(E_ref)

    def _calc_nu_uncertainty(self) -> float:
        """Calculate propagated nu uncertainty"""
        if not self.include_nu_uncertainty:
            return 0.0
        var = sum(
            (iso.atom_fraction * NU_SIGMA.get(iso.name, 0.01)) ** 2
            for iso in self.isotopes
        )
        return np.sqrt(var) / sum(iso.atom_fraction for iso in self.isotopes)

    def get_nu_at_energy(self, energy_eV: float) -> float:
        """
        Mixture-averaged ν̄ at a given energy, weighted by atom fraction.
        Uses tabulated ENDF data from HDF5 files.
        Only includes isotopes that have a fission channel.
        """
        nu_sum = 0.0
        frac_sum = 0.0
        for iso in self.isotopes:
            try:
                nu_i = self.nuclear_data.get_nu(iso.name, energy_eV)
                if nu_i > 0:
                    nu_sum += iso.atom_fraction * nu_i
                    frac_sum += iso.atom_fraction
            except (ValueError, KeyError):
                pass  # isotope has no fission channel
        return nu_sum / frac_sum if frac_sum > 0 else 2.45

    def get_nu_for_fission(self, isotope: str, energy_eV: float) -> float:
        """
        Energy-dependent ν̄ for a specific fission event.
        Reads directly from the ENDF tabulated data in the HDF5 file.
        This is what is used in the transport loop.
        """
        return self.nuclear_data.get_nu(isotope, energy_eV)

    def sample_fission_energy(self, rng, isotope: Optional[str] = None) -> float:
        """Sample fission neutron energy"""
        if not self.use_watt_spectrum:
            return abs(rng.normal(2.0e6, 0.7e6))

        if isotope is None:
            isotope = self.isotopes[0].name

        if isotope in WATT_PARAMS:
            params = WATT_PARAMS[isotope]
            return sample_watt_spectrum(params["a"], params["b"], rng)
        return abs(rng.normal(2.0e6, 0.7e6))

    def get_macroscopic_xs(self, energy_eV: float) -> Dict[str, float]:
        """Get macroscopic cross sections (cm⁻¹)"""
        xs = {
            "total": 0.0,
            "elastic": 0.0,
            "inelastic": 0.0,
            "fission": 0.0,
            "capture": 0.0,
        }

        for iso in self.isotopes:
            N = self.atom_densities[iso.name]

            for reaction in ["elastic", "inelastic", "fission", "capture"]:
                try:
                    sigma = self.nuclear_data.get_xs(iso.name, reaction, energy_eV)
                    macro = (
                        N * sigma * 1e-24
                    )  # convert barn → cm² and multiply by atom density
                    xs[reaction] += macro
                    xs["total"] += macro
                except:
                    pass

        return xs

    def mean_free_path(self, energy_eV: float) -> float:
        """Mean free path (cm)"""
        xs = self.get_macroscopic_xs(energy_eV)
        return 1.0 / xs["total"] if xs["total"] > 0 else 1e10

    def sample_isotope(self, energy_eV: float, rng) -> str:
        """Sample collision isotope using summed microscopic XS"""

        weights = {}
        total = 0.0

        for iso in self.isotopes:
            N = self.atom_densities[iso.name]
            sigma_sum = 0.0

            # Build total from available channels
            for reaction in ["elastic", "inelastic", "fission", "capture"]:
                try:
                    sigma = self.nuclear_data.get_xs(iso.name, reaction, energy_eV)
                    sigma_sum += sigma
                except:
                    pass

            weight = N * sigma_sum
            weights[iso.name] = weight
            total += weight

        if total <= 0.0:
            return self.isotopes[0].name

        xi = rng.random() * total
        cumsum = 0.0

        for name, weight in weights.items():
            cumsum += weight
            if xi < cumsum:
                return name

        return self.isotopes[0].name

    def sample_reaction(self, isotope: str, energy_eV: float, rng) -> str:
        """Sample reaction using summed microscopic XS (no MT=1)"""

        xs = {}
        xs_total = 0.0

        for reaction in ["elastic", "inelastic", "fission", "capture"]:
            try:
                sigma = self.nuclear_data.get_xs(isotope, reaction, energy_eV)
            except:
                sigma = 0.0
            xs[reaction] = sigma
            xs_total += sigma

        if xs_total <= 0.0:
            return "capture"

        xi = rng.random() * xs_total
        cumsum = 0.0

        # Order matters only for numerical stability, not physics
        for reaction in ["fission", "capture", "elastic", "inelastic"]:
            cumsum += xs[reaction]
            if xi < cumsum:
                return reaction

        return "capture"

    """
    def sample_reaction(self, isotope: str, energy_eV: float, rng) -> str:
        #Sample reaction type
        try:
            xs_tot = self.nuclear_data.get_xs(isotope, 'total', energy_eV)
            if xs_tot == 0:
                return 'capture'
            
            xs_fiss = self.nuclear_data.get_xs(isotope, 'fission', energy_eV)
            xs_capt = self.nuclear_data.get_xs(isotope, 'capture', energy_eV)
            xs_elas = self.nuclear_data.get_xs(isotope, 'elastic', energy_eV)
            
            P_fiss = xs_fiss / xs_tot
            P_capt = xs_capt / xs_tot
            P_elas = xs_elas / xs_tot
            
            xi = rng.random()
            
            if xi < P_fiss:
                return 'fission'
            elif xi < P_fiss + P_capt:
                return 'capture'
            elif xi < P_fiss + P_capt + P_elas:
                return 'elastic'
            return 'inelastic'
        except:
            return 'capture'
    """

    def scatter_neutron(
        self, isotope: str, reaction: str, E_in: float, v_in: np.ndarray, rng
    ) -> Tuple[float, np.ndarray]:
        """Scatter neutron (handles elastic AND inelastic)"""
        if reaction not in ["elastic", "inelastic"]:
            return E_in, v_in

        mt = 2 if reaction == "elastic" else 51

        if reaction == "inelastic":
            try:
                levels = self.nuclear_data.db.get_inelastic_scattered_energy(
                    isotope, E_in
                )
                if levels:
                    total_xs = sum(lv["xs_barn"] for lv in levels)
                    if total_xs > 0:
                        xi = rng.random() * total_xs
                        cumsum = 0.0
                        for lv in levels:
                            cumsum += lv["xs_barn"]
                            if xi < cumsum:
                                mt = lv["mt"]
                                break
            except:
                mt = 2

        try:
            mu_lab, E_out, phi = self.nuclear_data.sample_scattering(
                isotope, mt, E_in, rng
            )
            v_out = self.nuclear_data.get_scattered_direction(mu_lab, phi, v_in)
            return E_out, v_out
        except:
            return E_in * 0.98, isotropic_direction(rng)


# =============================================================================
# CORRECTED TRANSPORT - Following original logic
# =============================================================================


def run_fast_reactor_keff(
    geometry: Geometry,
    fuel: FastReactorFuel,
    n_neutrons: int = 1000,
    n_inactive: int = 10,
    n_active: int = 20,
    seed: int = 42,
    verbose: bool = True,
    track_isotope_stats: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    CORRECTED k_eff calculation

    Key fixes:
    1. Store ONE fission site per fission (not nu sites)
    2. Energy-dependent ν: each fission contributes ν(isotope, E_incident)
       k = Σ ν_i / n_start  (summed over all fission events)
    3. Handle ALL reactions: fission, capture, elastic, inelastic
    """

    rng = np.random.default_rng(seed)

    # Initialize
    positions = geometry.sample_initial(n_neutrons, rng)
    energies = np.array([fuel.sample_fission_energy(rng) for _ in range(n_neutrons)])
    directions = np.array([isotropic_direction(rng) for _ in range(n_neutrons)])

    keff_samples = []
    entropy_samples = []

    n_reflections = 0
    n_boundary = 0
    isotope_fissions = {} if track_isotope_stats else None

    n_cycles = n_inactive + n_active

    if verbose:
        print(f"Running {n_cycles} cycles ({n_inactive} inactive + {n_active} active)")
        print(
            f"  {n_neutrons} neutrons/cycle | Watt: {fuel.use_watt_spectrum} | "
            f"ν(2 MeV ref): {fuel.nu:.4f} ± {fuel.nu_sigma:.4f} [energy-dependent]"
        )

    params = geometry.get_params_dict()
    char_length = params.get("radius", params.get("side", 10.0))

    for cycle in range(n_cycles):
        n_start = len(positions)

        entropy = compute_shannon_entropy(positions, char_length)
        entropy_samples.append(entropy)

        # CRITICAL: Store ONE fission site per fission
        fission_sites = []
        fission_energies = []
        fission_directions = []
        nu_sum = 0.0  # Accumulate energy-dependent ν per fission

        for i in range(n_start):
            x = positions[i].copy()
            E = energies[i]
            v = directions[i].copy()

            for _ in range(100):  # Max collisions
                mfp = fuel.mean_free_path(E)
                d_coll = rng.exponential(mfp)
                d_bound = geometry.distance_to_boundary(x, v)

                if d_bound < d_coll:
                    # Boundary
                    n_boundary += 1
                    if rng.random() < fuel.reflector_albedo:
                        # Reflect
                        n_reflections += 1
                        x = (
                            x + (d_bound - 0.01) * v
                        )  # move to just before boundary (OLD direction)
                        v = isotropic_direction(rng)  # THEN pick new direction
                        continue
                    else:
                        # Escape
                        break

                # Collision
                x = x + d_coll * v

                isotope = fuel.sample_isotope(E, rng)
                reaction = fuel.sample_reaction(isotope, E, rng)

                if reaction == "capture":
                    # Absorbed
                    break

                elif reaction == "fission":
                    # STORE ONE FISSION SITE (nu accumulated per-event)
                    fission_sites.append(x.copy())
                    fission_energies.append(fuel.sample_fission_energy(rng, isotope))
                    fission_directions.append(isotropic_direction(rng))
                    nu_sum += fuel.get_nu_for_fission(isotope, E)

                    if track_isotope_stats:
                        isotope_fissions[isotope] = isotope_fissions.get(isotope, 0) + 1
                    break

                else:
                    # Scatter (elastic OR inelastic)
                    E, v = fuel.scatter_neutron(isotope, reaction, E, v, rng)

                    # if E < 1e3:  # Below 1 keV
                    #  break

        n_fissions = len(fission_sites)

        # CRITICAL: k_eff uses per-fission energy-dependent ν
        k_eff = nu_sum / n_start if n_start > 0 else 0.0

        if cycle >= n_inactive:
            keff_samples.append(k_eff)

        if verbose and (cycle % 10 == 0 or cycle < 3):
            status = "inactive" if cycle < n_inactive else "active"
            print(
                f"  Cycle {cycle+1:3d}/{n_cycles} ({status:8s}): k={k_eff:.4f}, fissions={n_fissions}, n={n_start}"
            )

        if n_fissions == 0:
            if verbose:
                print(f"  Extinct at cycle {cycle+1}")
            break

        # Sample next generation
        indices = rng.choice(
            len(fission_sites), size=n_neutrons, replace=len(fission_sites) < n_neutrons
        )

        positions = np.array(fission_sites)[indices]
        energies = np.array(fission_energies)[indices]
        directions = np.array(fission_directions)[indices]

    # Stats
    reflection_rate = n_reflections / n_boundary if n_boundary > 0 else 0.0
    cache_stats = fuel.nuclear_data.get_cache_stats()

    stats = {
        "reflection_rate": reflection_rate,
        "cache_hit_rate": cache_stats["hit_rate"],
        "cache_hits": cache_stats["hits"],
    }

    if track_isotope_stats:
        stats["isotope_fissions"] = isotope_fissions

    # Add nu uncertainty to total error
    if fuel.include_nu_uncertainty and len(keff_samples) > 0:
        keff_mean = np.mean(keff_samples) if len(keff_samples) > 0 else 0.0
        if len(keff_samples) > 1:
            sigma_stat = np.std(keff_samples, ddof=1)
        else:
            sigma_stat = 0.0
        sigma_nu = keff_mean * (fuel.nu_sigma / fuel.nu)
        sigma_total = np.sqrt(sigma_stat**2 + sigma_nu**2)

        stats["sigma_statistical"] = sigma_stat
        stats["sigma_nu"] = sigma_nu
        stats["sigma_total"] = sigma_total

    if verbose:
        if len(keff_samples) > 0:
            print(
                f"\nk_eff: {np.mean(keff_samples):.4f} ± {np.std(keff_samples):.4f} | Cache: {cache_stats['hit_rate']:.1%}"
            )
            if "sigma_total" in stats:
                print(
                    f"  σ_stat={stats['sigma_statistical']:.4f}, σ_ν={stats['sigma_nu']:.4f}, σ_total={stats['sigma_total']:.4f}"
                )
        else:
            print(f"\nNo k_eff samples (extinction)")

    return np.array(keff_samples), np.array(entropy_samples), stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_fresh_u235_fuel(reflector_albedo: float = 0.90) -> FastReactorFuel:
    """
    Fresh HEU fuel matching the ORSphere / Godiva benchmark (HEU-MET-FAST-001).

    Composition (weight fractions):
      U-234:  1.02 wt%
      U-235: 93.71 wt%
      U-236:  0.00 wt%  (placeholder — set > 0 if your benchmark includes it)
      U-238:  5.27 wt%

    Density: 18.75 g/cm³
    """
    A_234 = 234.041
    A_235 = 235.044
    A_236 = 236.046
    A_238 = 238.051
    rho_total = 18.75  # g/cm³  (ORSphere benchmark value)

    # Weight fractions — Godiva / ORSphere benchmark
    w_234 = 0.0098
    w_235 = 0.932
    w_236 = 0.0004
    w_238 = 0.0576

    # Normalise in case fractions don't sum to 1.0 exactly
    w_sum = w_234 + w_235 + w_236 + w_238
    w_234 /= w_sum
    w_235 /= w_sum
    w_236 /= w_sum
    w_238 /= w_sum

    # Convert to atom fractions:  x_i = (w_i / A_i) / Σ(w_j / A_j)
    inv_A = (
        w_234 / A_234 + w_235 / A_235 + w_236 / A_236 + w_238 / A_238
        if w_236 > 0
        else w_234 / A_234 + w_235 / A_235 + w_238 / A_238
    )

    x_234 = (w_234 / A_234) / inv_A
    x_235 = (w_235 / A_235) / inv_A
    x_236 = (w_236 / A_236) / inv_A if w_236 > 0 else 0.0
    x_238 = (w_238 / A_238) / inv_A

    # Component densities from weight fractions
    rho_234 = w_234 * rho_total
    rho_235 = w_235 * rho_total
    rho_236 = w_236 * rho_total
    rho_238 = w_238 * rho_total

    isotopes = [
        IsotopeComponent("U234", atom_fraction=x_234, density_g_cm3=rho_234),
        IsotopeComponent("U235", atom_fraction=x_235, density_g_cm3=rho_235),
    ]

    # Only include U-236 if present in the composition
    if w_236 > 0:
        isotopes.append(
            IsotopeComponent("U236", atom_fraction=x_236, density_g_cm3=rho_236)
        )

    isotopes.append(
        IsotopeComponent("U238", atom_fraction=x_238, density_g_cm3=rho_238)
    )

    return FastReactorFuel(isotopes, reflector_albedo=reflector_albedo)


def get_waste_fuel(reflector_albedo: float = 0.90) -> FastReactorFuel:
    """
    Nuclear waste fuel for breeder reactor based on 60 GWd/t IAEA discharge data.

    Composition (Mass fractions):
    - U-238: 96.879% (Fertile matrix)
    - Pu-239: 1.037% (Lumped fissile transmutant)
    - U-235: 0.867% (Residual fissile)
    - U-236: 0.737% (Parasitic capture product)
    - Pu-240: 0.458% (Lumped fertile transmutant)
    - U-234: 0.022% (Residual natural enrichment)
    """

    # 1. Physical constants (Atomic masses)
    A_U234 = 234.041
    A_U235 = 235.044
    A_U236 = 236.046
    A_U238 = 238.051
    A_Pu239 = 239.052
    A_Pu240 = 240.054

    # Pure material densities (g/cm³)
    rho_U = 19.05
    rho_Pu = 19.84

    # 2. Base Mass Fractions (w_i) from IAEA 60 GWd/t data
    w_U234 = 0.00022
    w_U235 = 0.00867
    w_U236 = 0.00737
    w_U238 = 0.96879
    w_Pu239 = 0.01037
    w_Pu240 = 0.00458

    # Normalize to ensure floating-point math sums to exactly 1.0
    w_sum = w_U234 + w_U235 + w_U236 + w_U238 + w_Pu239 + w_Pu240
    w_U234 /= w_sum
    w_U235 /= w_sum
    w_U236 /= w_sum
    w_U238 /= w_sum
    w_Pu239 /= w_sum
    w_Pu240 /= w_sum

    # 3. Calculate Total Alloy Density
    # Dynamically weighted by the exact Uranium vs Plutonium mass split
    w_U_total = w_U234 + w_U235 + w_U236 + w_U238
    w_Pu_total = w_Pu239 + w_Pu240
    rho_total = (w_U_total * rho_U) + (w_Pu_total * rho_Pu)

    # 4. Convert Mass Fractions (w_i) to Atom Fractions (x_i)
    # Formula: x_i = (w_i / A_i) / Σ(w_j / A_j)
    inv_A_sum = (
        (w_U234 / A_U234)
        + (w_U235 / A_U235)
        + (w_U236 / A_U236)
        + (w_U238 / A_U238)
        + (w_Pu239 / A_Pu239)
        + (w_Pu240 / A_Pu240)
    )

    x_U234 = (w_U234 / A_U234) / inv_A_sum
    x_U235 = (w_U235 / A_U235) / inv_A_sum
    x_U236 = (w_U236 / A_U236) / inv_A_sum
    x_U238 = (w_U238 / A_U238) / inv_A_sum
    x_Pu239 = (w_Pu239 / A_Pu239) / inv_A_sum
    x_Pu240 = (w_Pu240 / A_Pu240) / inv_A_sum

    # 5. Calculate Component Densities (g/cm³)
    # The partial density of an isotope is simply its mass fraction * total density
    rho_U234_partial = w_U234 * rho_total
    rho_U235_partial = w_U235 * rho_total
    rho_U236_partial = w_U236 * rho_total
    rho_U238_partial = w_U238 * rho_total
    rho_Pu239_partial = w_Pu239 * rho_total
    rho_Pu240_partial = w_Pu240 * rho_total

    # 6. Build and return the Fuel object
    isotopes = [
        IsotopeComponent("U234", atom_fraction=x_U234, density_g_cm3=rho_U234_partial),
        IsotopeComponent("U235", atom_fraction=x_U235, density_g_cm3=rho_U235_partial),
        IsotopeComponent("U236", atom_fraction=x_U236, density_g_cm3=rho_U236_partial),
        IsotopeComponent("U238", atom_fraction=x_U238, density_g_cm3=rho_U238_partial),
        IsotopeComponent(
            "Pu239", atom_fraction=x_Pu239, density_g_cm3=rho_Pu239_partial
        ),
        IsotopeComponent(
            "Pu240", atom_fraction=x_Pu240, density_g_cm3=rho_Pu240_partial
        ),
    ]

    return FastReactorFuel(isotopes, reflector_albedo=reflector_albedo)


if __name__ == "__main__":
    print("=" * 70)
    print("CORRECTED FAST REACTOR OPTIMIZED")
    print("=" * 70)

    fuel = get_fresh_u235_fuel(reflector_albedo=0.90)
    sphere = Sphere(radius=22.0)

    print(f"\nTest: {sphere} with reflector")
    print("Expected: k_eff ≈ 1.15-1.25\n")

    keff, entropy, stats = run_fast_reactor_keff(
        sphere,
        fuel,
        n_neutrons=1000,
        n_inactive=10,
        n_active=20,
        verbose=True,
        track_isotope_stats=True,
    )

    if len(keff) > 0:
        print(f"\n✓ SUCCESS! Mean k_eff = {np.mean(keff):.4f} ± {np.std(keff):.4f}")
        print(f"✓ Cache: {stats['cache_hit_rate']:.1%}")

        if "isotope_fissions" in stats:
            print(f"\nFission contributions:")
            total = sum(stats["isotope_fissions"].values())
            for iso, count in sorted(
                stats["isotope_fissions"].items(), key=lambda x: x[1], reverse=True
            ):
                pct = count / total * 100 if total > 0 else 0
                print(f"  {iso}: {count} ({pct:.1f}%)")
    else:
        print(f"\n✗ Neutrons extinct")
