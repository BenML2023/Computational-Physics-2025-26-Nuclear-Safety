import openmc.data
import numpy as np
import os


# =============================================================================
# Picklable ν̄(E) wrappers  (multiprocessing requires pickle-safe callables)
# =============================================================================


class _ZeroNuBar:
    """Returns 0.0 — for isotopes without a fission channel."""

    def __call__(self, energy_eV):
        return 0.0

    def __repr__(self):
        return "_ZeroNuBar()"


class _LinearNuBar:
    """ν̄(E) = a + b × E(MeV), returns 0 below threshold."""

    def __init__(self, a, b, E_threshold_eV=0.0):
        self.a = a
        self.b = b
        self.E_threshold_eV = E_threshold_eV

    def __call__(self, energy_eV):
        if energy_eV < self.E_threshold_eV:
            return 0.0
        return self.a + self.b * (energy_eV / 1.0e6)

    def __repr__(self):
        return (
            f"_LinearNuBar(a={self.a}, b={self.b}, "
            f"thr={self.E_threshold_eV/1e6:.2f} MeV)"
        )


class _TabulatedNuBar:
    """Linear interpolation over tabulated (energy, ν̄) pairs."""

    def __init__(self, energies, values):
        self._energies = np.asarray(energies, dtype=np.float64)
        self._values = np.asarray(values, dtype=np.float64)

    def __call__(self, energy_eV):
        return float(np.interp(energy_eV, self._energies, self._values))

    def __repr__(self):
        return f"_TabulatedNuBar({len(self._energies)} pts)"


class _PolynomialNuBar:
    """ν̄(E) = Σ cᵢ × E_eV^i   (coefficients straight from ENDF Polynomial)."""

    def __init__(self, coefficients, E_threshold_eV=0.0):
        self._coef = list(coefficients)
        self.E_threshold_eV = E_threshold_eV

    def __call__(self, energy_eV):
        if energy_eV < self.E_threshold_eV:
            return 0.0
        result = 0.0
        e_pow = 1.0
        for c in self._coef:
            result += c * e_pow
            e_pow *= energy_eV
        return result

    def __repr__(self):
        return f"_PolynomialNuBar(coef={self._coef})"


class NuclearData:
    """
    Wrapper around OpenMC HDF5 nuclear data files.

    IMPORTANT: All random sampling uses the caller-supplied numpy RNG
    (``rng``) so that results are fully reproducible with a fixed seed.
    OpenMC's built-in ``.sample()`` methods use their own internal
    (unseeded) RNG and must NOT be called.
    """

    @staticmethod
    def _sample_tabular_seeded(tab, rng):
        """
        Inverse-CDF sample from an OpenMC Tabular / Uniform / Discrete
        angular distribution using the *seeded* numpy ``rng``.

        This replaces ``tab.sample(1)[0]`` which uses OpenMC's internal
        (unseeded) RNG and breaks reproducibility.

        Parameters
        ----------
        tab  : openmc.data.Tabular, Uniform, Discrete, or Isotropic
        rng  : numpy Generator (seeded)

        Returns
        -------
        float : sampled value (cos θ for angular distributions)
        """
        # --- Isotropic: uniform on [-1, 1] ---
        type_name = type(tab).__name__
        if type_name == "Isotropic":
            return rng.uniform(-1.0, 1.0)

        # --- Uniform: uniform on [a, b] ---
        if type_name == "Uniform":
            a = tab.a if hasattr(tab, "a") else -1.0
            b = tab.b if hasattr(tab, "b") else 1.0
            return rng.uniform(a, b)

        # --- Discrete: weighted choice ---
        if type_name == "Discrete":
            x = np.asarray(tab.x)
            p = np.asarray(tab.p)
            p = p / p.sum()
            return rng.choice(x, p=p)

        # --- Tabular: inverse CDF interpolation ---
        # tab.x = variable grid (cos θ),  tab.c = CDF at those points
        x = np.asarray(tab.x)
        cdf = np.asarray(tab.c)

        xi = rng.random()
        mu = float(np.interp(xi, cdf, x))

        return np.clip(mu, -1.0, 1.0)

    def __init__(
        self,
        data_dir="/Users/benweihrauch/Desktop/Nuclear Safety/Code/master/NuclearData",
    ):
        self.data_dir = data_dir
        self.cache = {}
        self.nu_bar_funcs = {}  # isotope → callable  ν̄(E_eV)
        # HDF5 library uses GNDS-style names
        self.iso_to_file = {
            "U234": "U234.h5",
            "U235": "U235.h5",
            "U236": "U236.h5",
            "U238": "U238.h5",
            "Pu239": "Pu239.h5",
            "Pu240": "Pu240.h5",
        }
        self.mt_map = {
            "total": 1,
            "elastic": 2,
            "inelastic": 4,
            "fission": 18,
            "capture": 102,
            "n2n": 16,
        }

    def _load(self, isotope):
        filename = self.iso_to_file.get(isotope)
        if not filename:
            raise ValueError(f"Isotope '{isotope}' not in naming map.")
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"HDF5 file not found: {path}\n"
                f"Download pre-processed library from https://openmc.org/official-data-libraries/"
            )

        print(f"[{isotope}] Loading from HDF5...")
        nuclide = openmc.data.IncidentNeutron.from_hdf5(path)
        self.cache[isotope] = nuclide

        # Extract ν̄(E) from fission reaction (MT=18) neutron product yield
        self._extract_nu_bar(isotope, nuclide)

        print(f"[{isotope}] Ready.")

    def _extract_nu_bar(self, isotope, nuclide):
        """
        Extract energy-dependent ν̄ from the HDF5 data and store it as a
        **picklable** wrapper (required for multiprocessing).

        Search strategy
        ---------------
        1. Neutron-product yield on MT=18 (standard for major actinides).
        2. Any product on MT=18 whose yield evaluates to a ν̄-like value (>1).
        3. Partial-fission reactions MT=19, 20, 21, 38 (same two passes).
        4. Known linear fallback for threshold fissioners (U-234, U-236).
        """

        # --- Known linear fallbacks for threshold fissioners ---------------
        # ν̄(E) = a + b × E(MeV),  valid above E_threshold
        # Coefficients consistent with ENDF/B-VIII.0 evaluations
        KNOWN_NUBAR = {
            "U234": {"a": 2.316, "b": 0.130, "E_thr": 0.8e6},
            "U236": {"a": 2.240, "b": 0.148, "E_thr": 0.6e6},
        }

        # Fission MTs to try: total, first-chance, second, third, fourth
        fission_mts = [mt for mt in [18, 19, 20, 21, 38] if mt in nuclide.reactions]

        if not fission_mts:
            print(f"[{isotope}]   No fission reactions found — ν̄ set to 0")
            self.nu_bar_funcs[isotope] = _ZeroNuBar()
            return

        # ------ Strategy 1 & 2: search products for a callable yield ------
        nu_raw = None
        source = ""

        for mt in fission_mts:
            rx = nuclide.reactions[mt]

            # Pass A – explicitly labelled neutron product
            for product in rx.products:
                if product.particle == "neutron" and product.yield_ is not None:
                    nu_raw = product.yield_
                    source = f"MT={mt} neutron product"
                    break
            if nu_raw is not None:
                break

            # Pass B – any non-photon product with a ν̄-like callable yield
            for product in rx.products:
                if product.yield_ is None:
                    continue
                # Skip photons/gammas — their multiplicity (~6–12) overlaps
                # with ν̄ range and would give a wrong answer
                if product.particle in ("photon", "gamma"):
                    continue
                try:
                    test = float(product.yield_(2.0e6))
                    if 1.5 < test < 5.5:  # sane ν̄ range (excludes γ mult.)
                        nu_raw = product.yield_
                        source = f"MT={mt} '{product.particle}' product"
                        break
                except Exception:
                    continue
            if nu_raw is not None:
                break

        # ------ Convert to pickle-safe wrapper ----------------------------
        if nu_raw is not None:
            wrapper = self._wrap_nu_bar(nu_raw, isotope)
            self.nu_bar_funcs[isotope] = wrapper
            self._print_nu_diag(isotope, wrapper, source)
            return

        # ------ Strategy 3: known linear fallback -------------------------
        if isotope in KNOWN_NUBAR:
            p = KNOWN_NUBAR[isotope]
            wrapper = _LinearNuBar(p["a"], p["b"], p["E_thr"])
            self.nu_bar_funcs[isotope] = wrapper
            print(
                f"[{isotope}]   ν̄(E) from linear fallback: "
                f"ν̄ = {p['a']:.3f} + {p['b']:.3f}·E(MeV), "
                f"threshold = {p['E_thr']/1e6:.1f} MeV, "
                f"ν̄(2 MeV) = {wrapper(2.0e6):.4f}"
            )
            return

        # ------ Nothing found at all -------------------------------------
        print(f"[{isotope}]   No ν̄ data found — set to 0")
        self.nu_bar_funcs[isotope] = _ZeroNuBar()

    # -----------------------------------------------------------------
    def _wrap_nu_bar(self, nu_raw, isotope):
        """
        Convert an openmc yield object (Tabulated1D, Polynomial, …)
        into one of our picklable wrappers.
        """
        # Tabulated1D — has .x (energies) and .y (values)
        if hasattr(nu_raw, "x") and hasattr(nu_raw, "y"):
            return _TabulatedNuBar(nu_raw.x, nu_raw.y)

        # Polynomial — has .coef  [c₀, c₁, …]  where ν̄(E_eV) = Σ cᵢ E^i
        if hasattr(nu_raw, "coef"):
            coef = list(nu_raw.coef)
            # Determine fission threshold from XS data if possible
            E_thr = 0.0
            try:
                nuclide = self.cache[isotope]
                if 18 in nuclide.reactions:
                    xs_dict = nuclide.reactions[18].xs
                    temp_key = list(xs_dict.keys())[0]
                    xs_tab = xs_dict[temp_key]
                    if hasattr(xs_tab, "x"):
                        E_thr = float(xs_tab.x[0])  # first tabulated energy
            except Exception:
                pass
            print(
                f"[{isotope}]   ν̄ stored as Polynomial: "
                f"coef = {[f'{c:.6e}' for c in coef]}, "
                f"E_thr = {E_thr/1e6:.2f} MeV"
            )
            return _PolynomialNuBar(coef, E_threshold_eV=E_thr)

        # Unknown callable — sample it into a table
        try:
            energies = np.logspace(-1, 7.3, 500)  # 0.1 eV to 20 MeV
            values = [float(nu_raw(E)) for E in energies]
            print(
                f"[{isotope}]   ν̄ sampled from unknown callable "
                f"({type(nu_raw).__name__})"
            )
            return _TabulatedNuBar(energies, values)
        except Exception:
            print(f"[{isotope}]   WARNING: could not convert ν̄ callable — set to 0")
            return _ZeroNuBar()

    # -----------------------------------------------------------------
    @staticmethod
    def _print_nu_diag(isotope, wrapper, source):
        """Print diagnostic ν̄ values at a few reference energies."""
        test_energies = [0.0253, 1.0e6, 2.0e6]
        vals = [f"{E/1e6:.4g} MeV→{wrapper(E):.4f}" for E in test_energies]
        print(f"[{isotope}]   ν̄(E) loaded from {source}: {', '.join(vals)}")

    # -----------------------------------------------------------------
    def get_nu(self, isotope, energy_eV):
        """
        Return the tabulated ν̄ (average neutrons per fission) at the given
        incident neutron energy.

        Parameters
        ----------
        isotope : str       e.g. "U235"
        energy_eV : float   incident neutron energy in eV

        Returns
        -------
        float   ν̄ at that energy
        """
        if isotope not in self.cache:
            self._load(isotope)

        if isotope not in self.nu_bar_funcs:
            raise ValueError(
                f"No ν̄ data for '{isotope}'. "
                f"Fission reaction (MT=18) may not exist in the HDF5 file."
            )

        return float(self.nu_bar_funcs[isotope](energy_eV))

    def get_xs(self, isotope, reaction, energy_ev):
        mt = self.mt_map.get(reaction.lower())
        if mt is None:
            raise ValueError(f"Unknown reaction '{reaction}'.")

        if isotope not in self.cache:
            self._load(isotope)

        nuclide = self.cache[isotope]

        # MT=4 (total inelastic) is sometimes absent — sum discrete levels instead
        if mt == 4 and 4 not in nuclide.reactions:
            inelastic_mts = [m for m in nuclide.reactions if 51 <= m <= 91]
            if not inelastic_mts:
                return 0.0
            total = 0.0
            e = np.array([float(energy_ev)])
            for m in inelastic_mts:
                xs_dict = nuclide.reactions[m].xs
                temp_key = list(xs_dict.keys())[0]
                try:
                    total += float(xs_dict[temp_key](e)[0])
                except Exception:
                    pass
            return total

        if mt not in nuclide.reactions:
            raise KeyError(
                f"MT={mt} ({reaction}) not found for {isotope}. "
                f"Available: {sorted(nuclide.reactions.keys())}"
            )

        xs_dict = nuclide.reactions[mt].xs
        temp_key = list(xs_dict.keys())[0]
        xs_func = xs_dict[temp_key]
        return float(xs_func(np.array([float(energy_ev)]))[0])

    def get_inelastic_levels(self, isotope):
        """
        Returns a list of (excitation_energy_eV, cross_section_function) tuples
        for each discrete inelastic level MT=51..91, plus the mass ratio A/(A+1).
        """
        if isotope not in self.cache:
            self._load(isotope)

        nuclide = self.cache[isotope]

        # Mass ratio A/(A+1) — AWR is atomic weight ratio (A relative to neutron mass)
        A = nuclide.atomic_weight_ratio
        mass_factor = A / (A + 1)

        levels = []
        for mt in sorted(m for m in nuclide.reactions if 51 <= m <= 91):
            rx = nuclide.reactions[mt]
            Q_i = rx.q_value  # excitation energy in eV (negative of Q for inelastic)
            xs_dict = rx.xs
            temp_key = list(xs_dict.keys())[0]
            xs_func = xs_dict[temp_key]
            levels.append(
                {
                    "mt": mt,
                    "level": mt - 50,  # level index 1, 2, 3...
                    "Q_eV": Q_i,  # Q-value in eV (negative = energy lost)
                    "E_thresh": -Q_i * (A + 1) / A,  # threshold energy in eV
                    "xs_func": xs_func,
                }
            )

        return mass_factor, levels

    def get_inelastic_scattered_energy(self, isotope, E_in_eV):
        """
        For a given incident energy, returns a list of open inelastic channels with:
        - level index
        - Q value
        - cross section at E_in
        - outgoing neutron energy (lab frame, avg over isotropic CoM distribution)
        Only returns levels whose threshold is below E_in.
        """
        mass_factor, levels = self.get_inelastic_levels(isotope)
        A = self.cache[isotope].atomic_weight_ratio

        results = []
        e_arr = np.array([float(E_in_eV)])

        for lv in levels:
            if E_in_eV < lv["E_thresh"]:
                continue  # channel not open yet

            sigma = float(lv["xs_func"](e_arr)[0])
            if sigma <= 0:
                continue

            # Outgoing energy in lab frame (target at rest, CoM isotropic)
            E_out = (A / (A + 1)) * ((A / (A + 1)) * E_in_eV + lv["Q_eV"])

            results.append(
                {
                    "mt": lv["mt"],
                    "level": lv["level"],
                    "Q_eV": lv["Q_eV"],
                    "xs_barn": sigma,
                    "E_in_eV": E_in_eV,
                    "E_out_eV": E_out,
                    "E_ratio": E_out / E_in_eV,  # how much energy the neutron retains
                }
            )

        return results

    def sample_scattering(self, isotope, mt, E_in_eV, rng=None):
        """
        Sample mu_lab and E_out_eV for a discrete inelastic level (MT=51-90)
        or elastic (MT=2).

        Returns:
            mu_lab  : cos(theta) in lab frame
            E_out_eV: outgoing neutron energy in lab frame
            phi     : azimuthal angle in [0, 2pi), sampled uniformly
        """
        if rng is None:
            rng = np.random.default_rng()

        if isotope not in self.cache:
            self._load(isotope)

        nuclide = self.cache[isotope]
        A = nuclide.atomic_weight_ratio
        rx = nuclide.reactions[mt]

        # --- 1. Sample mu in CoM frame from tabulated angular distribution ---
        dist = rx.products[0].distribution[0]  # UncorrelatedAngleEnergy
        angle = dist.angle  # AngleDistribution

        # Find bracketing incident energy points and interpolate CDF
        energies = angle.energy
        idx = np.searchsorted(energies, E_in_eV, side="right") - 1
        idx = np.clip(idx, 0, len(energies) - 2)

        # Sample from the nearest energy point (or interpolate between two)
        E_lo, E_hi = energies[idx], energies[idx + 1]
        f = (E_in_eV - E_lo) / (E_hi - E_lo)  # interpolation fraction

        # Sample using SEEDED rng (NOT OpenMC's internal unseeded .sample())
        if rng.random() < f:
            mu_cm = self._sample_tabular_seeded(angle.mu[idx + 1], rng)
        else:
            mu_cm = self._sample_tabular_seeded(angle.mu[idx], rng)

        # --- 2. Exact outgoing energy from kinematics (discrete level) ---
        if mt == 2:
            # Elastic: Q=0
            Q = 0.0
        else:
            Q = rx.q_value  # negative for inelastic (eV)

        E_out = (A / (A + 1)) * ((A / (A + 1)) * E_in_eV + Q)

        # --- 3. Convert mu from CoM to lab frame ---
        # gamma = v_cm / v_out_cm  =  sqrt(E_in / E_out) * 1/(A+1)  for elastic
        # For general case:
        gamma = (1 / (A + 1)) * np.sqrt(E_in_eV / E_out)
        mu_lab = (mu_cm + gamma) / np.sqrt(1 + 2 * gamma * mu_cm + gamma**2)

        # --- 4. Uniform azimuthal angle ---
        phi = rng.uniform(0, 2 * np.pi)

        return mu_lab, E_out, phi

    def get_scattered_direction(self, mu_lab, phi, omega_in):
        """
        Rotate incoming direction vector omega_in = (u, v, w) by
        polar angle mu_lab=cos(theta) and azimuthal angle phi.

        Returns new direction unit vector (u', v', w').
        Standard rotation used in all Monte Carlo transport codes.
        """
        u, v, w = omega_in
        sin_theta = np.sqrt(max(0.0, 1 - mu_lab**2))

        # Avoid division by zero when moving along z-axis
        if abs(w) > 1 - 1e-10:
            sign = np.sign(w)
            u_new = sin_theta * np.cos(phi)
            v_new = sin_theta * np.sin(phi)
            w_new = sign * mu_lab
        else:
            denom = np.sqrt(1 - w**2)
            u_new = (
                mu_lab * u + sin_theta * (u * w * np.cos(phi) - v * np.sin(phi)) / denom
            )
            v_new = (
                mu_lab * v + sin_theta * (v * w * np.cos(phi) + u * np.sin(phi)) / denom
            )
            w_new = mu_lab * w - sin_theta * denom * np.cos(phi)

        # Renormalize to correct floating point drift
        norm = np.sqrt(u_new**2 + v_new**2 + w_new**2)
        return np.array([u_new, v_new, w_new]) / norm


# --- SANITY CHECK ---
if __name__ == "__main__":
    db = NuclearData(
        data_dir="/Users/benweihrauch/Desktop/Nuclear Safety/Code/master/NuclearData"
    )

    rng = np.random.default_rng(42)

    print("\nSampling 5 inelastic scatters for U238 at 2 MeV (MT=51, level 1):")
    print(f"{'mu_lab':<10} | {'E_out (keV)':<14} | {'phi':<8}")
    print("-" * 40)

    omega_in = np.array([0.0, 0.0, 1.0])  # neutron travelling in +z

    for _ in range(5):
        mu_lab, E_out, phi = db.sample_scattering("U238", 51, 2.0e6, rng)
        omega_out = db.get_scattered_direction(mu_lab, phi, omega_in)
        print(f"{mu_lab:<10.4f} | {E_out/1e3:<14.4f} | {phi:<8.4f}")
        print(f"  direction: {omega_out}")


# --- SANITY CHECK FOR INELASTIC ---
"""     levels = db.get_inelastic_scattered_energy("U238", 2.0e6)

    nuclide = db.cache["U238"]
    rx = nuclide.reactions[51]
    dist = rx.products[0].distribution[0]
    angle = dist.angle

    print("Incident energy grid (first 5):", angle.energy[:5])
    print("Number of energy points:", len(angle.energy))
    print("Type of mu[0]:", type(angle.mu[0]))
    print("dir of mu[0]:", dir(angle.mu[0]))

    # Check first distribution
    mu0 = angle.mu[0]
    if hasattr(mu0, "x"):
        print("mu[0].x (first 5):", mu0.x[:5])  # cos(theta) grid
        print("mu[0].y (first 5):", mu0.y[:5])  # pdf values
        print("mu[0].c (first 5):", mu0.c[:5])  # cdf values """

# Inelastic print
"""     print("\nInelastic levels for U238 at 2 MeV:")
    print(
        f"{'Level':<8} | {'Q (keV)':<12} | {'E_thresh (keV)':<16} | {'XS (b)':<10} | {'E_out (MeV)':<12} | {'E retained':<10}"
    )
    print("-" * 80)

    levels = db.get_inelastic_scattered_energy("U238", 2.0e6)
    for lv in levels:
        print(
            f"{lv['level']:<8} | "
            f"{lv['Q_eV']/1e3:<12.3f} | "
            f"{-lv['Q_eV']/1e3*(239/238):<16.3f} | "  # threshold in keV
            f"{lv['xs_barn']:<10.5f} | "
            f"{lv['E_out_eV']/1e6:<12.5f} | "
            f"{lv['E_ratio']:<10.4f}"
        ) """

# samples of all cross sections
"""
    energies = {"Thermal (0.025eV)": 0.0253, "Fast (2MeV)": 2.0e6}
    isotopes = ["U235", "U238", "Pu239", "Pu240"]
    # Added elastic and inelastic to the list
    reactions = ["fission", "capture", "elastic", "inelastic"]

    print("=" * 75)
    print(
        f"{'Isotope':<8} | {'Reaction':<10} | {'Energy Level':<18} | {'XS (barn)':<10}"
    )
    print("=" * 75)

    for iso in isotopes:
        for rx in reactions:
            for label, e_val in energies.items():
                try:
                    sigma = db.get_xs(iso, rx, e_val)
                    print(f"{iso:<8} | {rx:<10} | {label:<18} | {sigma:<10.4f}")
                except Exception as e:
                    # Catching cases where 'inelastic' might not exist for certain isotopes/libraries
                    print(f"{iso:<8} | {rx:<10} | {label:<18} | ERROR: {e}")
            print("-" * 75) """
