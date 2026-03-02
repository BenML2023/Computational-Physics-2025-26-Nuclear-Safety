# Monte Carlo Criticality Uncertainty Analysis

A from-scratch Monte Carlo neutron transport code for computing critical masses of fissile assemblies and quantifying the uncertainty budget from nuclear data.

Built in Python using ENDF/B-VIII.0 cross sections via OpenMC's HDF5 data format.

---

## What This Does

Given a fissile fuel composition (e.g. 93 wt% enriched uranium) and a geometry (sphere, cube, or cylinder), the code finds the **critical dimension** -- the size at which the neutron chain reaction is exactly self-sustaining (k_eff = 1.0) -- and computes a complete **uncertainty budget** broken down into statistical and systematic components.

The pipeline answers three questions:

1. **How reproducible is the answer?** (Phase 1 -- statistical uncertainty from independent replica runs)
2. **How sensitive is R_crit to each nuclear data channel?** (Phase 2 -- perturbation analysis with the sandwich rule)
3. **How does a reflector change everything?** (Phase 3 -- albedo scan with full uncertainty at each point)

The final output is a combined uncertainty: **sigma_total = sqrt(sigma_stat^2 + sigma_syst^2)**, with a variance budget showing whether statistical noise or nuclear data uncertainties dominate.

---

## Physics

### Neutron Transport

The transport is a standard power-iteration Monte Carlo criticality calculation:

1. **Source neutrons** are sampled uniformly within the geometry with energies drawn from a Watt fission spectrum (isotope-dependent parameters from ENDF).
2. Each neutron is tracked collision-by-collision:
   - **Free flight**: distance to next collision sampled from exponential distribution using energy-dependent macroscopic total cross section Sigma_t(E).
   - **Boundary check**: if the neutron reaches the surface before colliding, it either escapes (bare) or reflects back with probability equal to the reflector albedo.
   - **Collision**: an isotope is sampled weighted by N_i * sigma_t,i(E), then a reaction channel (fission, capture, elastic, inelastic) is sampled from the partial cross sections.
   - **Fission** terminates the neutron and stores the fission site. The number of next-generation neutrons is tallied via the energy-dependent nu-bar(E) from ENDF.
   - **Capture** terminates the neutron (absorbed).
   - **Scattering** updates the neutron energy and direction using angular distributions from OpenMC's processed ENDF data (tabulated PDFs with seeded inverse-CDF sampling for reproducibility).
3. After all neutrons in a generation are processed, **k_eff = sum(nu-bar) / N_start** for that cycle.
4. Fission sites are resampled to form the next generation source (with the standard resample-to-N algorithm).
5. The first N_inactive cycles are discarded (source convergence), and k_eff is averaged over the remaining N_active cycles.

Shannon entropy of the fission source spatial distribution is tracked for convergence diagnostics.

### Critical Radius Search

For a given fuel and geometry type, the code runs k_eff calculations at several sizes spanning the expected critical range, fits a quadratic k_eff(R) curve, and finds the root where k_eff = 1.0 via bisection. This is done in parallel across a multiprocessing worker pool.

### Uncertainty Quantification

**Phase 1 -- Statistical uncertainty (sigma_stat):** N independent replica searches are run with different RNG seeds. The sample standard deviation s of the resulting R_crit values is the statistical uncertainty. The standard error SE = s/sqrt(N) quantifies the uncertainty on the mean.

**Phase 2 -- Systematic uncertainty (sigma_syst, sandwich rule):** Each nuclear data channel (fission, capture, elastic, inelastic cross sections and nu-bar for each isotope) is perturbed by +1% using *correlated sampling* (identical RNG seed as the baseline). This gives the sensitivity coefficient S_i = dR / d(sigma). The physical uncertainty contribution from each channel is:

```
delta_R_i = S_i * epsilon_i
```

where epsilon_i is the fractional 1-sigma uncertainty from ENDF covariance evaluations. The total systematic uncertainty is computed via the sandwich rule including the U-235 fission-capture cross-correlation:

```
sigma_syst^2 = sum(delta_R_i^2) + 2 * rho * delta_R_fission * delta_R_capture
```

**Phase 3 -- Reflector albedo scan:** Phases 1 and 2 are repeated at each albedo value from 0.0 (bare) to 0.9 (heavy reflector), producing R_crit(albedo) curves with full error bars.

### A Note on Fission Sensitivity

The sensitivity of R_crit to the U-235 fission cross section is surprisingly small (~0.008 cm for a +1% perturbation). This is correct physics, not a bug. Since k_inf = nu * sigma_f / (sigma_f + sigma_gamma), increasing sigma_f increases both the neutron source (numerator) and the absorption sink (denominator). The net effect scales as sigma_gamma / sigma_a, which is approximately 7% for HEU -- a near-perfect cancellation. By contrast, nu-bar appears only in the numerator, so its sensitivity is roughly 14x larger.

---

## Code Architecture

```
cs_getter.py                         Nuclear data interface (OpenMC HDF5)
    |
    v
fast_reactor_optimized.py            Transport engine + fuel definitions
    |
    v
fast_reactor_accel.py                Accelerated transport (pre-tabulated XS)
    |
    v
uncertainty_pipeline_parallelized.py Orchestrator (Phases 1/2/3)
```

### `cs_getter.py` -- Nuclear Data Interface

Wraps OpenMC's HDF5 nuclear data files. Provides:

- **`get_xs(isotope, reaction, energy)`** -- microscopic cross sections (barns) for elastic, inelastic, fission, capture, and total reactions.
- **`get_nu(isotope, energy)`** -- energy-dependent average fission neutron yield nu-bar(E), extracted from ENDF product yields. Supports Tabulated, Polynomial, and Linear representations with isotope-specific fallbacks for threshold fissioners like U-234 and U-236.
- **`sample_scattering(isotope, MT, E_in, rng)`** -- angular distribution sampling using seeded inverse-CDF. This replaces OpenMC's unseeded internal sampler to ensure full reproducibility with a fixed seed.
- **`get_inelastic_scattered_energy(isotope, E_in)`** -- discrete inelastic level sampling with threshold logic.

All random sampling uses a caller-supplied `numpy.random.Generator` so results are fully deterministic for a given seed.

Supported isotopes: U-234, U-235, U-236, U-238, Pu-239, Pu-240.

### `fast_reactor_optimized.py` -- Transport Engine

Core classes:

- **`FastReactorFuel`** -- holds isotope composition, atom densities, and nuclear data reference. Computes macroscopic cross sections, samples isotopes and reactions, handles scattering kinematics with proper CM-to-lab frame transformation.
- **`Geometry`** subclasses (`Sphere`, `Cube`, `Cylinder`) -- uniform source sampling, distance-to-boundary calculation, volume computation.
- **`run_fast_reactor_keff()`** -- the full power-iteration transport loop (reference implementation; the accelerated version in `fast_reactor_accel.py` is used in practice).

Fuel definitions:

- **`get_fresh_u235_fuel()`** -- 93 wt% HEU matching the Godiva/ORSphere benchmark (HEU-MET-FAST-001), density 18.75 g/cm3.

### `fast_reactor_accel.py` -- Accelerated Transport

**`PreTabulatedFuel`** pre-computes all microscopic cross sections and nu-bar on a fine log-spaced energy grid (5000 points, 100 eV to 25 MeV) at initialization time. During transport, lookups use a single binary search + linear interpolation instead of repeated HDF5 queries through the cache. This gives a 3-8x speedup with identical physics.

Pre-computed arrays:

- `micro_xs[n_iso, n_E, 4]` -- elastic, inelastic, fission, capture per isotope
- `nu_bar[n_iso, n_E]` -- fission neutron yield per isotope
- `macro_total[n_E]` -- macroscopic total cross section (for MFP)
- `iso_weights_cumsum[n_iso, n_E]` -- cumulative N_i * sigma_t,i (for isotope sampling)
- `rxn_cumsum[n_iso, n_E, 4]` -- cumulative partial XS (for reaction sampling)

Scattering angular distributions and inelastic level kinematics still delegate to the original `FastReactorFuel` object since they are not easily pre-tabulatable.

**`perturb_fast_fuel()`** creates a perturbed copy by multiplying one slice of the `micro_xs` or `nu_bar` array by a factor (e.g. 1.01), then recomputing all derived macroscopic quantities. This takes ~1 ms versus ~10 s for full re-tabulation from HDF5 -- critical for running 10 perturbation channels efficiently.

### `uncertainty_pipeline_parallelized.py` -- Orchestrator

Manages the three-phase analysis with parallel execution:

**One-time tabulation:** Cross sections are tabulated exactly once in the main process. All worker processes receive the pre-built `PreTabulatedFuel` via the multiprocessing pool initializer -- no redundant HDF5 reads.

**Parallel scan:** Each critical radius search dispatches N_SCAN_POINTS x N_SIMS_PER_PT tasks across N_WORKERS processes. Workers compute k_eff at assigned (geometry, size, seed) combinations; the main process collects results, fits k_eff(R), and finds the crossing.

**Correlated sampling:** Perturbation runs use the *same* RNG seed as the baseline. This means the random walk trajectories are identical except where the perturbed cross section causes a different outcome, dramatically reducing noise on dR.

**Sandwich rule:** Literature uncertainties (epsilon_i) from ENDF/B-VIII.0 covariance evaluations are stored for all 10 channels. The `compute_sandwich_uncertainty()` function multiplies sensitivities by these to get physical delta_R_i values and combines them with the U-235 fission-capture cross-correlation (rho = 0.5).

**Report output:** A text report with the full sensitivity table (S_i, epsilon_i, delta_R_i, delta_R_i^2), sandwich-rule calculation breakdown, and combined uncertainty budget. Phase 3 produces a four-panel diagnostic plot (R_crit vs albedo, critical mass vs albedo, uncertainty components, per-channel sensitivities).

---

## Configuration

All simulation parameters are set as module-level constants at the top of `uncertainty_pipeline_parallelized.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GEOMETRY_TYPE` | `"sphere"` | `"sphere"`, `"cube"`, or `"cylinder"` |
| `N_NEUTRONS` | 200 | Neutrons per cycle |
| `N_INACTIVE` | 25 | Inactive (burn-in) cycles |
| `N_ACTIVE` | 150 | Active (scoring) cycles |
| `N_REPLICAS` | 10 | Phase 1 independent replicas |
| `PERTURBATION` | 0.01 | Fractional perturbation (+1%) |
| `N_SCAN_POINTS` | 7 | Sizes per critical radius search |
| `N_WORKERS` | 12 | Parallel worker processes |
| `REFLECTOR_ALBEDO` | 0.0 | Baseline reflector albedo |
| `RUN_PHASE1/2/3` | `True` | Toggle individual phases |

The `PARAM_RANGES` dict sets the scan interval for each geometry (must bracket the critical dimension). `LITERATURE_UNCERTAINTIES` contains the ENDF covariance values.

For production runs, increase `N_NEUTRONS` to 1000+ and `N_ACTIVE` to 500+ for tighter statistics (at proportionally longer runtime).

---

## Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
- OpenMC Python API (`openmc.data` for HDF5 nuclear data parsing)
- ENDF/B-VIII.0 HDF5 data library -- download from [openmc.org](https://openmc.org/official-data-libraries/) and place the isotope `.h5` files (`U234.h5`, `U235.h5`, `U236.h5`, `U238.h5`) in the path configured in `cs_getter.py`

## Usage

```bash
# Run the full uncertainty pipeline
python uncertainty_pipeline_parallelized.py

# Output:
#   uncertainty_results_sphere.txt   -- full report with uncertainty budget
#   plots/albedo_scan/*.png          -- diagnostic plots (Phase 3)
```

Toggle phases with `RUN_PHASE1`, `RUN_PHASE2`, `RUN_PHASE3` flags. A typical Phase 3 albedo scan with 5 points takes 20-60 minutes depending on hardware and neutron count.

---

## Benchmarks

The bare HEU sphere should converge to R_crit of approximately 8.7 cm (Godiva critical mass of roughly 52 kg), consistent with the ICSBEP HEU-MET-FAST-001 benchmark.

---

## License

This project is for educational and research purposes.

