# Special displacement method (ZG supercells)

The special displacement method of Zacharias and Giustino (Phys. Rev. B **94**, 075125 (2016);
Phys. Rev. Research **2**, 013357 (2020)) replaces the thermal average over phonon
configurations by **one supercell per temperature**. Its atoms are displaced along every
phonon mode of the supercell with the thermal amplitude and a set of signs chosen so that the
clamped-ion expectation value of any observable equals the Williams–Lax average to second
order in the displacements. PAOFLOW builds that cell from the harmonic force constants, writes
the `pw.x` input, and reads the scf back into the PAO basis: the Hamiltonian of the displaced
cell carries the temperature dependence of the electrons (band shifts, inhomogeneous
widths of the static disorder, phonon-assisted optical transitions). What the method does not
give is the dynamical part (lifetimes, Fan–Migdal frequency dependence, non-adiabatic
effects): the ZG cell is the adiabatic, static limit of the electron–phonon self-energy. The
design note is [special-displacement-plan.md](special-displacement-plan.md).

## Library: `PAOFLOW.phonon.special_displacement`

Procedural functions, in the order of the recipe of `ZG.x` (EPW):

| function | what it does |
|---|---|
| `supercell_qpoints(dims)` | the q-points of a diagonal `dims` supercell in the primitive reciprocal basis, Γ first |
| `split_qpoints(q)` | the time-reversal-invariant set **A** (2q = G) and one member of every ±q pair, set **B** |
| `bose_occupation(freq_thz, T)`, `zg_amplitudes(freq_thz, T, is_gamma)` | $n_{\mathbf q\nu}$ and $\sigma_{\mathbf q\nu} = \sqrt{\hbar / 2 \omega_{\mathbf q\nu}}\,\sqrt{2 n_{\mathbf q\nu} + 1}$ in bohr·√amu; the acoustic modes at Γ and modes below `min_freq_thz` are frozen |
| `sign_table(nmode)`, `zg_sign_sequence(nmode, nq)` | the antithetic binary sign table of `ZG.x` ($2^{n_{\rm mode}-1}$ rows, capped at $2^{11}$), cycled with rotation and negation every block of q-points |
| `zg_error(signs, sigma, eig)` | the relative error of the mean-square displacements of the configuration against the thermal ones (the cross terms of the modes; set B weighted twice, real gauge at the A points) |
| `optimise_signs(sigma, eig, niters, error_thresh, seed, shuffle_qpoints)` | row permutations of the sign table (and a shuffle of the q-point order after the first attempt, which a monoatomic cell needs: only $3! = 6$ permutations exist otherwise) until the error is below `error_thresh` (default 0.05) |
| `zg_displacements(...)` | the displacement of every supercell atom, $\Delta\tau_{p\kappa} = \sqrt{M_0/M_\kappa}\sum_{\mathbf q\nu} S_{\mathbf q\nu}\, \sigma_{\mathbf q\nu}\, 2\,\mathrm{Re}[e^{i\mathbf q\cdot\mathbf R_p} e_{\kappa\nu}(\mathbf q)]$ with the $(1 + i)$ phase of the B points and $\cos$ at the A points (ZG.x Eq. 2) |
| `zg_configuration(phonon, dims, T, ...)` | all of the above from a phonopy object with force constants: the ZG supercell (`cell`), the ideal one (`reference`), displacements, signs, error, the thermal and configuration mean-square displacements |
| `phonopy_from_q2r(unitcell, fc_file)` | force constants of a `q2r.x` file (phonopy's `PH_Q2R` reader) on the PAOFLOW unit cell; the ZG supercell is independent of the force-constant supercell |
| `reference_configuration(conf)` | the undisplaced supercell of a configuration (no temperature, `<prefix>_zg_ref`) |
| `write_zg_inputs(data_controller, confs, zg_dir, ...)` | one `pw.x` scf input per configuration, a `<prefix>.npz` with displacements, signs, q-points, amplitudes and mean-square displacements, and the manifest `zg.json` |

Conventions: displacements in bohr, frequencies in THz, masses in amu (the ZG.x numbers are in
Å); the phonopy eigenvectors are mass-weighted, so the atomic displacement of a mode is
$e_{\kappa\nu}/\sqrt{M_\kappa}$; time reversal is assumed (no SOC-driven ±q asymmetry); the
acoustic sum rule is applied by phonopy, not by the ZG step. The identity
$\langle u^2\rangle_{\rm conf} = \langle u^2\rangle_T\,(1 + \text{error})$ is exact and is a unit
test.

### Electrons at the lattice temperature in the dielectric function (2 November 2026)

`dielectric_tensor(temperature=T)` replaces the DFT-smearing occupations of the Kubo–Greenwood sum (and the
Drude weight) by Fermi–Dirac occupations at `k_B T`, with the chemical potential re-determined on the
interpolated grid so that the electron count is conserved (`do_epsilon.fermi_dirac_occupations`, all MPI ranks).
`direct_dielectric(temperature_k=)` and `thermal_dielectric` expose it, and `zg.py --dielectric` uses the lattice
temperature of every cell unless `dielectric=dict(..., electronic_temperature=False)`: at 4500 K `k_B T` is
0.39 eV and the interband onsets of a metal move by that much. First use: the spectral emissivity of TaC at the
pyrometry temperatures (`production/XTa_project/TaC_elphon/thermal/`).

### Thermal force constants (the anharmonic self-consistent step, 2 November 2026)

`thermal_force_constants(elphon_dir, phonon_reference, ridge, translation_average)` reads the forces of the
displaced runs of a ZG cell (`electron_phonon(supercell_matrix=1)` on the cell, Cartesian ± pairs or the random
mode, through `elphon.fd_vertex.fd_force_constants`) and returns the local curvature at the thermal configuration
in Ry/bohr² and phonopy layout, averaged over the sub-cell translations of the reference supercell
(`translation_average_force_constants`: blocks sharing the primitive atoms and the relative translation are
averaged; idempotent). `phonopy_with_force_constants(unitcell, supercell_matrix, fc)` wraps them in a phonopy
object (`calculator='qe'`, primitive `'P'`) and `PAOFLOW.special_displacement(force_constants=fc,
force_constants_supercell=S)` builds the next ZG cell from them: one iteration of the A-SDM loop of Zacharias
and Giustino (the thermal force constants replace the harmonic ones until the ZG cell is self-consistent).
Pb demo in the validation README (`Workspace/PAOFLOW/validation/Pb_fd_fixed_basis`, `thermal_fc_demo.py`):
zero-point curvature within 1–5 % of the harmonic phonons, 300 K LA hardened by 3–6 % at fixed volume, the
regenerated 300 K cell within 0.5 % in rms displacement, i.e. converged at the first iteration for Pb.

## Orchestrator and workflow layer

```python
from PAOFLOW.PAOFLOW import PAOFLOW

pf = PAOFLOW(savedir='qe/nscf/pb.save', outputdir='thermal/zg')
pf.special_displacement(
    [0.0, 100.0, 300.0], supercell_matrix=3, force_constants_file='phonons/output/pb.fc',
    zg_dir='seed0', pp_dir='qe', prefix='pb', kgrid=(4, 4, 4), nbnd=324, seed=0, reference=True,
)   # -> thermal/zg/seed0/pb_zg_T{0,100,300}K.in, pb_zg_ref.in, *.npz, zg.json
```

`PAOFLOW.special_displacement` takes the force constants from `arry['phonopy']`
(`phonons()` with forces) or from a `q2r.x` file, loops the temperatures and writes the inputs;
the configurations stay on the controller as `arry['zg_configurations']`. The scf of each cell
is a plain `pw.x` run (`disk_io` default, the save is read back by PAOFLOW).

`PAOFLOW.elphon.workflows.thermal` turns the saves into observables:

- `thermal_spectral_function(system, zg_save, path, sym_points, ...)`: PAO Hamiltonians of the
  primitive save and of the ZG save (`pao_model_from_save`, the supercell model rescaled to the
  primitive `alat`, the atoms mapped to their ideal sites from the `.npz`), unfolded along a path
  with `models.band_unfold.unfold_bands`, and broadened into $A(\mathbf k, E; T) = \sum_n W_n(\mathbf k)\,
  g(E - E_n(\mathbf k))$ (Gaussian $g$); the sum rule $\int A\,dE = N_{\rm orb}^{\rm pc}$ is stored.
  Under MPI the PAOFLOW runs are collective, the unfolding runs on rank 0 and the result is
  broadcast.
- `ensemble_spectral_function(results)`: the configuration average of several seeds at one
  temperature; `band_statistics(result, band)`: first moment (shift) and second central moment
  (spread) of the unfolded weight of a primitive band at every k of the path, with the Gaussian
  broadening removed in quadrature; `peak_width_at_crossing(result)`: position and FWHM of the
  peak where a band crosses a given energy (the Fermi level of a metal).
- `static_linewidth(eliashberg_npz, temps)`: the adiabatic Eliashberg plateau
  $\Gamma(T) = \pi\int d\Omega\,\alpha^2F(\Omega)\,[2 n_B(\Omega, T) + 1]$, the half width that a
  static ensemble of displaced cells produces at the Fermi crossing of a metal (the Williams–Lax
  width); the FWHM is $2\Gamma$.
- `thermal_dielectric(system, zg_save, outdir, **kw)`: the direct dielectric function of the ZG
  cell (`optics.direct_dielectric`), i.e. $\varepsilon_2(\omega; T)$ of the Williams–Lax theory
  with the phonon-assisted transitions of the displaced cell.
- `zg_cells(zg_dir)`, `write_zg_run_script(zg_dir)`, `symmetry_points(system, points)`,
  `save/load_thermal_spectral`, `write/load_thermal_summary`, `plot_thermal_spectral_function`,
  `plot_thermal_summary`: the bookkeeping of the driver.

### The driver `thermal/zg.py` and the `THERMAL` dictionary

`thermal/zg.py` (a template of `gen/elphon_templates`, copied into the system folders and
written by `paoflow-gen elphon` when `thermal` is selected) has four phases:

```bash
python zg.py --write                    # thermal/zg/seed<s>/<prefix>_zg_T<T>K.in (+ _zg_ref.in), run_zg.sh
(cd zg && NP=8 bash run_zg.sh)          # pw.x scf of every cell (skips finished runs)
mpirun -np N python zg.py --spectral    # A(k, E; T) of every save, ensemble means, thermal_summary.{json,dat}
mpirun -np N python zg.py --dielectric  # epsilon(omega; T) of every ZG cell
python zg.py --plot                     # spectral_<label>.png, thermal_summary.png
```

The `THERMAL` dictionary of `system.py` holds the defaults (`temps_k`, `supercell`, `kgrid`,
`nbnd_per_atom`, `force_constants` (`None`: `phonons/output/<prefix>.fc` of `phonons.py`, run
first if missing), `seeds` (one cell per seed and temperature, averaged), `error_thresh`,
`reference`, `path`, `sym_points` (`None`: the lattice defaults of PAOFLOW's band paths),
`nk_per_seg`, `energy_window`, `broadening_ev`, `crossing_ev`, `half_window_ev`,
`dielectric`); every command-line option overrides them. The summary table prints, per
temperature, the number of cells, the band crossing `crossing_ev`, the peak position and FWHM at
the crossing, the mean shift and spread along the path, the sum rule and, when
`superconductivity/output/eliashberg.npz` exists, $2\Gamma(T)$ of the Eliashberg plateau. The
undisplaced reference cell (written with the first seed) measures the offset of the supercell
PAO bands from the primitive ones (basis, k-sampling), to which the thermal shifts are referred.

## Validation

- **Generator vs `ZG.x`** (Pb, `ph.x` 4³ + `q2r.x`, 4³ supercell, 64 atoms, 0 and 300 K): the
  same sets A/B (36 q-points: 8 + 28), the vibrational potential energy of the configuration
  0.023520 vs 0.023505 Ry (0 K) and 0.180641 vs 0.180640 Ry (300 K), rms displacement 0.1298 vs
  0.1298 and 0.3880 vs 0.3884 bohr, per-component mean-square displacements within 5 % of the
  thermal ones (within 4 % for `ZG.x`). The first sign sequence differs (row order of the table),
  the optimised error does not.
- **Pb 3³ cells, single configuration** (27 atoms, 0 / 100 / 300 K): the unfolded sum rule is
  exact (18.000 at every k); the band crossing $E_F$ along Γ–X appears as one sharp peak per k
  (0.97 / 0.91 / 0.84 of the weight) shifted by +24 / +63 / +143 meV almost uniformly along the
  band and starts to split at 300 K (per-k spread 77 meV). A single small cell therefore gives a
  random shift rather than a width: the width needs the configuration average over seeds (and
  larger cells), which is what the `seeds` ensemble of the driver provides. The reference numbers
  of the production α²F (scalar-relativistic) are $\Gamma_{\rm static}$ = 7.9 / 22.2 / 64.0 meV at
  0 / 100 / 300 K (FWHM 16 / 44 / 128 meV).
- **Undisplaced 27-atom reference cell** (the driver's `reference` entry): its unfolded bands
  coincide with the primitive ones to −3 / −4 meV at the k-points of the common grid (the same
  DFT eigenvalues up to the two scf runs) and to +3 meV at the Fermi crossing, with a residual slope
  of ±10 meV at ±0.4 eV between the two PAO models (off-grid interpolation: the supercell and the
  primitive real-space boxes wrap the long-range hoppings differently). That slope is common to
  all cells of the same size and cancels in the shifts referred to the reference, which is why
  the driver reports them; the 0 K cell lies +19 to +23 meV above the reference along the band
  (zero-point shift +21 ± 2 meV, weight 0.97 on one state). The peak measure reports the unfolded
  state relative to the primitive band *at the same path point*: on a band that moves 110 meV per
  k-step the absolute peak position carries the discretisation (±55 meV), not the physics.
- **Pb 4³ cells, ensemble of three configurations** (64 atoms, GPU 47–52 min per cell): referred to the undisplaced 64-atom cell, the Fermi-crossing band shifts by −4 / −17 / −41 meV and the static-disorder spread of the ensemble spectral function is 8.2 / 21.3 / 54.4 meV with 5 meV broadening at 0 / 100 / 300 K, i.e. Gaussian-equivalent widths of 15 / 49 / 127 meV against the Eliashberg plateau $2\Gamma$ = 16 / 43 / 122 meV computed on the same 4³ q-grid (within 15 %). The 3³ cells are not usable for Pb: their 14 q-points sample the Kohn-anomaly region where the 3³ sub-grid of the DFPT vertex gives λ = 2.7 against 0.8 converged, which produces large, seed-independent second-order shifts (+23 / +85 / +226 meV); the shift of a ZG cell is the sign-independent $|u_q|^2$ term on its own q-grid and converges only with the cell size, while the seeds average the interference terms that make the width.
- Unit tests: `tests/unit/phonon/test_special_displacement.py` (q-sets, amplitudes and their
  classical limit, sign table and sequence, error, the exact mean-square-displacement identity,
  the phonopy cell, the input writer with the reference cell) and
  `tests/unit/elphon/test_thermal_workflow.py` (band statistics, ensemble average and npz round
  trip, the Einstein-spectrum plateau, cell discovery and run script, lattice symmetry points,
  the summary files and figure).

## References

- M. Zacharias and F. Giustino, *One-shot calculation of temperature-dependent optical spectra
  and phonon-induced band-gap renormalization*, Phys. Rev. B **94**, 075125 (2016).
- M. Zacharias and F. Giustino, *Theory of the special displacement method for electronic
  structure calculations at finite temperature*, Phys. Rev. Research **2**, 013357 (2020).
- M. Zacharias, G. Volonakis, F. Giustino and J. Even, *Anharmonic lattice dynamics via the
  special displacement method*, Phys. Rev. B **108**, 035155 (2023) (A-SDM, out of scope).
- H. Lee *et al.*, *Electron–phonon physics from first principles using the EPW code*,
  npj Comput. Mater. **9**, 156 (2023) (`ZG.x`).
- P. B. Allen and V. Heine, J. Phys. C **9**, 2305 (1976); F. E. Williams, Phys. Rev. **82**, 281
  (1951); M. Lax, J. Chem. Phys. **20**, 1752 (1952).
