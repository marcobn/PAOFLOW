# ARPES simulation in PAOFLOW: a native photoemission module with band-resolved self-energies (plan)

Status (October 27 2026): **implemented** as `src/PAOFLOW/arpes/` and
`src/PAOFLOW/elphon/self_energy.py`, see [arpes.md](arpes.md) for the user documentation.
This note keeps the assessment of chinook and the design decisions. Decision taken with the
user: do not depend on chinook at run time; port the photoemission core of
[chinook](https://github.com/qmlab-ubc-ca/chinook) (R. P. Day, B. Zwartsenberg, I. S. Elfimov
and A. Damascelli, npj Quantum Materials 4, 54 (2019); MIT licence, Copyright (c) 2018 Ryan
Patrick Day) with attribution, built on PAOFLOW's own Hamiltonian, basis, spherical harmonics
and Gaunt coefficients, and extend it to the band-resolved electron-phonon self-energy of the
dense-q machinery. All items of the plan are implemented, including the spinor bases, the chinook parity
fixture and the Pb, MgB₂ and Si examples; the Debye–Waller term is not planned.

## What chinook contains and what we take

chinook is 10.7k lines of pure Python. Its dependency graph splits cleanly:

| chinook module (lines) | content | decision |
|---|---|---|
| `ARPES_lib` (1293): `experiment` (datacube, `M_compute`, `spectral`, spin projectors, `all_Y`, `projection_map`, `Gmat_make`, `G_dic`, `pol_2_sph`, `gen_SE_KK`) | the photoemission physics: plane-wave final state, dipole matrix elements per state, intensity assembly, Gaussian resolution, Fermi factor, Kramers–Kronig helper | **re-implement, vectorised** (chinook loops over states in Python); keep the formalism and the conventions |
| `radint_lib` (473) + `adaptive_int` (157) | radial integrals $B^{l'}_{nl}(k) = i^{l'}\int dr\,R_{nl}(r)\,r^3 j_{l'}(kr)$ with an adaptive integrator; 'slater', 'hydrogenic', 'grid', 'exec', 'fixed' radial functions; final-state phase shifts | **re-implement** on the PAO radial grids (Simpson on `basis[i]['r']`, as `radialfft_simpson`), tabulated on a kinetic-energy grid; keep phase shifts and 'fixed' |
| `electron_configs` (260) + `electron_configs.txt`, `atomic_mass` (104) + txt | Slater $Z_{\rm eff}$ screening rules and hydrogenic radial functions | **port with attribution** (fallback when an orbital has no PAO radial function, e.g. user tight-binding models) |
| `Ylm` (363): `Y(l, m)` up to $l = 4$, `gaunt`, `Yproj` | complex spherical harmonics and Gaunt coefficients | **not ported**: PAOFLOW has the QE real harmonics (`projection/do_atwfc_proj.calc_ylmg`, QE `ylmr2` order) and real Gaunt coefficients (`hamiltonian/_two_center.real_gaunt_coefficient`); the dipole operator components $x, y, z$ are real $l = 1$ harmonics, so the whole matrix element is formulated in the real basis of the PAO Hamiltonian |
| `orbital` (341): `projdict`, `rot_projection`; `wigner` (187), `rotation_lib` (180) | cubic-harmonic labels as $Y_l^m$ projections; Wigner-D rotation of orbitals for sample tilt | **not ported**: real harmonics need no projection table; rotations from `hamiltonian/pao_sym` (Wigner-D, `get_wigner_so`) |
| `klib` (401): `kz_kpt`, `kmesh`; `tilt` (184): `gen_kpoints` | $k_f$, $k_z(h\nu, k_\parallel, V_0, W)$, momentum and emission-angle meshes, azimuthal rotation | **port** (~100 lines, attributed) into `kinematics.py` |
| `intensity_map` (185) | map container, writers | **re-implement** minimal (`IntensityMap` dataclass, npz + dat, GPAO plots) |
| `TB_lib`, `H_library`, `SlaterKoster`, `build_lib` (1.7k) | tight-binding models | **not needed**: `HRs` is the model (`models/sk_fitting` exists for SK fits) |
| `slab` (899), `surface_vector` (268) | finite slab from bulk hoppings along a Miller index | **native replacement** (~150 lines): slab $H$ from `HRs` with a vacuum gap, orbital depths; PAOFLOW's doubling/surface tools stay for the semi-infinite case |
| `dos`, `FS_tetra`, `tetrahedra`, `operator_library`, `Tk_plot`, `matplotlib_plotter`, `orbital_plotting`, `lattice_lib`, `v3find`, `angle_expt` (3.5k) | DOS, Fermi surfaces, expectation values, GUIs | **not ported** (PAOFLOW has DOS, Fermi surfaces, spin textures and GPAO) |

Net: about 1.5k new lines, of which ~400 are adapted from chinook. Licensing: MIT code inside
a GPL-3.0-or-later project is fine as long as the MIT notice travels with it; add
`THIRD_PARTY_NOTICES.md` at the repository root (chinook's licence text, the commit ported,
the list of adapted functions), a module docstring in `arpes/__init__.py` citing Day et al.
and asking users to cite it, and a per-function "adapted from chinook `klib.kz_kpt`" line.

## Formalism (one-step model, plane-wave final state, dipole approximation)

For a state $|n\mathbf k\rangle = \sum_i c_{i,n\mathbf k}\,|\phi_i,\mathbf k\rangle$ in the PAO
Bloch basis (orbital $i$: atom $\kappa_i$ at $\boldsymbol\tau_i$, real harmonic $l_i m_i$,
radial function $R_i(r)$ from `arry['basis'][i]['wfc']` on `['r']`) and a photoelectron
$\mathbf k_f$ with $|\mathbf k_f|^2 = 2m(h\nu + \varepsilon_{n\mathbf k} - W)/\hbar^2$,
$\mathbf k_{f\parallel} = \mathbf k_\parallel$ and $k_{f z}$ from the inner potential $V_0$
(chinook `kz_kpt`), the dipole matrix element for polarisation component $\alpha \in \{x, y, z\}$ is

$$
M^\alpha_{n\mathbf k} = \sum_i c_{i,n\mathbf k}\,\sigma_i\,e^{-d_i/2\lambda}\,e^{-i(\mathbf k_f - \mathbf k)\cdot\boldsymbol\tau_i}
\sum_{l' = l_i \pm 1}\; B^{l'}_{i}(|\mathbf k_f|)\sum_{m'} G(l_i m_i;\,1\alpha;\,l' m')\,
Y_{l' m'}(\hat{\mathbf k}_f),
$$

with $B^{l'}_i(k) = i^{l'}\!\int_0^\infty dr\,R_i(r)\,r^3 j_{l'}(kr)$ (the "$r^3$" is
$r^2\,dr$ times the dipole $r$), $G$ the real Gaunt coefficients, $Y_{l'm'}$ the real
harmonics (QE order), $\sigma_i$ an orbital cross-section factor and $d_i$ the depth below the
surface with the escape depth $\lambda$ (`mfp`). The phase $e^{-i(\mathbf k_f-\mathbf k)\cdot\boldsymbol\tau_i}$
is the gauge term that chinook absorbs into its atomic-gauge hopping vectors; PAOFLOW's
cell-periodic gauge makes it explicit (it reduces to $e^{-i(k_{fz}-k_z)\tau_{iz}}$ for bulk
calculations with $k_z = k_{fz}$, i.e. unity). Then

$$
I(\mathbf k, \omega) = \sum_n \Big|\sum_\alpha \epsilon_\alpha M^\alpha_{n\mathbf k}\Big|^2\,
A_{n\mathbf k}(\omega)\,f(\omega, T)\;\otimes\;g_{\Delta E}\otimes g_{\Delta k},\qquad
A_{n\mathbf k}(\omega) = -\frac1\pi\,\mathrm{Im}\,\frac{1}{\omega - \varepsilon_{n\mathbf k} - \Sigma_{n\mathbf k}(\omega) + i0^+},
$$

with the polarisation $\boldsymbol\epsilon$ in the sample frame (chinook: Cartesian, or
spherical components $Y_{1,\pm1}, Y_{1,0}$) and Gaussian resolutions. Spin-ARPES: the spinor
components of $M$ projected on the detector axis (chinook `sarpes_projector`), for
fully-relativistic PAO bases after the $|l\,j\,m_j\rangle \to |l\,m\rangle\otimes|\sigma\rangle$
rotation (`clebsch_jm_matrix`). Everything is an array over (k, band, α): one `einsum`
chain per k-block, no per-state loop.

The self-energy is the extension: chinook accepts $\Sigma(\mathbf k, \omega)$ shared by all
bands; here $\Sigma_{n\mathbf k}(\omega, T)$ is band- and k-resolved, from

1. the isotropic Eliashberg form from $\alpha^2F$ (`eliashberg_dense_q`), in the digamma
   closed form (Engelsberg–Schrieffer; $\mathrm{Re}\,\Sigma'(0) = -\lambda$);
2. the anisotropic Fermi-surface form from $\alpha^2F_{n\mathbf k}$ of `fermi_surface_coupling`;
3. the full Fan–Migdal $\Sigma_{n\mathbf k}(\omega, T)$ at arbitrary k from the dense vertex
   (EPW `specfun_el`): $\sum_{m\nu\mathbf q}|g_{mn\nu}|^2\big[\tfrac{n+f_m}{\omega-\varepsilon_m+\omega_\nu+i\eta}+\tfrac{n+1-f_m}{\omega-\varepsilon_m-\omega_\nu+i\eta}\big]$,
   whose on-shell imaginary part is the rate of `carrier_scattering` (consistency test);
   needs a frequency-resolved kernel and an off-grid `bloch_states_at` for k, k+q (q on the
   dense grid, irreducible wedge);
4. user models (constant, polynomial, Σ(ω) grid, Kramers–Kronig from Im Σ) for fits.

## Package layout

| module | content |
|---|---|
| `arpes/basis.py` | `PhotoemissionBasis.from_data_controller(dc)`: per orbital atom, $Z$, position (Å), $(n, l, m)$, radial function, depth, $\sigma_i$; spinor rotation for FR runs; Slater/hydrogenic fallbacks (`arpes/atomic_radial.py`, ported) |
| `arpes/radial.py` | `radial_integrals(basis, kinetic_ev, phase_shifts=None)` → $B^{l\pm1}_i$ tables + interpolation; 'fixed' values |
| `arpes/kinematics.py` | $k_f$, $k_z(h\nu, k_\parallel, V_0, W)$, momentum cube / k-path / emission-angle mesh, azimuthal rotation, slit geometry (ported from `klib`/`tilt`) |
| `arpes/matrix_elements.py` | `dipole_matrix_elements(basis, eigvecs, k, kf)` → `(nk, nband, 3)`; polarisation contraction; spin projection |
| `arpes/spectral.py` | `spectral_function(e, sigma, w)`, self-energy adapters (constant / grid / k / nk), Fermi factor, resolution convolution, `intensity(...)` |
| `arpes/slab.py` | finite slab Hamiltonian from `HRs` along a Miller index with vacuum, orbital depths; bulk-$\Sigma$ approximation for slab states |
| `arpes/experiment.py` | `ArpesExperiment` (hv, W, V0, T, pol, resolution, kz mode, mfp, cube/path/angles, spin axis), `simulate(data_controller, self_energy=None)` → `IntensityMap`; `arpes/io.py` writers |
| `elphon/self_energy.py` | items 1–3 above, `quasiparticle_renormalization` |
| `PAOFLOW.py` / `GPAO.py` | `arpes_basis()`, `electron_self_energy(...)`, `arpes(experiment, self_energy=)`; `plot_arpes` (maps, EDC/MDC, kink) |
| `pyproject.toml` | no new dependency (numpy, scipy); `THIRD_PARTY_NOTICES.md` |

Bands and eigenvectors come from PAOFLOW's `HRs` at the cube points (own Bloch evaluation,
no commensurability with any grid), so an ARPES run reads as `PAOFLOW(...)` →
`projections` → `pao_hamiltonian` → `arpes(experiment)`; with e-ph,
`electron_self_energy(k=experiment.kpoints, ...)` in between.

## Validation

1. Parity with chinook on its own tutorial models (graphene, Sr₂IrO₄ from the chinook docs)
   using Slater radial functions: identical $|M|^2$ maps to 1e-8 (chinook run once in a
   throwaway environment with numpy < 2; the maps become test fixtures).
2. Gauge: a two-site chain with one $s$ orbital per site gives the known $e^{i\mathbf k\cdot\boldsymbol\tau}$
   interference; bulk $k_z$ path reproduces the expected $h\nu$ periodicity.
3. Real-harmonic formulation: $\sum_\alpha|M^\alpha|^2$ summed over a closed shell is
   rotation-invariant; selection rules $l' = l\pm1$, $\Delta m$ by polarisation.
4. Radial integrals: PAO vs Slater functions for Si $3p$ (shape, magnitude); 'fixed' round trip.
5. Self-energy: $-\mathrm{Im}\,\Sigma_{n\mathbf k}(\varepsilon_{n\mathbf k})$ equals
   $\hbar/2\tau_{n\mathbf k}$ of `carrier_scattering_dense_q` on grid k (1e-10); slope
   $-\lambda$ against `eliashberg_dense_q`; Kramers–Kronig; EPW `specfun_el` parity on the
   Si/Pb data of the DFPT port.
6. Physics: Pb(111) kink and $1+\lambda$ (Reinert et al. 2003); MgB₂ σ-band kink at the
   $E_{2g}$ energy (Uchiyama et al. 2002); Si valence-band thermal broadening and shift.

## Phases and effort

- **Phase 0** (4–5 days): `arpes/` with basis, radial, kinematics, matrix elements, spectral,
  experiment, io; `THIRD_PARTY_NOTICES.md`; unit tests 2–4; chinook parity fixtures (1).
- **Phase 1** (3–4 days): `elphon/self_energy.py` (isotropic, anisotropic, Fan–Migdal at
  arbitrary k, MPI over q), tests 5, EPW parity.
- **Phase 2** (2–3 days): slabs, spin-ARPES, FR spinor rotation, `PAOFLOW`/`GPAO` methods,
  docs (`Electron-Phonon-Coupling.md` section "Spectral functions and ARPES", a user-guide page).
- **Phase 3** (2 days + runs): examples `examples/elphon_example/{Pb,MgB2,Si}/arpes.py` with
  reports (Pb FR grids on fedora), CI with the elphon asset.

About three weeks. Decisions proposed: real-harmonic formulation with PAOFLOW's Gaunt
coefficients (no complex-$Y$ projection tables); PAO radial functions by default, Slater
fallback; cell-periodic gauge with the explicit $\boldsymbol\tau$ phase; no GUI; chinook
cited in the docs and the module docstring; the self-energy API takes arrays `(nk, nband, nw)`
so any source (e-ph, impurity, user fit) plugs in.
