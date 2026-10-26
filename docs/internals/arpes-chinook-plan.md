# ARPES simulation with chinook: self-energy renormalisation and spectral functions (plan)

Status (October 26 2026): assessment and design, nothing implemented. The aim is to simulate
ARPES spectra from the PAO Hamiltonian with the electron-phonon self-energy of the dense-q
machinery, using [chinook](https://github.com/qmlab-ubc-ca/chinook) (Day, Zwartsenberg,
Elfimov and Damascelli, npj Quantum Materials 4, 54 (2019); MIT licence; PyPI `chinook`
1.1.3; last commit February 2026) for everything photoemission-specific, in the way
`phonon/do_phonopy.py` uses phonopy: PAOFLOW builds the chinook objects from its own data and
drives them, chinook stays an optional dependency.

## What chinook does and where PAOFLOW plugs in

chinook is a pure-Python (numpy, scipy, matplotlib) tight-binding + ARPES code, 10.7k lines:

| chinook object | role | PAOFLOW injection |
|---|---|---|
| `orbital(atom, index, label 'nlXX', pos [Å], Z, spin, lam, sigma)` (`orbital.py`) | basis state: real cubic harmonic `lXX` of principal number `n` on an atom; `projdict` maps the labels to $Y_l^m$ | one orbital per PAO basis function: atom, shell ($n$, $l$) from `arry['shells']`/the basis files, $m$ in QE's `ylmr2` order ($z, x, y$; $z^2, xz, yz, x^2-y^2, xy$), position `tau`, atomic number from the species |
| `TB_model(basis, {'type': 'list', 'list': [[i, j, Rx, Ry, Rz, t_ij]], 'avec', 'cutoff', ...})` (`TB_lib.py`) | Hamiltonian as hopping list; $H_{ij}(\mathbf k) = \sum_n t^n_{ij} e^{i\mathbf k\cdot\mathbf R^n_{ij}}$ with $\mathbf R_{ij}$ the **full connecting vector in Å** (orbital positions included: atomic gauge, cf. `H_library.sk_build`), $\mathbf k$ in Å⁻¹, upper triangle $i \le j$ | `HRs` (nawf, nawf, R) with PAOFLOW's $H(\mathbf k) = \sum_R H(R) e^{-2\pi i \mathbf k\cdot R}$: hopping $t_{ij}(\mathbf R_{ij} = \boldsymbol\tau_j - \boldsymbol\tau_i - R\,a)$ $= H_{ij}(R)$, i.e. the sign flip already handled by `models/band_unfold.paoflow_to_model_dict` (`fourier_sign = -1`); drop $|t| <$ tol |
| `ARPES_lib.experiment(TB, ARPES_dict)`: `datacube()` diagonalises on the (kx, ky) cube at $k_z(h\nu, V_0, W)$, computes the dipole matrix elements $M_{n\mathbf k}$ (plane-wave final state, radial integrals $B^{l'}_{nl}(k)$, Gaunt coefficients, polarisation, spin), `spectral()` assembles $I(\mathbf k,\omega) = \sum_n |M_{n\mathbf k}|^2 A_{n\mathbf k}(\omega) f(\omega)$ and convolves with the resolution | the experiment | keep as is; `experiment.spectral` only supports a self-energy that is **k-dependent but band-independent** (`SE_gen`: 'constant', 'poly', 'grid', 'func'(kx, ky, w)); the state-resolved $\Sigma_{n\mathbf k}(\omega)$ needs a small override (below) |
| `radint_lib` radial integrals: 'slater', 'hydrogenic', '**grid**' (user radial functions $R_{nl}(r)$ on a grid per 'atom-n-l'), 'exec', 'fixed' | $B^{l'}_{nl}(k) = i^{l'}\int dr\, R_{nl}(r) r^3 j_{l'}(kr)$ | the PAO radial functions themselves: `BASIS/<El>/<nl>.dat` tables (`r`, $R(r)$) or the UPF `PP_PSWFC` read by `inputs/read_upf.py` → `rad_type='grid'`, so the matrix elements use the same orbitals as the Hamiltonian |
| `slab.py` (`gen_TB(..., slab_dict={'miller', 'thick', 'vac', 'termination'})`), `mfp` depth attenuation | surface termination and escape depth | works directly on the injected bulk model |
| spin: `basis` doubled (`spin_double`), SOC as on-site $\lambda\,\mathbf L\cdot\mathbf S$ (`H_library.SO`), spin-ARPES projectors | | PAOFLOW's FR Hamiltonian lives in the $|l\,j\,m_j\rangle$ spinor PAO basis; rotate to $|l\,m\rangle\otimes|\sigma\rangle$ with `projection/do_atwfc_proj.clebsch_jm_matrix` and pass the full spinor hopping list with `lam = 0` (SOC already inside); scalar-relativistic runs use `spin: {'bool': False}` |
| `klib.kpath`, `TB_model.solve_H`, `dos.py`, `FS_tetra.py`, `matplotlib_plotter`, `intensity_map` | bands, DOS, Fermi surfaces, interactive explorer, map I/O | free with the injected model |

Caveats found in the code: chinook's dependency pins (`numpy<2`, `matplotlib<3.9` in
`requirements.txt`) are incompatible with the `work` environment; the code uses nothing
numpy-2-specific at a first grep, so the plan is to pin `chinook>=1.1.3` in an `arpes` extra
and verify under numpy 2 (patch upstream if needed, the code base is small). Matrix elements
are evaluated state by state in Python (`M_compute`); a 200×100 cube with 20 bands is
4·10⁵ states, minutes, acceptable. The final state is a free-electron plane wave (no
scattering phase shifts unless given as `phase_shifts`), the standard chinook limitation.

## Self-energies that PAOFLOW can supply

1. **Isotropic Eliashberg self-energy** from $\alpha^2F(\Omega)$ of `eliashberg_dense_q`
   (Engelsberg–Schrieffer / Grimvall, constant-DOS form):
   $\Sigma(\omega, T) = \int_0^\infty d\Omega\,\alpha^2F(\Omega)\Big[-2\pi i\,(n_\Omega + \tfrac12)
   + \psi\big(\tfrac12 + i\tfrac{\Omega-\omega}{2\pi T}\big) - \psi\big(\tfrac12 - i\tfrac{\Omega+\omega}{2\pi T}\big)\Big]$
   (digamma $\psi$; $\mathrm{Re}\,\Sigma'(0) = -\lambda$ at $T = 0$, $\mathrm{Im}\,\Sigma(\omega) = -\pi\int d\Omega\,\alpha^2F(\Omega)[2 n_\Omega + f_{\Omega-\omega} + f_{\Omega+\omega}]$).
   Cheap, k-independent, the kink of metals (Pb, MgB₂ σ bands via the anisotropic
   $\lambda_{n\mathbf k}$ of `fermi_surface_coupling`: $\alpha^2F_{n\mathbf k}$ per Fermi-surface
   state is already accumulated there). Maps onto chinook's `'grid'` self-energy directly.
2. **State-resolved Fan–Migdal self-energy** on arbitrary k (the ARPES cube, a k-path):
   $\Sigma_{n\mathbf k}(\omega,T) = \sum_{m\nu}\int\frac{d^3q}{\Omega_{BZ}}\,|g_{mn\nu}(\mathbf k,\mathbf q)|^2
   \Big[\frac{n_{\mathbf q\nu} + f_{m\mathbf k+\mathbf q}}{\omega - \varepsilon_{m\mathbf k+\mathbf q} + \omega_{\mathbf q\nu} + i\eta}
   + \frac{n_{\mathbf q\nu} + 1 - f_{m\mathbf k+\mathbf q}}{\omega - \varepsilon_{m\mathbf k+\mathbf q} - \omega_{\mathbf q\nu} + i\eta}\Big]$,
   EPW's `elecselfen`/`specfun_el` (no Debye–Waller term, as EPW). Everything exists in
   `elphon/`: `prepare_dense_vertex` → $g(R_e, R_p)$, `g_Re_at_q`, `band_vertex` at any
   (k, k+q) given eigenvectors, phonons from `phonon_at_q`, the long-range model, the
   Bose/Fermi factors and the delta kernel of `carrier_scattering.scattering_rate_kernel`
   (its on-shell imaginary part is exactly $-\mathrm{Im}\,\Sigma_{n\mathbf k}(\varepsilon_{n\mathbf k})$,
   which gives the consistency test). What is missing is a frequency-resolved kernel with
   complex denominators and the evaluation at k off the dense grid: `precompute_dense_electrons`
   keeps `Hn_flat`/`NintH` in its velocity terms but diagonalises on the grid only; a
   `bloch_states_at(electrons, k_cryst)` helper (eigh of $H(\mathbf k)$, phases
   $e^{2\pi i\mathbf k\cdot n}$ for `vertex_phases`) makes k and k+q free, with q on the dense
   grid and the irreducible-wedge folding of `irreducible_mesh` for the q sum. Cost: $N_k^{\rm path}
   \times N_q$ diagonalisations of nawf×nawf and one `band_vertex` per (k, q): 200 k × 24³ q
   is minutes on 8 ranks. Adaptive smearing (`adaptive_smearing.py`) applies to $\eta$.
3. **Spectral function** $A_{n\mathbf k}(\omega) = -\frac1\pi\,\mathrm{Im}\,[\omega - \varepsilon_{n\mathbf k} - \Sigma_{n\mathbf k}(\omega) + i0^+]^{-1}$,
   quasiparticle strength $Z_{n\mathbf k}$, the renormalised dispersion and the mass
   enhancement $1 + \lambda_{n\mathbf k}$; temperature series (Debye-Waller-free thermal
   broadening and the T-dependent kink).

## Architecture

New subpackage `src/PAOFLOW/arpes/` (no `do_*` names), optional extra `arpes = ["chinook"]`:

| module | content |
|---|---|
| `chinook_model.py` | `chinook_basis(data_controller)` → list of `orbital` (labels from `shells` + QE `ylmr2` order, positions in Å, $Z$, spin), `chinook_hoppings(data_controller, tol)` → hopping list in the atomic gauge, `chinook_model(data_controller, *, tol=1e-6, spin='auto', slab=None)` → `TB_model`; the FR spinor rotation (`clebsch_jm_matrix`); `assert_bands_match(model, data_controller, kpath)` (round trip: chinook eigenvalues == PAOFLOW's at 1e-8 eV) |
| `radial.py` | `pao_radial_functions(data_controller)` → chinook `'grid'` `rad_args` from the basis tables / UPF pseudo-wavefunctions (`r` to the origin, one entry per 'atom-n-l') |
| `experiment.py` | `PAOFLOWExperiment(experiment)`: thin subclass with `spectral_from_self_energy(sigma)` that takes $\Sigma_{n\mathbf k}(\omega)$ per state of `pks` (shape `(npks, nω)`) or per k, reusing `Mk`, the Fermi factor, the resolution convolution and the `intensity_map` output; `states(experiment)` → the (k, band, ε) list of the cube (for the self-energy evaluation) |
| `elphon/self_energy.py` | `eliashberg_self_energy(omega, a2f, w, T)` (digamma form, numerical check), `fan_migdal_self_energy(coupling, electrons, k_cryst, w, temps, eta=..., adaptive_smearing=None, ...)` on arbitrary k from the dense vertex (driver analogous to `carrier_scattering_dense_q`: shared vertex window, q loop distributed over ranks, irreducible q), `spectral_function(e, sigma, w)`, `quasiparticle_renormalization(sigma, w)` |
| `PAOFLOW.py` | `arpes_model(tol=...)` (stores nothing; returns the chinook model), `electron_self_energy(kpath | cube, ...)`, `arpes_spectrum(arpes_dict, self_energy=...)` returning the `intensity_map`; plots in `GPAO` (band map, EDC/MDC, kink) |

Data flow (metal with a kink): `PAOFLOW` → `pao_hamiltonian` → `arpes_model()` → chinook
`experiment(model, ARPES_dict)` → `datacube()` (matrix elements) → `states()` gives the
(k, band) list → `fan_migdal_self_energy(..., k_cryst=states.k, w)` with the vertex of
`prepare_dense_vertex` (cached by `coupling_cache`) → `spectral_from_self_energy` → map.
For the isotropic route the self-energy is `('grid', w, Σ(w))` and chinook's own `spectral`
is enough. Slabs: no slab self-energy from the bulk dense-q machinery; use the bulk
$\Sigma_{n\mathbf k}$ at the slab state's projected bulk $k_z$ (a "bulk self-energy" approximation,
documented), or the isotropic $\Sigma(\omega)$.

Grid consistency: the ARPES cube k-points are arbitrary (no commensurability with the dense
q-grid needed, k+q is diagonalised); the kz of each cube point comes from chinook
(`klib.kmesh(ang, X, Y, kz, Vo, hv, W)`), and the self-energy is evaluated at the 3D point.

## Validation

1. Round trip: chinook model from PAOFLOW reproduces the PAOFLOW bands on a path (Si, Pb FR
   with the spinor rotation) to 1e-8 eV; chinook `dos_broad` vs PAOFLOW `dos`.
2. Isotropic Σ: slope $-\lambda$ at $T \to 0$ against `eliashberg_dense_q`'s λ (Pb: 1.07 adaptive /
   1.18 fixed, this run); Im Σ(ω → ∞) = $-\pi\lambda\langle\omega\rangle$ sum rule.
3. State-resolved Σ: $-\mathrm{Im}\,\Sigma_{n\mathbf k}(\varepsilon_{n\mathbf k}, T)$ equals
   $\hbar/2\tau_{n\mathbf k}$ of `carrier_scattering_dense_q` on grid k (same vertex, same
   smearing: identity to 1e-10); Kramers–Kronig between Re and Im; EPW `specfun_el` parity on
   the Si and Pb data sets of the DFPT port (same nscf, same coarse grids).
4. Physics: Pb kink at ω ≈ 5–10 meV with $1 + \lambda$ mass enhancement (Reinert et al.
   2003 ARPES on Pb(111)); MgB₂ σ-band kink at the $E_{2g}$ energy (~70 meV) with the anisotropic
   coupling (Uchiyama et al. 2002); Si valence band thermal broadening and the 300 K band
   shift against the zero-point-free dispersion (temperature series of the examples).
5. Matrix elements: chinook's own tutorial (graphene/Sr₂IrO₄) rebuilt from PAOFLOW's model
   with Slater radial functions vs PAO radial functions: polarisation dependence changes,
   the dispersion does not.

## Phases and effort

- **Phase 0** (1–2 days): `arpes` extra, `chinook_model.py`, `radial.py`, round-trip tests,
  `PAOFLOW.arpes_model()`; check chinook under numpy 2 (fork/pin if needed).
- **Phase 1** (3–4 days): `elphon/self_energy.py`: isotropic form (digamma + numerical
  quadrature cross-check), `bloch_states_at`, frequency-resolved Fan–Migdal kernel (numpy;
  Rust later if profiling asks), arbitrary-k driver with MPI over q, tests 2–3, EPW parity.
- **Phase 2** (2–3 days): `arpes/experiment.py` (state-resolved injection, kz handling,
  slabs with the bulk-Σ approximation), `PAOFLOW.electron_self_energy` /
  `arpes_spectrum`, `GPAO` plots, docs section in `Electron-Phonon-Coupling.md`
  ("Spectral functions and ARPES") and a `docs/user_guide` page.
- **Phase 3** (2 days + runs): examples `examples/elphon_example/{Pb,MgB2,Si}/arpes.py` with
  REPORTs (fedora for the Pb FR dense grids), unit tests on synthetic models, CI job with the
  elphon asset (Si).

Decisions proposed: chinook kept external (optional extra, not vendored) unless the numpy-2
pin forces a fork; the self-energy lives in `elphon/` (physics), the chinook glue in
`arpes/` (I/O), mirroring `phonon/do_phonopy.py`; no Debye–Waller term (EPW parity first);
the FR spinor rotation is required for Pb-type cases (chinook's own $\lambda\,\mathbf L\cdot\mathbf S$
is not used); the isotropic route ships first because it needs no new dense-q code.
