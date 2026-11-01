# Plan: the special displacement method (Zacharias–Giustino) in PAOFLOW

Status: plan, 1 November 2026. Nothing implemented yet. Written while the fixed-basis finite-difference
vertex is being validated against the DFPT vertex on Pb (`Workspace/PAOFLOW/validation/Pb_fd_fixed_basis`);
the two routes are complementary and this plan assumes that validation lands.

## 1. What the method gives, and what it does not

The Williams–Lax (WL) theory writes a temperature-dependent electronic or optical observable as the thermal
average of its clamped-ion value over the nuclear configurations of the harmonic crystal,

$$\langle O\rangle_T=\prod_{\mathbf q\nu}\int du_{\mathbf q\nu}\,\frac{e^{-u_{\mathbf q\nu}^2/2\sigma_{\mathbf q\nu}^2}}{\sqrt{2\pi}\,\sigma_{\mathbf q\nu}}\;O(\{u\}),\qquad
\sigma_{\mathbf q\nu}^2=\frac{\hbar}{2M_0\omega_{\mathbf q\nu}}\,(2n_{\mathbf q\nu}+1),$$

with zero-point motion through the $+1$. The special displacement method (SDM; Zacharias and Giustino, PRB 94,
075125 (2016); PRR 2, 013357 (2020)) replaces the multidimensional average by **one** supercell configuration,

$$\Delta\tau^{\rm ZG}_{p\kappa}=\frac{2}{\sqrt{N_p}}\sqrt{\frac{M_0}{M_\kappa}}\sum_{\mathbf q\in\mathcal B,\nu}S_{\mathbf q\nu}\,\sigma_{\mathbf q\nu}\,
\mathrm{Re}\!\left[e^{i\mathbf q\cdot\mathbf R_p}\,e_{\kappa\nu}(\mathbf q)\right],\qquad S_{\mathbf q\nu}=\pm1,$$

where the sum runs over one member of every $\pm\mathbf q$ pair ($\mathcal B$) and the modes at the
time-reversal-invariant points ($\mathcal A$, counted once), and the sign sequence $S$ is chosen so that the
cross terms of the second-order expansion of $O(\{u\})$ cancel in the large-supercell limit. One DFT
calculation of the ZG supercell then yields $\langle O\rangle_T$ to second order in the displacements exactly and,
in practice, to all orders. Per temperature: one ZG configuration, one scf.

What it delivers in PAOFLOW terms (fluctuation sector of the e-ph problem, see `williams_lax_vs_eliashberg.md`
of 31 October): temperature-dependent band structures and gaps including the zero-point renormalisation with
the full Debye–Waller term (no rigid-ion approximation) and all static multi-phonon terms; unfolded spectral
functions $A(\mathbf k,\omega;T)$ with inhomogeneous widths; optical spectra $\varepsilon_2(\omega;T)$ including
phonon-assisted (indirect) absorption and Urbach-like tails, all from one supercell per temperature. It does not
give the response sector: no lifetimes with the phonon energy resolved, no kink or mass enhancement, no Bloch
$T^5$ transport, no Holstein optics, no pairing. Those stay with the DFPT or finite-difference **vertex** route.
The natural hybrid is SDM for the static renormalisation and the vertex route for the rest; at second order the
two sectors are additive.

So "limiting the finite-difference calculation to a single supercell" is exact for the WL observables above,
and impossible for the vertex itself: one ZG configuration is one linear combination of all mode derivatives.
(ZG.x's `single_ph_displ` option displaces along single modes for a direct supercell evaluation of the diagonal
$g(\mathbf q)+g(-\mathbf q)$ elements; that is the mode-resolved frozen-phonon route, not the SDM.)

## 2. Reference implementation at hand

`~/Programs/qe-7.5/EPW/ZG` on music is compiled: `ZG.x` (ZG configurations from a `q2r` force-constant
file, matdyn-style input: `dim1..3`, `T`/`T_array`, `synch`, `niters`, `compute_error`, `error_thresh`,
`incl_qA`, `q_external`, `single_ph_displ`, non-analytic LO–TO through matdyn's interpolation),
`bands_unfold.x` + `pp_spctrlfn.x` (spectral function unfolding), `epsilon_Gaus.x`, and the tutorial tarball
(`tutorial.tar.gz`, Si). Its README documents the sign algorithm: a "+,−,+,−" sequence over the ordered
modes and random permutations (`niters`, default 15000) until the error from the coupling of modes with the
same $\mathbf q$ at different branches falls below `error_thresh` (default 0.30 of the diagonal terms); the
same signs are kept for every temperature of `T_array`. The eigenvector phase ambiguity makes configurations
differ between runs (README point 6), so the cross-check against `ZG.x` must compare at the level of the
second-order invariants ($\langle u^2\rangle$ per atom, the thermal average of a test observable), not of the
raw coordinates, unless the same eigenvectors and signs are fed to both.

## 3. What PAOFLOW already has (reuse)

| piece | where | use in the SDM |
|---|---|---|
| phonopy driver: supercell, displacements, force constants from forces, NAC | `phonon/do_phonopy.py` (`init_phonopy`, `generate_displacements`, `produce_force_constants`, `attach_nac`) | harmonic phonons on the supercell-commensurate q-grid; LO–TO via `attach_nac` |
| DFPT force constants | `phonon/qe_force_constants.py`, `elphon/qe_matdyn.py` (q2r + matdyn in PAOFLOW) | the same from `ph.x` dyn files, matching `ZG.x`'s input |
| modes at arbitrary q with eigenvectors and masses | `elphon/gkq.py::phonon_modes` | $e_{\kappa\nu}(\mathbf q)$, $\omega_{\mathbf q\nu}$, $M_\kappa$ |
| QE input writer for displaced supercells, unique prefixes, manifest | `phonon/io.py::write_displaced_supercells`, `elphon/io.py` | write the ZG scf input(s) |
| band unfolding of a supercell PAO Hamiltonian to the primitive zone (Fourier-sign aware since 23 Oct) | `models/band_unfold.py` (`unfold_bands`, `paoflow_to_model_dict`, sparse variant) | $A(\mathbf k,\omega;T)$ and the gap renormalisation |
| dielectric tensor on any cell, with Löwdin dipoles | `response/do_epsilon.py` through `PAOFLOW.dielectric_tensor`, `elphon/workflows/optics.py::direct_dielectric` | $\varepsilon_2(\omega;T)$ of the ZG cell |
| system folders, drivers, generator | `elphon/workflows`, `gen/elphon_system.py`, `examples/elphon/<System>` | a `thermal/` workflow folder per system |
| sparse backend | `sparse/` | cells beyond ~1000 orbitals |

## 4. New code

### 4.1 `phonon/special_displacement.py` (library, procedural)

- `commensurate_qpoints(supercell_matrix)` → the q-grid of the supercell, split into the sets $\mathcal A$
  (q ≡ −q) and $\mathcal B$ (one of each ± pair), in the ordering ZG.x uses (so the sign sequences are
  comparable).
- `zg_amplitudes(frequencies_thz, masses, temperature_k)` → $\sigma_{\mathbf q\nu}$; acoustic modes at Γ
  dropped, imaginary modes raise.
- `zg_sign_sequence(frequencies, eigenvectors, ..., niters, error_thresh, rng)` → $S_{\mathbf q\nu}$: the
  alternating sequence and its permutations, the ZG error function (cross terms between branches at the same q
  relative to the diagonal terms), stop below the threshold; deterministic seed; optional `synch` (continuous
  eigenvector gauge across q by maximum overlap with the previous q, the "Berry connection" of ZG.x).
- `zg_displacements(phonon, supercell_matrix, temperature_k, ...)` → `(cell, displacements, meta)` with the
  equation above; `meta` keeps signs, amplitudes, mode list, the per-atom $\langle u^2\rangle$ sum rule and the
  error value. Several temperatures share the signs (as `T_array` with `nconfs = 1`).
- `thermal_mean_square_displacement(phonon, supercell_matrix, temperature_k)` → the analytic
  $\langle u_{\kappa\alpha}^2\rangle_T$ for the unit tests and for the user's convergence check.
- `write_zg_supercell(data_controller, cell, temperature_k, ...)` → QE scf input via `phonon/io.py`, prefix
  `<prefix>_zg_T<T>`, manifest `zg.json`.

### 4.2 User-facing method and workflow

- `PAOFLOW.special_displacement(supercell_matrix, temperatures, forces=None, ...)` next to
  `electron_phonon`: phase 1 writes the ZG inputs (needs `phonons()` or a `.fc`/dyn set); phase 2 (`forces='qe'`)
  reads the ZG saves and stores the thermal PAO Hamiltonians (`arry['zg_HRs'][T]`).
- `elphon/workflows/thermal.py` with one function per property, in the style of the other workflow modules:
  `thermal_bands` (unfolded spectral function along a path, per T), `thermal_gap` (direct and indirect gap
  renormalisation vs T, zero-point value from the T = 0 cell), `thermal_dielectric` (interband
  $\varepsilon_2(\omega;T)$ of the ZG cell, optionally with the Drude part), `thermal_absorption` (absorption
  coefficient and, through the existing slab Kirchhoff code, the emissivity). Driver template `thermal/zg.py`
  in `gen/elphon_templates`, system-folder keys `THERMAL = dict(supercell=(4,4,4), temps_k=[...], ...)`.
- `paoflow-gen elphon`: `thermal` added to the workflow menu.

### 4.3 Conventions to pin down (write them in the docstrings)

- Displacement in bohr, amplitudes from frequencies in THz and masses in amu (`gkq.py` has the constants).
- phonopy eigenvectors are mass-weighted; the atomic displacement of mode ν is $e_{\kappa\nu}/\sqrt{M_\kappa}$
  (see `phonon_modes`), which fixes the $\sqrt{M_0/M_\kappa}$ factor.
- q in the supercell's own reciprocal lattice versus the primitive one; $\mathbf R_p$ are the primitive
  translations of the sub-cells (the `fold.py` bookkeeping, `supercell_atom_translations`).
- Time reversal assumed (no SOC-driven $\pm\mathbf q$ asymmetry); the TRIM set $\mathcal A$ handled with real
  eigenvectors.

## 5. Validation (Pb is the benchmark; Si follows for the optical absorption)

Pb is kept as the benchmark system: every number below exists already for it (DFPT vertex 12³/6³, SR and FR
Eliashberg, ARPES kink, ρ(T), emissivity, dHvA), the cells are small (1 atom per primitive cell: 27, 64, 216
atoms for 3³, 4³, 6³), and the fixed-basis finite-difference test runs on the same crystal. Sizes: 3³ for the
code checks, 4³ for production, 6³ for the convergence check and for the quantitative λ (the Γ-only coupling of
a 6³ cell is the information of the 6³ DFPT grid of the production runs).

1. **Sum rule (unit test).** The ZG configuration's $\langle u_{\kappa\alpha}^2\rangle$ equals the analytic
   thermal value to the precision of the cross-term cancellation; the error function decreases with the
   supercell size (toy force constants, fcc and diamond).
2. **Against `ZG.x` on music.** Same `q2r` force constants (Pb: `ph.x` 6³ on the primitive cell, as in
   `Pb_elphon/sr/phonon6` on fedora, or regenerated in `Pb_fd_test/prim`), same supercell (3³, 4³) and T. Compare the per-atom $\langle u^2\rangle$, the error value, and, with the
   eigenvectors and signs exported from one code into the other, the coordinates themselves.
3. **Pb band renormalisation.** Temperature-dependent unfolded bands of the ZG cell (Fermi-surface shifts,
   band widths) against the real part of our Fan–Migdal fluctuation term with Bose factors; convergence 3³ →
   4³ → 6³.
4. **Pb spectral function (the limit of the method on our own data).** Unfolded $A(\mathbf k,\omega;T)$ across
   the Fermi crossing used for the kink: width equal to the Eliashberg plateau
   $\pi\!\int\!\alpha^2F(\Omega)(2n+1)\,d\Omega$ (12.6 meV at 10 K from the FR run), no kink, no
   mass enhancement.
5. **Pb ZG + finite-difference vertex (§9).** λ(T), α²F(T), ρ(T) from the vertex at the ZG geometry against
   the harmonic route with Bose factors and against several ZG sign sequences.
6. **Si phonon-assisted absorption (second system).** $\varepsilon_2(\omega;T)$ of the ZG cell against the PAOFLOW HBB kernel
   with the adiabatic switch ($\hbar\omega_\nu\to0$, weights $2n+1$): must agree away from the onset; the
   difference to the nonadiabatic kernel is the response sector (thresholds, $E_g\pm\hbar\omega_\nu$). This is the
   cross-check proposed in the WL note.
7. **Polar case (later).** GaAs or SiC: LO–TO through NAC; the supercell-size dependence ZG themselves leave
   open.

## 6. Cost

Per temperature one scf of the ZG supercell on the Γ-centred supercell grid (2³ is usually enough). Si 4×4×4 is
128 atoms with 1664 standard orbitals; Pb 4×4×4 is 64 atoms with 1152 orbitals. These are the cells where the
GPU build of pw.x pays (large FFTs, few k-points), so the music workstation is the right place. PAOFLOW:
projections and `pao_hamiltonian` on 1–2 thousand orbitals are fine in the dense code; the unfolding and the
dielectric tensor on a 2³ supercell grid are minutes. A 26-temperature series like the Si emissivity set is 26
scf runs, embarrassingly parallel.

## 7. Milestones

| step | content | estimate |
|---|---|---|
| M1 | `special_displacement.py` with the unit tests of §5.1; conventions documented | 1 day |
| M2 | `ZG.x` cross-check on Pb and Si force constants (music) | ½ day |
| M3 | `PAOFLOW.special_displacement`, `workflows/thermal.py`, driver template, generator menu | 1 day |
| M4 | Pb: ZG cells 3³/4³/6³ at 0, 100, 300 K on the music GPU; unfolded bands and spectral width vs the plateau; ZG + fixed-basis FD vertex λ(T) (§9); docs page and `examples/elphon/Pb/thermal` | 2 days of runs, 1 day of writing |
| M5 | Si: ZPR and $\varepsilon_2(\omega;T)$ vs the HBB kernel with the adiabatic switch | later |

## 8. Risks and open points

- Small supercells: the sign optimisation matters and the result depends on the mode ordering; follow ZG.x
  closely and keep `niters`/`error_thresh` as user knobs.
- Degenerate modes and the eigenvector gauge: use the `synch` idea (maximum-overlap continuation) and report
  the error value; never compare coordinates between codes without the same gauge.
- Anharmonic systems (A-SDM, Zacharias et al. PRB 108, 035155 (2023)) are out of scope for the first version.
- The ZG cell is a genuine DFT calculation: the PAO orbitals follow the atoms, and no fixed-basis question
  arises (that question belongs to finite differences of the vertex).
- Memory: supercells beyond ~2000 orbitals need the sparse backend for the Hamiltonian and the unfolding.

## 9. ZG + finite differences: the vertex of a thermal configuration (the actual target)

The intended use is not the SDM alone but the sequence: ZG configuration at temperature T → one scf → its PAO
Hamiltonian is the **reference** of the finite-difference vertex route of §1 of `dfpt-coupling-port.md` and of
the fixed-basis test now running → displaced cells around the ZG geometry → $dV/du$ of the thermally
disordered crystal. Later the same machinery takes snapshots of a molecular-dynamics trajectory as references.

Why it is consistent: the fixed-basis equivalence (finite difference = matrix of $\partial_uV$ in the reference
orbitals, on the good subspace) is a statement about any reference structure, so it holds unchanged around the
ZG cell; the orbitals are then centred at the ZG positions for all displaced cells. What is new physically is
that the vertex is evaluated in the presence of the static thermal disorder, to all orders in the ZG
displacement: the fluctuation sector enters the electrons and the coupling non-perturbatively, and the response
sector (lifetimes, kink, transport, pairing) is then built from this vertex with the phonons of the same
configuration. For alloys and disordered cells this is the only formulation available; for crystals it is the
route to temperature-dependent couplings beyond the harmonic Bose-factor dependence.

What the same displaced cells give for free: their **forces** are the finite-difference force constants at the
ZG geometry, i.e. the temperature-renormalised (anharmonic) IFCs of the A-SDM iteration (Zacharias, Volonakis,
Giustino, Even, PRB 108, 035155 (2023); `ZG.x` flag `ASDM`). One set of runs therefore yields the thermal
phonons and the thermal vertex together, and iterating ZG → forces → IFCs → ZG closes the A-SDM loop.

Pieces to add on top of §4, in order:

1. **Arbitrary reference cell.** `electron_phonon` generates displacements of the primitive cell's phonopy
   supercell; it needs a `reference_cell=` mode that takes a given structure (the ZG cell, an MD snapshot) and
   displaces every atom along x, y, z (±), with no symmetry reduction, and an analyse phase that keeps the
   supercell tensor $g(\kappa\alpha;\mathbf R_e^{\rm sc})$ without the fold (`assemble_eph_tensor(fold=False)`).
   The fixed basis is the reference's.
2. **Supercell-native Eliashberg.** The supercell is its own unit cell: coarse grid = the supercell k-grid,
   q = Γ (3N modes from the FD force constants or from the ZG phonons), dense k by PAO interpolation of the
   supercell Hamiltonian. Requires the Γ-mode fix in the Eliashberg drivers (all modes at Γ are zeroed today,
   `do_pao_eph_dense_q.py:856`, `do_pao_eph.py:643`) and `qgrid=(1,1,1)`. The minimal-image interpolation of
   q inside the supercell zone comes after, with the same `fold.py` bookkeeping applied to the supercell as
   the "primitive" cell.
3. **Cost control: random displacements and local least squares.** Around a disordered reference there is no
   symmetry, so Cartesian ± displacements cost 6N scf runs (64-atom Pb: 384 runs of ~40 min on the GPU).
   Displacing all atoms at once by small random vectors in M configurations and fitting $dH/du_{\kappa\alpha}$
   element by element with the locality of the hopping derivative (only atoms within the range of the pair
   $(i,j)$ enter) reduces this to M ≈ 50–100 runs independent of N, exactly as ALM/hiPhive extract force
   constants. Same manifest format, a new `displacement_mode='random'` and a least-squares analyse phase; the
   fit residual is the diagnostic. This is the piece that makes the MD-snapshot extension affordable.
4. **Sparse vertex storage.** $n_{\rm awf}^2\times3N\times N_R$ is 110 GB for a 4×4×4 Pb cell; the bond-list
   layout of the sparse backend is required before any cell beyond ~30 atoms.
5. **MD snapshots.** Readers for QE (and later VASP) trajectories into the `reference_cell` mode; averaging of
   the mode-resolved λ over snapshots; the ZG cell as the single representative configuration against which the
   snapshot average is checked.

Validation sequence for this combined route, all on Pb and all against numbers we already have:

- T = 0 ZG cell (zero-point displacements) + fixed-basis FD vs the harmonic DFPT vertex of the undisplaced
  crystal: the difference is the zero-point static-disorder effect on the coupling, expected at the per-cent
  level for Pb; the band-basis Fermi-window elements are the quantity to compare.
- λ(T) and α²F(T) from ZG + FD at 100 and 300 K against the harmonic route with Bose factors (the FR Pb
  campaign); then against the average over a handful of ZG configurations with different sign sequences
  (`nconfs > 1` in ZG.x terms) to test that one configuration represents the ensemble for $|g|^2$, which the ZG
  proof covers only for observables expanded to second order.
- The thermal IFCs from the forces against the harmonic ones (Pb is nearly harmonic below 300 K; the
  quasi-harmonic ρ(T) deviation above 300 K noted in the Pb report is the first target of the A-SDM loop).
