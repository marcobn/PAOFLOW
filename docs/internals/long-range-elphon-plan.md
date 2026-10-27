# Long-range (Fröhlich and quadrupole) electron-phonon coupling: plan

Status (Oct 25 2026): **implemented** (module `elphon/long_range.py`, hooks below, unit tests
`tests/unit/elphon/test_long_range.py`); the SiC parity with EPW and the integration asset are in
progress (see "Validation" at the end). Without a model the behaviour is unchanged:
`check_long_range_terms` (`elphon/do_pao_eph.py`) refuses polar materials (|Z*| > 0.1) unless
`allow_missing_long_range=True` and warns for non-polar insulators (quadrupole term missing). With
`long_range=LongRangeModel(...)` the drivers subtract and restore the model; the Born charges and ε∞
come from either coupling source (`load_epw_coupling`, `CoarseVertexSetup.long_range_tensors()`),
and `LongRangeModel.from_coupling(coupling_dir, save_dir, quadrupoles=...)` builds the model from
the `ph.x` outputs and the nscf save.

## Why the interpolation needs it

The coupling of a polar crystal has a part that diverges as $1/|q|$ (Fröhlich), and the dynamical
matrix has a non-analytic dipole-dipole part (LO-TO splitting). Neither is a smooth function of q, so
the Fourier interpolation from the coarse grid to the dense one cannot represent them: the dense
$g(k,q)$ near Γ is wrong by the whole long-range part and the LO branch collapses onto the TO. The
standard cure (Verdi and Giustino, PRL 115, 176401 (2015); Sjakste et al., PRB 92, 054307 (2015);
EPW `lpolar`) is subtract-and-restore: remove an analytic long-range model on the coarse grid, interpolate
the short-range remainder, add the model back at every dense q. The quadrupole extension (Brunin et
al., PRL 125, 136601 (2020); Poncé et al., PRR 3, 043022 (2021)) is the next order in q and is what
non-polar semiconductors such as Si need for the small-q behaviour.

## EPW's implementation (QE 7.5 / EPW 6, `EPW/src/longrange.f90`)

Vertex, `rgd_blk_epw` (3D branch), in the Wannier gauge, Rydberg units ($e^2 = 2$):

$$
g^{\mathcal L}_{mn,\kappa\alpha}(k,q) = i\,\frac{4\pi e^2}{\Omega}\sum_{G\ne -q}
\frac{e^{-(q+G)^2/4\alpha}}{(q+G)\cdot\varepsilon^\infty\cdot(q+G)}\,
e^{-i(q+G)\cdot\tau_\kappa}\Big[(q+G)\cdot Z^*_\kappa\Big]_\alpha\,\delta_{mn}
\;+\;\text{quadrupole: } -\tfrac{i}{2}(q+G)\cdot Q^\kappa_\alpha\cdot(q+G)
$$

- `alph = 1.0` in units of $(2\pi/a)^2$ (`metric`), `gmax = 14`: the G box is the set with
  $(q+G)^2/4\alpha < 14$ (relative damping $e^{-14}$), built around the smallest $|q+G|$ (`find_min_qG`).
- The overlap $\langle u_{m,k+q}|u_{n,k}\rangle$ is taken as $[U_{k+q}U_k^\dagger]_{mn}$, which is the
  identity in the Wannier gauge: EPW applies `coeff_delta(imode)` times $\delta_{mn}$ (no Wannier-centre
  phase). A position-operator correction `coeff_r` (from the Wannier position matrix `crrw`/`rrf`) is
  used only in the 2D quadrupole branch.
- `zeu(ipol, i, na)`: first index field, second displacement, as PAOFLOW's `born_charges (nat, 3, 3)`.
- Subtracted once per coarse (k, q) in the Wannier gauge, after the rotation by the U matrices and
  before the $k \to R_e$ transform (`bloch2wannier.f90`); added back at every fine q in the Wannier gauge
  after the $R_e \to k'$ transform and before the rotation to the fine Bloch states (`wannier2bloch.f90`).
- Phonons, `rgd_blk`: the dipole-dipole term of the dynamical matrix with the same Ewald sum
  (dipole, dipole-quadrupole and quadrupole-quadrupole pieces), including the q = 0 self term subtracted
  from the diagonal (the ASR of the long-range part); subtracted from the coarse $D(q)$ before
  $q \to R_p$, added at the fine q. `matdyn.x` does the same (`PHonon/PH/rigid.f90`).

## Mapping onto PAOFLOW

The PAO gauge $g^{\rm PAO}_{ij}(k,q) = \sum_{mn} A_{mi}(k+q)\,d_{mn}(k,q)\,A^*_{nj}(k)$ (cell-phase
Bloch sums of atom-centred orbitals) plays the role of the Wannier gauge, so the long-range vertex is
**k-independent and diagonal in the orbitals**: $g^{\mathcal L}_{ij,\kappa\alpha}(q) = \delta_{ij}\,
c_{\kappa\alpha}(q)$, one G-sum per q. Optional refinement: keep the orbital-centre phase
$e^{i(q+G)\cdot\tau_i}$ (exact for point orbitals, `convention='centres'`); the default mirrors EPW
(`convention='epw'`) because the convention must be the same on both grids and parity needs EPW's.
The quadrupole term enters the same coefficient. Metals keep $Z^* = 0$ and are unaffected; free-carrier
screening of the Fröhlich term in doped semiconductors (EPW `lscreen`) is out of scope.

### New module `elphon/long_range.py` (~400 lines)

| Function | Role |
|---|---|
| `LongRangeTensors` | ε∞ (3,3), Z* (nat,3,3), optional Q (nat,3,3,3), τ (alat), alat, Ω, at/bg; `from_dfpt(setup)`, `from_epb(epw)`, `read_quadrupole_fmt(path)` (EPW's `quadrupole.fmt`) |
| `ewald_g_box(q_cart, bg, alpha=1.0, gmax=14.0)` | the (q+G) list of EPW's box, excluding q+G = 0 |
| `long_range_mode_coefficients(q_cryst, tensors, alpha, gmax, quadrupole)` | $c_{\kappa\alpha}(q)$, complex (3 nat,), Ry/bohr, Cartesian displacement index |
| `long_range_vertex_pao(q_cryst, tensors, nawf, orbital_positions=None, convention='epw')` | $(nawf, nawf, 3nat)$ to subtract/add |
| `long_range_dynamical_matrix(q_cryst, tensors, alpha, gmax)` | `rgd_blk` incl. the self term, $(3nat, 3nat)$, Ry mass-weighted after division by $\sqrt{M_\kappa M_{\kappa'}}$ |

### Hooks (numerics unchanged when `long_range=None`)

1. **Coarse subtraction** in `prepare_dense_vertex` (`do_pao_eph_dense_q.py`): the per-q PAO-gauge
   vertex $g_q(k)$ $(nk, nawf, nawf, 3nat)$ exists for both sources just before `pao_vertex_to_cells`
   (`'dfpt'`: `pao_vertex_k` / `unfolded_pao_vertices` members; `'epw'`: inside `vertex_from_epw` before
   `vertex_pao_R`, which needs a `long_range` argument). Subtract $g^{\mathcal L}(q)$ from every k. With
   `unfold='irreducible'` subtract at each star member after the rotation (the model is evaluated at the
   member's q directly, so covariance is automatic).
2. **Dense restoration** in `band_vertex` (`elph_bloch.py`): new argument `long_range` $(nawf, nawf,
   3nat)$ added to `gk` after the Wigner-Seitz phases and before the mode and band rotations. Three
   callers compute it once per dense q: `lambda_q_dense_ws_fast` (Eliashberg, transport, pair coupling),
   `carrier_scattering_dense_q` (mobility/SERTA/MRTA), `phonon_assisted_absorption_dense_q`.
3. **Phonons**: `phonon_interp_from_dyn` (dense drivers) and `phonon_interpolator`
   (`phonon/qe_force_constants.py`, `q2r_matdyn`) take `long_range=tensors`; subtract $D^{\mathcal L}(q)$
   from each coarse $D(q)$ before $q \to R_p$, add it in `phonon_at_q` before diagonalising. The
   `is_polar` warning of `q2r_matdyn` goes away. At exactly Γ the non-analytic limit is
   direction-dependent; dense grids only hit Γ itself (zeroed anyway) and paths may pass a direction.
4. **API**: drivers gain `long_range='dipole' | 'dipole+quadrupole' | None` (default `'dipole'` when
   $|Z^*| > 0.1$, else None), `quadrupoles=` (array or `quadrupole.fmt` path), `ewald_alpha=1.0`;
   `check_long_range_terms` becomes the dispatcher (raise only when the tensors are missing, i.e. `ph.x`
   without `epsil`). The vertex cache key gets the long-range settings.
5. **Cost**: one Ewald sum per q (~$10^3$ G vectors, microseconds) on both grids; the subtraction is a
   broadcast over k. Memory unchanged.

## Tests

Unit (synthetic, fast, `tests/unit/elphon/test_long_range.py`):

1. **Ewald box**: the sum is converged (α = 1 vs α = 0.5 agree to $10^{-8}$ after the complementary
   real-space part is absent, i.e. the G box alone is converged at gmax = 14).
2. **Subtract-then-add identity** on a coarse grid for a random vertex: `prepare_dense_vertex` with
   `long_range` followed by `band_vertex` at the coarse q returns the original $g$ to $10^{-12}$ (both
   conventions).
3. **Fröhlich limit** of a cubic diatomic rigid-ion model (scalar $Z^*$, ε∞, masses): along $q \to 0$,
   $|g^{\mathcal L}_{\rm LO}(q)| \to \frac{4\pi e^2}{\Omega}\frac{Z^*}{\varepsilon^\infty |q|}
   \frac{1}{\sqrt{2\mu\,\omega_{\rm LO}}}$ (with the mode factor), the TO modes get zero, and the LO-TO
   splitting of `long_range_dynamical_matrix` equals $\omega_{\rm LO}^2-\omega_{\rm TO}^2 =
   4\pi e^2 Z^{*2}/(\Omega\mu\varepsilon^\infty)$ (Lyddane-Sachs-Teller for the rigid-ion model).
4. **Symmetry covariance**: $g^{\mathcal L}(Sq)$ equals `rotate_pao_vertex` applied to $g^{\mathcal L}(q)$
   (consistency with the irreducible-q unfolding), and $D^{\mathcal L}$ is Hermitian with the self term
   satisfying the ASR at q → 0.
5. **Quadrupole term**: with $Z^* = 0$ and a model $Q$, $|g| \propto |q|^0$ at small q and the
   dipole-quadrupole cross term of $D^{\mathcal L}$ matches a finite-difference derivative of the dipole term.

Parity with EPW (local scripts, then an integration asset):

6. **3C-SiC**, the EPW test case `test-suite/epw_polar` (2 atoms, 20 Ry, scf 6³, `ph.x` 3³ with
   `epsil`, nscf 3³ with 8 bands, `lpolar`): run EPW with `prtgkk = .true.` on a short (k, q) list
   (`filkf`/`filqf`) through Γ, and compare $\sum_{\rm degenerate}|g_{mn\nu}(k,q)|^2$ with PAOFLOW's
   `band_vertex` on the same list from the minimal PAO basis (Si 3s3p, C 2s2p, 8 orbitals: the 4 valence
   bands are reproduced exactly). Expect agreement to the interpolation accuracy of the two bases
   (≈ 1 % away from the coarse q) and to $10^{-6}$ at coarse q; the $1/|q|$ behaviour near Γ is the
   decisive check. Asset ≈ 10 MB (`sic/` next to `si/` in the elphon tarball), CI test
   `test_long_range_parity.py`.
7. **Phonons**: LO-TO splitting of SiC at Γ-approach and the whole dispersion against `matdyn.x` with
   the same Z*/ε∞ (`q2r_matdyn` path), to $10^{-3}$ THz.
8. **GaN mobility** (`test-suite/epw_mob_polar`, LDA, 2³/2³ coarse): room-temperature electron mobility
   of the PAOFLOW carrier driver vs EPW's `epw2.in`/`epw3.in` output; literature values (Poncé et al.,
   PRB 100, 085204 (2019)) as the physical reference. Not a CI test (minutes to hours).
9. **Si with quadrupoles** (`test-suite/epw_qdpt`, `quadrupole.fmt` from Brunin et al.): mobility and the
   phonon-assisted absorption threshold change by a few per cent; compare with the Si example numbers.

## Order of work

1. Module + unit tests 1-5 (2 days). 2. Hooks 1-3 and the dispatcher (2 days). 3. SiC parity and the
asset (2 days, needs the EPW build already used for the DFPT port). 4. Quadrupole input and the Si check
(1 day). 5. Docs: this note, the e-ph guide's "Long-range" bullet, the Si example README (quadrupoles),
`check_long_range_terms` docstring.

## Validation (Oct 25-26 2026)

Local scripts in `Workspace/PAOFLOW/dfpt_port/sic{,6,40}/` (QE 7.5 / EPW 6.0, PAOFLOW PBE pseudopotentials,
3C-SiC at a = 8.237 bohr: ε∞ = 7.51, Z* = +2.65 (Si) / −2.83 (C)).

- EPW's `prtgkk` output needs a wide `fsthick` (with 10 eV the deep valence band is dropped from the
  printed block and the sums are not comparable); `epw2_fs100.out` is the reference.
- At a coarse point (k = Γ, q = (1/3, 1/3, 0)): the DFPT band vertex of the 8-band and 4-band nscf runs
  and EPW's `.epb` agree to $4\times10^{-15}$; PAOFLOW's dense path with the model reproduces them to
  0.4 % (standard basis, 26 orbitals; 6 % with the minimal basis: projectability), EPW's fine-grid
  values too.
- Away from the coarse points on the 3³/3³ grids the two interpolations differ by up to 20 % for the
  LO mode near Γ and up to 60 % for the weaker modes: the short-range remainder is interpolated with
  different localised bases (four sp³ Wannier functions vs the PAO set), and the PAO long-range term
  carries the physical valence-subspace overlap $\sum_{mn}|\langle u_{k+q}|u_k\rangle|^2/4$ (0.93 at
  $q = 1/12$, 0.42 at $q = 1/3$ along (1,1,0)) where the four-band Wannier model uses the identity. Both
  are consistent subtract-and-restore schemes; the convergence with the coarse grid (6³/6³ run) is the
  meaningful comparison, see below.
- **Convergence with the coarse grid (6³ k / 6³ q, same fine points, standard basis):** PAOFLOW and
  EPW agree to 1-5 % at every (k, q, mode group), worst 12 % (the LA mode at |q| = 1/12 along (1,0,0));
  the LO mode near Γ, dominated by the Fröhlich term, agrees to 2-5 %. The two subtract-and-restore
  schemes therefore converge to the same answer, and the residual on 3³ grids is the short-range
  interpolation, not the long-range model.
- Integration test `tests/integration/elphon/test_long_range_sic.py` (3³ asset at 40 Ry): coarse-point
  identity < 2 %, LO coupling grows by > 2.5× when |q| halves only with the model and exceeds the plain
  interpolation by > 3× near Γ, EPW agreement < 2 % at coarse points and < 70 % elsewhere, LO-TO
  splitting within 3 % of the rigid-ion value from ε∞ and Z*.
- `q2r_matdyn` (Oct 26 2026): `long_range='auto'` builds the model from the dyn files
  (`LongRangeModel.from_dynamical_matrices`, `zasr='simple'`, QE's Ewald parameter
  $\alpha = (a/2\pi)^2$ in $(2\pi/a)^2$ units, which `rigid.f90` uses with G in $2\pi/a$; EPW
  uses 1), subtracts its force constants before the q → R transform, writes the `T` flag with
  α, ε∞ and Z* into the `flfrc` file, restores the term at every q and takes the
  direction-dependent limit at Γ along a path (`gamma_limit_points`, as `matdyn.x`). SiC 3³:
  `matdyn.x` reproduced to 0.01 cm⁻¹ (optical branches 1e-4), LO-TO at Γ 790/952 cm⁻¹
  (`tests/unit/phonon/test_qe_force_constants.py`, data `tests/unit/phonon/data/sic_dyn`).
- **Quadrupoles from finite-q DFPT** (`elphon/quadrupoles.py`, Oct 26 2026): instead of ABINIT's
  long-wave DFPT, the multipole expansion of the induced electron density of single-q `ph.x` runs
  (`fildrho`): $\varepsilon_L(\hat q)\,[-\Omega\langle\tilde n_{\kappa\beta}\rangle e^{iq\cdot\tau_\kappa}
  - iq_\beta Z^{\rm ion}_\kappa] = -iq\cdot Z^* - \tfrac12 q\cdot Q^\kappa_\beta\cdot q + O(q^3)$.
  The screening of a finite-q DFPT run acts on the total (ion + electron) charge, so the point ion
  that moves rigidly with the atom is added before dividing out $\varepsilon_L$; the imaginary part
  then reproduces the Born charges of the Γ run (and fixes the sign convention of the file), the
  real part at $q^2$ is the quadrupole, free of the $q^2$ dependence of $\varepsilon_L$ and of the
  ionic term (imaginary, orders $q^3$ and $q$). Six directions × two magnitudes
  (`single_q_phonon_inputs`), least squares on the symmetric quadratic forms, Richardson
  extrapolation in $|q|$, `write_quadrupole_fmt` for EPW's format and `LongRangeModel(quadrupoles=...)`.
  Driver: `coupling/quadrupoles.py` of every example (`PAOFLOW.elphon.workflows.quadrupoles`).
  Validation on Si (`Workspace/PAOFLOW/quadrupoles/si`: scf 12³ k, 38 Ry, PBE `Si.upf`, Γ run
  with `epsil` ε∞ = 13.11, twelve single-q runs with |q| = 0.02 and 0.04 (2π/a), 45 s each):
  the only non-zero components are $Q_{x,yz}$ and its permutations, $Q^{\rm Si_1}_{x,yz} = -14.37$
  (|q| = 0.02), $-14.21$ (0.04), extrapolated $-14.42$ e bohr, atom 2 opposite in sign, against
  13.66 e bohr of Brunin et al., PRL 125, 136601 (2020) (ABINIT long-wave DFPT, different
  pseudopotential and cutoff); the fitted Born charges agree with the Γ run to 0.03 e (Z* ≈ 0 for
  Si, the residual is the $q^3$ term of the $|q| = 0.04$ samples amplified by ε∞). A charge-density
  file convention would have given $Z^*_{\rm el} - Z_{\rm ion}$ with the wrong sign and is
  detected by `calibrate_sign`. On the polar 3C-SiC (`examples/elphon/SiC/coupling/quadrupoles.py`,
  6³ k, 60 Ry, PBE, |q| = 0.02 and 0.04, 10 s per run): $Q^{\rm Si}_{x,yz} = +7.49$ and
  $Q^{\rm C}_{x,yz} = -3.01$ e bohr (Si at the origin), against 7.41 and −2.63 fitted from small-q
  DFPT by Poncé et al., Phys. Rev. Research 3, 043022 (2021), Table IV (Abinit LDA: 7.03 / −1.42);
  the fitted Born charges reproduce the Γ-point $Z^* = +2.65 / -2.83$ to 0.05 e, which checks the
  screening bookkeeping and the relative sign of the dipole and quadrupole terms (EPW's
  `coeff_delta = facq (i q·Z* + ½ q·Q·q)`). Not done: convergence of Q with the k grid and cutoff.
