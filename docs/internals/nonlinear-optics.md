# Non-linear optics: shift and injection currents, second-harmonic generation

`PAOFLOW.nonlinear_optics(nk, ...)` computes the second-order optical responses of the PAO
Hamiltonian: the shift current (bulk photovoltaic effect), the injection currents and the
second-harmonic susceptibility, on a dense k-grid interpolated from `HRs` without storing any
`(nk, nawf, nawf)` tensor. The engine lives in
[hamiltonian/bloch_derivatives.py](../../src/PAOFLOW/hamiltonian/bloch_derivatives.py)
(Bloch Hamiltonian and its first and second k-derivatives at arbitrary k) and
[response/nonlinear_optics.py](../../src/PAOFLOW/response/nonlinear_optics.py) (band geometry,
kernels, driver, writers). It implements the formulation of Wang, Liu, Kang, Gu, Xu and Duan,
Phys. Rev. B **96**, 115147 (2017) and Chen, Ye, Zou, Gu, Xu and Duan, Phys. Rev. B **105**,
075123 (2022), with the PAO basis replacing the Wannier functions. The physics (the response
formulas, the sum rule, the shift vector, the symmetry constraints, the broadening, the units
and the approximations) is on the theory page
[Second-order optical response](../learn/theory/nonlinear-optics.md); this page covers how the
code evaluates it.

## Workflow

```python
pf = PAOFLOW.PAOFLOW(savedir='GaAs.save', ...)
pf.read_atomic_proj_QE(); pf.projectability(); pf.pao_hamiltonian()
spectra = pf.nonlinear_optics(nk=(60, 60, 60), emax=4.0, eta=0.05, scissor=0.86,
                              components=['xyz'])
pf.finish_execution()
```

`nonlinear_optics` needs `HRs`, so it runs after `pao_hamiltonian()` (and `add_external_fields`)
and before `interpolated_hamiltonian()` / `pao_eigh()`, which release it. It writes two-column
files (photon energy in eV, value): `shift_<abc>.dat` and `shiftm_<abc>.dat` (µA/V²),
`injectc_<abc>.dat` and `injectm_<abc>.dat` (A V⁻² s⁻¹), `shgr_/shgi_/shga_<abc>.dat`
(Re, Im, |χ| in pm/V). With `twoD=True` the sheet responses are written instead (multiplied by
the slab height: nm µA/V², nm A V⁻² s⁻¹, pm²/V). `GPAO.plot_optical` knows the prefixes.
Tight-binding models (`PAOFLOW(model=...)`) work the same way, with the orbital centres taken
from the model's `tau`/`naw` (or passed as `orbital_positions`).

## Conventions

Units inside the kernels: energies eV, lengths bohr, k in 1/bohr, ħ = 1 (`v` is ħ**v** in
eV bohr, `r` in bohr, `r_{;a}` in bohr²); the SI prefactors are applied once at the end.

**Bloch Hamiltonian.** PAOFLOW stores `HRs` such that $H(\mathbf k) = \sum_{\mathbf n}
e^{-i\mathbf k\cdot\mathbf R_{\mathbf n}} H(\mathbf n)$ (lattice gauge), with the pair-resolved
Wigner–Seitz images of the real-space cells of `elphon.elph_bloch`. Derivatives are explicit sums
with the factors $-iR^a_{\mathbf n}$ and $-R^a_{\mathbf n}R^b_{\mathbf n}$. With the position operator
diagonal on the orbital centres $\boldsymbol\tau_i$ (the tight-binding approximation), the
velocity and the "curvature" operators of the atomic-gauge Hamiltonian, written in the lattice gauge,
are

$$
v_a = \partial_a H + i\,\Delta\tau_a\circ H,\qquad
w_{ab} = \partial_a\partial_b H + i\,\Delta\tau_b\circ\partial_a H + i\,\Delta\tau_a\circ\partial_b H
- \Delta\tau_a\Delta\tau_b\circ H,
$$

$(\Delta\tau_a)_{ij} = (\tau_j - \tau_i)_a$. Their band matrices $U^\dagger v_a U$,
$U^\dagger w_{ab} U$ (with $U$ the eigenvectors of the lattice-gauge $H$) are those of
$\partial_a H^A$ and $\partial_a\partial_b H^A$ in the eigenbasis of the atomic-gauge Hamiltonian;
the test `test_bloch_derivatives.py` checks both against the exact operators of hand-built models and
against finite differences.

**Band geometry** (`band_geometry`, `BandGeometry`): the operators $v_a$, $w_{ab}$ are rotated to
the band basis, and the band-basis quantities of the theory page
([Band geometry](../learn/theory/nonlinear-optics.md#band-geometry-from-the-pao-hamiltonian)) are
stored per k-block: `omega` ($\omega_{nm} = E_n - E_m$), `cluster` and `pair_mask` (degenerate
groups), `inv_omega`, `velocity`, `velocity_intra`, `position` ($r^a_{nm}$), `position_delta`
(the covariant form of $r^b_{nm}\Delta^a_{nm}$) and `position_derivative` ($r^b_{nm;a}$, axes
`[a, b]`). The sum rule of
[Generalized derivative and the sum rule](../learn/theory/nonlinear-optics.md#generalized-derivative-and-the-sum-rule)
is evaluated in commutator form,
$r^b_{nm;a} = (i/\omega_{nm})\big[[v^a_{\rm intra}, v^b]_{nm}/\omega_{nm} - w^{ab}_{nm} + [v^b, G^a]_{nm}\big]$,
$G^a_{pm} = v^a_{pm}/\omega_{pm}$ between clusters and 0 inside, so the $p = n, m$ terms of the
band sum are inside the commutator (the module docstring of `response/nonlinear_optics.py` writes
the sum as $p \neq n, m$, which leaves out the $v^a_{nm}\Delta^b_{nm}/\omega_{nm}$ term that the
code includes). The sign of the curvature term is pinned by
`test_generalized_derivative_matches_covariant_finite_differences`, which compares $r^b_{nm;a}$ with
the k-derivative of $r^b_{nm}$ in a parallel-transport gauge (Richardson-extrapolated central
differences).

**Responses** (`nonlinear_optical_spectra`). The formulas are on the theory page:
[shift current](../learn/theory/nonlinear-optics.md#shift-current),
[injection current](../learn/theory/nonlinear-optics.md#injection-current),
[SHG](../learn/theory/nonlinear-optics.md#second-harmonic-generation). In the code
$\int[dk] \to \frac{1}{N_k\Omega}\sum_k$ (`norm = spin_factor / (nkd * cell_volume)`), and the
kernels map to accumulators and files as

| quantity | kernel, accumulator | file, unit (3D) |
|---|---|---|
| shift current $\sigma^{abc}$ | `accumulate_shift_and_injection`, `out['shift']` | `shift_abc.dat`, µA/V² |
| magnetic shift current $\sigma_M^{abc}$ | same, `out['shiftm']` | `shiftm_abc.dat`, µA/V² |
| circular injection $\eta_C^{abc}$ | same, `out['injectc']` | `injectc_abc.dat`, A V⁻² s⁻¹ |
| magnetic injection $\eta_M^{abc}$ | same, `out['injectm']` | `injectm_abc.dat`, A V⁻² s⁻¹ |
| SHG $\chi^{abc}$ | `accumulate_shg_histograms`, `hist1`/`hist2` → `shg_from_histograms` | `shgr/shgi/shga_abc.dat`, pm/V |

In `accumulate_shift_and_injection` the occupation array is `F[k, n, m]` $= f_m - f_n$, and only
pairs with $F > 0$, $\omega'_{nm} > 0$ and $\omega'_{nm}$ within five widths of the photon grid are
kept. The SI prefactors per unit of the dimensionless k-sums are `C_SHIFT` $= \pi e^2/\hbar =
7.647\times10^{-4}$ A/V², `C_INJECT` $= 2\pi e^3/\hbar^2 = 2.324\times10^{12}$ A V⁻² s⁻¹ and
`C_SHG` $= e/(2\epsilon_0) = 9048$ pm/V. The field convention and the factor 2 against Sipe and
Shkrebtii are in
[Second-order response to an optical field](../learn/theory/nonlinear-optics.md#second-order-response-to-an-optical-field).

**δ-functions** (`delta_kernel`; physics in
[Broadening](../learn/theory/nonlinear-optics.md#broadening)) are evaluated directly on the
photon-energy grid for the pairs inside the window, in pair blocks.

**SHG histograms.** Every term of $\chi_e + \chi_i$ has the form $W/(E - c\,\omega - i\eta)$ with $E$
a transition energy and $c \in \{1, 2\}$ (one common $i\eta$ in every denominator, the $1/\tau$ of Chen
*et al.*). The complex weights are accumulated per component into two histograms of $E$ (bin `shg_bin`,
default `eta/5`, linear two-point binning) and convolved once with the resonant denominators on the
photon grid. The cost per k-point is $O(n_{\rm awf}^3)$ for the three-band term and independent of the
number of photon energies; `test_shg_histograms_match_direct_evaluation` checks it against an explicit
evaluation.

**Scissor** ([Scissor shift](../learn/theory/nonlinear-optics.md#scissor-shift)). The conduction bands are those with index
$\geq$ `nocc`; `scissor_pair` shifts $\omega'_{nm}$ of the δ-kernels and `energies_scissored`
the energies of the SHG kernels (resonances and the explicit $1/\omega_{mn}$ factors), while
`band_geometry` runs on the unshifted energies. Test:
`test_scissor_shifts_the_delta_spectra_rigidly`.

**Occupations** (`occupations`; [Occupations and spin](../learn/theory/nonlinear-optics.md#occupations-and-spin)). The band
edges and the default mid-gap Fermi level are taken from the eigenvalues on the coarse grid of
`HRs`; `nelec / spin_factor` must be an integer at 0 K without `fermi_energy`.

**Degeneracies (U(N)-covariant since October 28 2026).** Physics in
[Degenerate bands](../learn/theory/nonlinear-optics.md#degenerate-bands). Bands are grouped in clusters by the
transitive closure of consecutive gaps below `degeneracy_tol` (`degenerate_clusters`); $r$, $G$
and the band sums are restricted to pairs in different clusters, and every quantity is invariant
under a change of basis inside a cluster. The band velocities enter only through the intra-cluster
blocks $v^a_{\rm intra}$ of $v^a$: the sum rule reads
$r^b_{nm;a} = (i/\omega_{nm})\big[[v^a_{\rm intra}, v^b]_{nm}/\omega_{nm} - w^{ab}_{nm} + [v^b, G^a]_{nm}\big]$
(from $(\omega r^b)_{;a} = v^a_{NN} r^b - r^b v^a_{MM} + \omega_{NM} r^b_{;a}$ for the clusters
$N \ni n$, $M \ni m$), and $\Delta^a_{mn} r^b_{mn}$ of the injection current and of the SHG terms
$x_3$, $t_5$ is replaced by $(v^a_{\rm intra} r^b - r^b v^a_{\rm intra})_{mn}$
(`BandGeometry.position_delta`). The diagonal form $v^a_{nn} - v^a_{mm}$ is recovered when every
cluster has one band. Tests: random U(2) rotations inside the exact doublets of a folded supercell
leave the shift, injection and SHG weights of a k-point unchanged ($10^{-10}$), and the
x-supercell on a 4³ grid reproduces the primitive cell on an (8, 4, 4) grid, doublets included.
GaAs (norm-conserving, k-resolved test at a Λ point): the SHG histograms went from 2.5 % / 18 %
differences between symmetry-equivalent k to $10^{-10}$.

**Position dipoles (since October 28 2026).** Physics in
[Position dipoles](../learn/theory/nonlinear-optics.md#position-dipoles). `position_dipoles=True` (or `'intra-atomic'`) goes
beyond the tight-binding position operator: with the Löwdin dipoles $d(\mathbf k)$ of
`hamiltonian/position_dipoles.py` the position operator is $\tau + d$, the velocity gains
$i[H, d]$ (so $r = r^{TB} + D$ between clusters, $D = U^\dagger d U$) and the covariant derivative
becomes

$$
r^b_{nm;a} = r^{TB,b}_{nm;a} + (U^\dagger \tilde\partial_a d^b U)_{nm} + [G^{TB,a}, D^b]_{nm}
- i[D^a_{\rm intra}, r^b]_{nm},
$$

with $\tilde\partial_a d = \partial_a d + i(\tau_j - \tau_i)_a d_{ij}$ the atomic-gauge derivative
of the dipoles in the lattice basis (the same gauge term as in the velocity), $G^{TB}$ from the
tight-binding velocity and $D_{\rm intra}$ the intra-cluster blocks of $D$ (the diagonal dipole
shifts the Berry connection). The gradient is taken by central differences of the dipole
callable (step $10^{-4}$ bohr$^{-1}$; Richardson check on GaAs $3\times10^{-5}$ of a gradient
of 15 bohr²; the dipoles are periodic in k to $10^{-11}$ and continuous across the fold), which
costs six extra dipole evaluations per k-block. Tests: the finite-difference derivative of
$r^{TB} + D$ in a parallel-transport gauge (with the Berry-connection shift of the diagonal
dipoles), invariance under orbital unitaries applied to $H$ and $d$ together, and the U(N)
covariance inside exact doublets with dipoles.

**PAW augmentation of the dipoles (October 28 2026).** With ultrasoft / PAW pseudopotentials
the PAO orbitals are pseudo functions; the position matrix of their all-electron counterparts
carries the partial-wave correction
$\sum_{a,ij}\langle\tilde\varphi_\mu|\tilde p^a_i\rangle[\langle\phi^a_i|\mathbf r|\phi^a_j\rangle
- \langle\tilde\phi^a_i|\mathbf r|\tilde\phi^a_j\rangle]\langle\tilde p^a_j|\tilde\varphi_\nu\rangle$
and the overlap the analogous $q^a_{ij}$ term. `position_dipole_operator(augmentation='auto')`
adds both to $S(\mathbf k)$, $\partial_{\mathbf k}S$ and $X(\mathbf k)$ before the Löwdin formula
(`ProjectorAugmentation`, `build_projector_augmentation`, `augmentation_kspace` in
`hamiltonian/position_dipoles.py`): the partial waves come from `PP_FULL_WFC` of the UPF
(parsed as `UPF.aewfc`, `UPF.pswfc_partial`), the radial dipole integrals
$\int(u_iu_j - \tilde u_i\tilde u_j)\,r\,dr$ are combined with the angular factors
$\langle Y_{l_im}|\hat r_\alpha|Y_{l_jm'}\rangle$ (sphere quadrature of the two-centre module),
and the projector-orbital overlaps are two-centre blocks of the same machinery, Bloch-summed
with the lattice-gauge phases of the tables. Ultrasoft files without partial waves get the
overlap term only, with a warning. Test: the on-site $\langle 4s|r|4p\rangle$ of Ga
reconstructed from the pseudo-orbitals and the partial waves equals the pseudo value plus the
augmentation to machine precision (2.487 bohr against 2.547 without the correction, and 2.490
from the independent all-electron table), and the module's on-site $\Delta S$, $\Delta X$ of an
isolated Ga atom reproduce these radial values ($S + \Delta S = 1$ for the pseudo-orbitals). In
the GaAs crystal the two-centre $\Delta S(\mathbf k)$ agrees with the plane-wave
$\langle\varphi|S|\varphi\rangle - \langle\varphi|\varphi\rangle$ to $7\times10^{-3}$. GaAs
with the 23-band PAW basis: $\chi_{xyz}(0)$ 158 → 193 pm/V with the augmentation (norm-conserving
178, measured 166–180). With an all-electron-only basis $S + \Delta S$ of the crystal is nearly
singular and the Löwdin dipoles blow up; use the UPF-orbital basis route.

**Symmetry reduction (since October 28 2026).** `symmetry=True` (default) reduces the dense
grid to its irreducible wedge under the point-group operations read from Quantum ESPRESSO
(`arrays['sym_rot']`, integer matrices acting on fractional k, the time-reversal-combined ones
excluded; a list of Cartesian matrices can be given for models) and symmetrises the spectra
once at the end; why this is exact is in
[Symmetry reduction of the k-mesh](../learn/theory/nonlinear-optics.md#symmetry-reduction-of-the-k-mesh). The wedge accumulates the
closure of the requested components under the group (`utils/point_groups.py`:
`reduce_k_grid`, `component_closure`, `symmetrize_components`); GaAs with `xyz, yzx, xxx`
needs 9 of 27 components on 1/24 of the grid. The operations are checked on the coarse grid
(band energies invariant to $10^{-5}$ eV, else an error). Time reversal is not used for the
reduction. Test: gapped graphene (D3h, 12 operations), wedge vs full 18² grid identical to
$10^{-9}$ of the scale for every kind; GaAs 8³: identical to $10^{-10}$.

**Spin and 2D** ([Occupations and spin](../learn/theory/nonlinear-optics.md#occupations-and-spin),
[Bulk and sheet units](../learn/theory/nonlinear-optics.md#bulk-and-sheet-units)). With `nspin = 2` each channel is computed
separately (`_0`, `_1` files) and the unsuffixed files hold the sum. `write_nonlinear_outputs`
applies the slab height (`slab_height_bohr`) to every response when `twoD=True`.

## Validation (unit tests, `tests/unit/response/test_nonlinear_optics.py`)

| test | model | checks |
|---|---|---|
| velocity and curvature operators | random two-site model, honeycomb | $D^\dagger v D$, $D^\dagger w D$ equal the exact atomic-gauge operators and their finite differences |
| generalized derivative | random model without symmetries | sum rule vs parallel-transport finite differences, $10^{-5}$ relative |
| gauge invariance | random model | spectra unchanged under unitaries mixing orbitals on a site |
| Berry curvature | Haldane model | $-2\mathrm{Im}\sum_c r^x_{vc}r^y_{cv}$ integrates to the Chern number ($\pm1$, 0) |
| inversion | SSH / Rice–Mele chain | all responses vanish with inversion; odd under the inversion of the chain; invariant under a half-cell translation |
| D3h selection rules | gapped graphene | $\sigma^{yyy} = -\sigma^{yxx} = -\sigma^{xxy} = -\sigma^{xyx}$, $\sigma^{xxx} = 0$, no circular injection, no T-odd responses; the same for $\chi$ |
| time reversal | Haldane with strain | magnetic shift and injection currents appear with the flux and vanish without |
| scissor | gapped graphene | rigid shift of the δ-spectra |
| SHG histograms | random model | equal to the explicit evaluation |
| Kleinman symmetry | strained honeycomb | real $\chi(0)$, $\chi^{xxy} = \chi^{xyx}$ exactly, $\chi^{xxy} = \chi^{yxx}$ to $10^{-3}$ at 48² k |
| units and files | — | prefactors, component parsing, the writer (3D and 2D units) |

Literature targets for the examples: GaAs $\chi_{xyz}$ with a 0.86 eV scissor (Wang *et al.* 2017,
Fig. 4; Rashkeev *et al.* 1998; Nastos *et al.* 2005) and $\sigma_{xyz}$ (Ibañez-Azpiroz *et al.* 2018);
monolayer GeS shift current and SHG (Wang *et al.* 2017, Fig. 1; Rangel *et al.* 2017); the
Rice–Mele shift vector (Fregoso, Morimoto and Moore 2017). The examples are in
`examples/nlo_examples/` (`Graphene`, `GaAs`, `GeS`; the graphene one is explained in the how-to
[How to compute the non-linear optical response of a tight-binding model](../learn/howtos/nonlinear-optics-model.md)),
with the report `examples/nlo_examples/REPORT.md`.

## Performance

Per k-point: 10 complex matrix products (nws × nawf²) for $H$, $\partial H$, $\partial^2 H$, one
`eigh`, the band rotations and the $O(n_{\rm awf}^3)$ sum rule and three-band contraction. For nawf = 26
on a 12³ coarse grid (≈ 2·10³ Wigner–Seitz cells) this is a few ms per k-point per rank; k-points are
split over the MPI ranks (`comm.Allreduce` at the end) and processed in blocks of `kblock`. Knobs:
`components` (the three-band term is evaluated per component), `hopping_cutoff` (drop negligible
cells), `second_harmonic=False` when only the δ-spectra are needed.

## Practical notes from the GaAs validation

- **Degeneracy tolerance.** DFT-derived PAO Hamiltonians have many nearly degenerate pairs
  (semicore and ghost bands). With `degeneracy_tol = 1e-4` eV the symmetry-forbidden
  $\sigma^{xxx}$ of GaAs came out at 30 % of $\sigma^{xyz}$; with 10 meV (the default) the
  leakage is 0.8 % and the allowed components are unchanged. Analytic models can use `1e-4`.
- **Symmetry of the PAO Hamiltonian (PAW / ultrasoft pseudopotentials).** With PAW
  pseudopotentials the GaAs Hamiltonian is symmetric at the grid points to 1e-14 at general k but
  fails by up to 0.65 eV at a few high-symmetry points (16 of 4096), its real-space images are
  symmetric only to 1 meV and the eigenvalues interpolated at an off-grid k spread by 5–25 meV
  over the 24 T_d images, which the $1/\omega$ factors amplify: equivalent components
  ($\chi_{xyz}$ vs $\chi_{yzx}$) differ by 5 % at 40³ and 2–4 % at 60³, and a no-symmetry nscf
  does **not** change this. The cause is the Löwdin re-orthonormalisation of the PAW
  pseudo-wavefunctions in the internal projections (the raw projections and the `projwfc.x`
  ones obey the symmetry to 1e-10; the symmetry expansion, the Wigner–Seitz images, the band
  selection and the Hamiltonian construction were ruled out, see `examples/nlo_examples/REPORT.md`
  §3.2). With the PseudoDojo norm-conserving Ga/As set the same basis gives projections
  symmetric to 1e-10, H(k) symmetric to 3e-9 eV at every grid point, an interpolation spread of
  3e-8 meV and a real-space Hamiltonian 15 times better localised. **Fixed** (October 2026):
  the internal projections now use the augmented metric $S = 1 + \sum q_{ij}|\beta_i\rangle\langle\beta_j|$
  for ultrasoft / PAW pseudopotentials ([internal-basis.md](internal-basis.md)); the PAW GaAs
  Hamiltonian is then symmetric too (spread 3e-3 meV). The projectabilities changed with it
  (the old values were inflated): with the all-electron tables alone 12 GaAs bands pass 0.95;
  the default basis route for PAW inputs now takes the valence shells from the UPF
  pseudo-orbitals and only the augmenting shells from the tables (`pseudo_orbitals='auto'`,
  [internal-basis.md](internal-basis.md)), which gives 23.
- **Degenerate bands on symmetry lines.** With an exactly symmetric Hamiltonian the shift
  current obeys the T_d relations to 1e-10, but $\chi_{xyz}$ and $\chi_{yzx}$ still differ by
  10 % (1–2 eV) and 2 % (2–4 eV) at 40³, and the circular injection current, forbidden in T_d,
  is not zero: the SHG and injection kernels are gauge-dependent for exactly degenerate bands
  (the Λ-line points of the grid, where `eigh` returns an arbitrary basis of the doublets;
  k-resolved test in the GaAs validation folder: 18 % at a Λ point, 1e-10 elsewhere), the
  shift-current kernel is not. The error decreases with the grid density (170 % at 8³) and does
  not depend on the degeneracy tolerance or the histogram bin. The U(N)-covariant treatment of
  the degeneracies (above) removes this dependence.
- **GaAs numbers** (PAW, symmetric 16³ nscf, scissor 0.86 eV, η = 0.05 eV):
  $|\chi_{xyz}(0)| = 110$ pm/V at both 40³ and 60³, 341 → 337 pm/V at 1.5 eV, peak 610 (40³) →
  480 (60³) pm/V near 2.6 eV (the resonance region is not converged at 60³), against
  100–170 pm/V and 400–600 pm/V in the literature (Nastos 2005, Wang 2017, experiment
  $2d_{14} \approx 170$ pm/V); $\sigma_{xyz}$ reaches $-30$ (40³) / $-27$ (60³) µA/V² at 3.9–4 eV.
  The T-odd responses and the circular injection are $10^{-5}$ of their natural scale. With the
  current code (augmented projections, covariant kernels) and the position dipoles the static
  $\chi_{xyz}$ is 178 pm/V (norm-conserving) / 156 pm/V (PAW) against 103 / 120 pm/V with the
  tight-binding position operator: the dipoles close most of the gap to the measured 166–180 pm/V
  (`examples/nlo_examples/REPORT.md` §3.2).
- **GeS numbers** (monolayer, nosym 18×18×1 nscf, 200² dense grid, η = 0.05 eV, PBE gap 1.68 eV,
  sheet units with the 18 Å cell): shift-current peaks $\sigma^{xxx} = 46$, $\sigma^{xyy} = 48$,
  $\sigma^{yxy} = 26$ nm µA/V² (about 75 µA/V² for a 0.6 nm layer, the order of magnitude of Rangel
  *et al.* 2017), the mirror-forbidden $yyy$ and $xxy$ below $10^{-8}$, T-odd files identically zero.
  The repository BASIS tables projected the Ge 3d bands at 0.90 and gave a gapless PAO Hamiltonian;
  the tables generated from the pseudopotentials (`paoflow-genbasis-ps`, preset `standard`) give 26
  bands above 0.95.
- **Basis.** The 26-orbital Ga/As basis (no 4D/5D/6S/6P) projects the Ga 3d semicore bands
  poorly (projectability 0.45) and cannot be used; the 62-orbital configuration of example10
  (18 bands above 0.95) is the working one. Cost at nawf = 62, 40³ k, three components with
  SHG: 15 min on 8 ranks.
- **Orbital centres of the built-in graphene model.** `models.graphene` stores `tau` rounded to
  five decimals (0.28867, 0.57735 instead of $1/(2\sqrt3)$, $1/\sqrt3$), which breaks the
  threefold rotation of the position operator at the $10^{-5}$ level. On the 300² run of the
  how-to the forbidden circular injection `injectc_xxy` reaches $8.6\times10^4$ nm A V⁻² s⁻¹
  (150²: $1.1\times10^5$, 600²: $8.5\times10^4$) and $|\sigma^{yyy}+\sigma^{yxx}|$ is
  $7\times10^{-4}$ nm µA/V²; with `orbital_positions=[[1/3, 1/3, 0], [2/3, 2/3, 0]]` they drop to
  $2\times10^{-4}$ and $10^{-13}$.
- **Memory.** The δ-kernels are evaluated in pair blocks of about 16 MB and the three-band SHG
  term in k-chunks of about 2·10⁶ triples; the Wigner–Seitz images of `HRs` (nws × nawf² complex)
  are the largest resident array (300 MB for GaAs at 16³).

## Not implemented (follow-ups)

The position dipoles, the U(N)-covariant derivative and the symmetry reduction of
[nonlinear-optics-phase2-plan.md](nonlinear-optics-phase2-plan.md) are implemented (above);
k-resolved integrand maps and a Rust kernel for the post-`eigh` algebra are not.

## References

- C. Wang, X. Liu, L. Kang, B.-L. Gu, Y. Xu and W. Duan, *First-principles calculation of nonlinear optical responses by Wannier interpolation*, Phys. Rev. B **96**, 115147 (2017).
- H. Chen, M. Ye, N. Zou, B.-L. Gu, Y. Xu and W. Duan, *Basic formulation and first-principles implementation of nonlinear magneto-optical effects*, Phys. Rev. B **105**, 075123 (2022).
- J. E. Sipe and A. I. Shkrebtii, *Second-order optical response in semiconductors*, Phys. Rev. B **61**, 5337 (2000).
- C. Aversa and J. E. Sipe, Phys. Rev. B **52**, 14636 (1995); S. N. Rashkeev, W. R. L. Lambrecht and B. Segall, Phys. Rev. B **57**, 3905 (1998); F. Nastos, B. Olejnik, K. Schwarz and J. E. Sipe, Phys. Rev. B **72**, 045223 (2005).
- J. Ibañez-Azpiroz, S. S. Tsirkin and I. Souza, *Ab initio calculation of the shift photocurrent by Wannier interpolation*, Phys. Rev. B **97**, 245143 (2018).
- B. M. Fregoso, T. Morimoto and J. E. Moore, Phys. Rev. B **96**, 075421 (2017); T. Rangel *et al.*, Phys. Rev. Lett. **119**, 067402 (2017).
